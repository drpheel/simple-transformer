"""
A minimal Transformer built from scratch — both decoder-only AND encoder-decoder.

Read this file top-to-bottom. Building blocks:

  1. SelfAttention       — Q/K/V from the same sequence (causal OR bidirectional)
  2. CrossAttention      — Q from decoder, K/V from encoder  ← how the two halves talk
  3. FeedForward         — per-token MLP
  4. EncoderBlock        — bidirectional self-attn + MLP
  5. DecoderBlock        — causal self-attn + cross-attn + MLP
  6. MiniGPT             — decoder-only language model (GPT-style)
  7. EncoderDecoderLM    — full seq2seq Transformer (original paper style)

Nothing here uses torch.nn.MultiheadAttention. We implement attention
explicitly so you can see the matrix multiplies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """
    Multi-head self-attention — the Q / K / V dance on ONE sequence.

    Mental model (keep this straight):
      • Input x is CONTENT: shape (T, C) — one feature vector per token.
        That is NOT the T×T association table. The T×T table is *derived*.
      • Query asks, Key answers the match, Value is what gets delivered.
      • Pipeline each forward:
            content (T,C) → cast Q/K/V → match Q vs K → T×T weights
                        → gather V → new content (T,C)

    Roles for each token (same x, three learned projections):
      Q (query)  — "what am I looking for right now?"   (the seeker)
      K (key)    — "what do I advertise as my tag?"     (the matchable label)
      V (value)  — "what payload do I hand over if chosen?"

    causal=True  → decoder / GPT: position i may only see positions <= i
    causal=False → encoder: every position can see the whole sequence
    """

    def __init__(
        self,
        n_embd: int,       # model width (token embedding size) = C = "cargo" width
        n_head: int,       # how many parallel attention heads (parallel dances)
        block_size: int,   # max sequence length (needed to pre-build the causal mask)
        dropout: float,    # dropout probability for attention + residual path
        causal: bool = False,  # True = GPT/decoder (no peeking ahead); False = encoder
    ):
        super().__init__()  # register this as a proper nn.Module
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"  # each head gets equal slice of dims

        self.n_head = n_head  # stash head count for reshape in forward
        self.head_dim = n_embd // n_head  # width of one head (= d_k in the paper)
        self.causal = causal  # remember whether we should apply the future-mask

        # One Linear → 3*C outputs is just W_Q, W_K, W_V stacked for speed.
        # Same input x is cast into three roles: seeker / tag / payload.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # After heads remix content, this mixes the concatenated heads together
        # so information from different heads can interact.
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)  # randomly zero some attention weights (regularization)
        self.resid_drop = nn.Dropout(dropout)  # dropout on the residual stream output

        if causal:
            # Lower-triangular routing mask: 1 = allowed, 0 = blocked (future tokens).
            # Only built for decoders — encoders don't need it (full bidirectional look).
            mask = torch.tril(torch.ones(block_size, block_size))  # 1s on/below diagonal, 0s above
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))  # non-param; moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------
        # INPUT = CONTENT (cargo), not relationships.
        #   x: (B, T, C)  — B batches, T tokens, C = n_embd feature channels.
        # Each row is "who is this token / what does it mean right now?"
        # (token embed + position already baked in by the calling model).
        # ------------------------------------------------------------------
        B, T, C = x.shape

        # ------------------------------------------------------------------
        # SCENE 1 — CAST THE ROLES (derive Q, K, V from the same x)
        #
        # Searching, being-findable, and handing-over-info are different jobs,
        # so we learn three remappings of the same content:
        #   Q = x @ W_Q   → seeker   ("what would be useful for me?")
        #   K = x @ W_K   → tag      ("what am I about, for matching?")
        #   V = x @ W_V   → payload  ("what do I actually give you?")
        #
        # Self-attention: Q, K, V all come from THIS sequence (hence "self").
        # Packed into one matmul for efficiency, then split.
        # ------------------------------------------------------------------
        qkv = self.qkv(x)  # (B, T, 3*C) — Q, K, V concatenated on the channel axis
        q, k, v = qkv.split(C, dim=-1)  # three tensors, each (B, T, C)

        # ------------------------------------------------------------------
        # MULTI-HEAD LAYOUT
        #
        # Split the C channels into n_head independent subspaces of size
        # head_dim (C = n_head * head_dim). Each head runs its own Q/K/V dance
        # (e.g. one head might track syntax, another nearby words, etc.).
        #
        #   view:      (B, T, C) → (B, T, n_head, head_dim)
        #   transpose: (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
        #
        # Why transpose? So the next matmul treats (batch, head) like a big
        # batch dim and multiplies over the last two dims (T × head_dim).
        # ------------------------------------------------------------------
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # seekers per head
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # tags per head
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # payloads per head

        # ------------------------------------------------------------------
        # SCENE 2 — MATCHING (queries meet keys) → builds the T×T table
        #
        # THIS is where relationships appear. T×T is NOT the key matrix;
        # keys are still (T, head_dim). T×T is the match table:
        #   score[i, j] = Q_i · K_j / √d
        #   row i = "how well does each key match what query i is looking for?"
        #
        # k.transpose(-2, -1): (..., T, d) → (..., d, T)
        # so (..., T, d) @ (..., d, T) → (..., T, T) in one GPU matmul.
        #
        # Scale by 1/√head_dim so dots don't grow with d (keeps softmax soft).
        # Values are NOT used yet — matching is only Q vs K.
        # ------------------------------------------------------------------
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)

        if self.causal:
            # Ban future keys BEFORE softmax so training matches generation:
            # at position i you must not route to tokens that don't exist yet.
            # -inf → after softmax those weights become 0.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over keys (last dim): each query's row becomes a budget of
        # mixing weights that sum to 1 — "who should I listen to, and how much?"
        att = F.softmax(att, dim=-1)  # still (B, n_head, T, T), now probabilities
        att = self.attn_drop(att)  # randomly drop some routes so we don't overfit one path

        # ------------------------------------------------------------------
        # SCENE 3 — GATHERING (attention weights meet values)
        #
        # Routing table buys payloads:
        #   output_i = Σ_j  att[i, j] * V_j
        #
        # att said WHO to listen to (from Q·K). V is WHAT you actually hear.
        # K and V are separate so you can match on one aspect and deliver another.
        # Shape: (B, n_head, T, T) @ (B, n_head, T, head_dim)
        #      → (B, n_head, T, head_dim)  — new content per token, per head.
        # ------------------------------------------------------------------
        y = att @ v

        # ------------------------------------------------------------------
        # SCENE 4 — WRITE-BACK to a single content vector per token
        #
        # Merge heads: (B, n_head, T, head_dim) → (B, T, C)
        # contiguous() needed after transpose so view() is valid.
        # proj = let heads talk to each other after concat.
        # Caller will usually do: x = x + attn(x)  (residual = keep old cargo too).
        # ------------------------------------------------------------------
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention — same Q/K/V dance, but across TWO sequences.

    Same roles as self-attention:
      Q asks, K answers the match, V is delivered.
    Difference: the seeker and the library are different people.
      Q  ← decoder states x      ("what am I trying to generate right now?")
      K,V ← encoder memory       ("what did the source say / what can I deliver?")

    So the match table is (T_dec × T_enc), not (T × T) on one sequence.
    No causal mask: every decoder position may read every encoder position.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()  # register as nn.Module
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Separate projections because Q and K/V come from different tensors.
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)  # decoder → Query (seeker)
        self.kv_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)  # encoder → Key+Value packed
        self.proj = nn.Linear(n_embd, n_embd, bias=False)  # mix heads after gathering
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # x:      decoder CONTENT  (B, T_dec, C)  — the seekers live here
        # memory: encoder CONTENT  (B, T_enc, C)  — the tags + payloads live here
        B, T, C = x.shape
        S = memory.size(1)  # encoder length (can differ from T)

        # SCENE 1 — cast roles from the RIGHT sequences
        q = self.q_proj(x)  # Query from decoder only: "what do I need from the source?"
        k, v = self.kv_proj(memory).split(C, dim=-1)  # Key+Value from encoder: tag + payload

        # Same multi-head layout as self-attn (heads next to batch for matmul).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_dec, d)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_enc, d)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_enc, d)

        # SCENE 2 — matching: each decoder query scores every encoder key.
        # Still NOT using V yet.
        #
        # q:              (B, n_head, T_dec, d)
        # k.transpose:    (B, n_head, d, T_enc)   ← swap last two dims of k
        # q @ k^T:        (B, n_head, T_dec, T_enc) — one score per (decoder pos, encoder pos)
        # / sqrt(d):      keeps dot products from blowing up as d grows (softmax stays soft)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # softmax over T_enc: each decoder row becomes a distribution over encoder positions
        # (sums to 1 along the last dim). Shape unchanged: (B, n_head, T_dec, T_enc).
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)  # randomly zero some weights; remaining stay as-is (no renorm)

        # SCENE 3 — gathering: pull encoder payloads into decoder positions
        y = att @ v  # (B, n_head, T_dec, head_dim)

        # SCENE 4 — write-back to decoder content width
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """
    Position-wise MLP — the transformer's per-token "thinking" step.

    Attention mixes information *across* tokens; this MLP then transforms each
    token's representation *on its own*. Without it, each layer would mostly
    reshuffle existing features. The expand→nonlinearity→contract pattern lets
    the model learn richer feature interactions than a single linear map could.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()  # register as nn.Module
        self.net = nn.Sequential(
            # Widen first so the nonlinearity has room to carve complex features;
            # 4× is the original Transformer default (cheap capacity boost).
            nn.Linear(n_embd, 4 * n_embd),
            # Nonlinearity is required: stacked linears alone collapse to one linear.
            # GELU is smoother than ReLU and is the usual modern default here.
            nn.GELU(),
            # Collapse back to n_embd so residual add / next layer shapes still match.
            nn.Linear(4 * n_embd, n_embd),
            # Dropout on the MLP output to regularize this high-capacity sublayer.
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same weights for every position — no mixing across the sequence axis.
        return self.net(x)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class EncoderBlock(nn.Module):
    """
    Encoder layer:
      LayerNorm → Bidirectional Self-Attention → residual
      LayerNorm → FeedForward                   → residual

    Bidirectional = every token sees the entire input (no causal mask).
    That is what lets the encoder build a rich representation of the source.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()  # register as nn.Module
        self.ln1 = nn.LayerNorm(n_embd)  # pre-norm before self-attention
        self.attn = SelfAttention(n_embd, n_head, block_size, dropout, causal=False)  # bidirectional self-attn
        self.ln2 = nn.LayerNorm(n_embd)  # pre-norm before MLP
        self.ffn = FeedForward(n_embd, dropout)  # position-wise feed-forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # self-attn residual: mix info across source tokens
        x = x + self.ffn(self.ln2(x))  # MLP residual: nonlinear transform per token
        return x  # return updated encoder states


class DecoderBlock(nn.Module):
    """
    Decoder layer (full encoder-decoder style):
      1. Causal self-attention  — look at already-generated tokens only
      2. Cross-attention        — look at encoder memory
      3. Feed-forward           — per-token MLP

    Compare to MiniGPT's DecoderOnlyBlock, which has step 1 + 3 but no step 2.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()  # register as nn.Module
        self.ln1 = nn.LayerNorm(n_embd)  # pre-norm before causal self-attn
        self.self_attn = SelfAttention(n_embd, n_head, block_size, dropout, causal=True)  # look only at past/current tgt tokens
        self.ln2 = nn.LayerNorm(n_embd)  # pre-norm before cross-attn
        self.cross_attn = CrossAttention(n_embd, n_head, dropout)  # read from encoder memory
        self.ln3 = nn.LayerNorm(n_embd)  # pre-norm before MLP
        self.ffn = FeedForward(n_embd, dropout)  # per-token MLP

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))  # causal self-attn residual (tgt ↔ tgt)
        x = x + self.cross_attn(self.ln2(x), memory)  # cross-attn residual (tgt ↔ src memory)
        x = x + self.ffn(self.ln3(x))  # MLP residual
        return x  # return updated decoder states


class DecoderOnlyBlock(nn.Module):
    """GPT-style block: causal self-attn + MLP (no encoder / no cross-attn)."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()  # register as nn.Module
        self.ln1 = nn.LayerNorm(n_embd)  # pre-norm before attention
        self.attn = SelfAttention(n_embd, n_head, block_size, dropout, causal=True)  # causal self-attn only
        self.ln2 = nn.LayerNorm(n_embd)  # pre-norm before MLP
        self.ffn = FeedForward(n_embd, dropout)  # per-token MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # causal self-attn residual
        x = x + self.ffn(self.ln2(x))  # MLP residual
        return x  # return updated hidden states


# Alias kept so older mental-model names still match the code.
TransformerBlock = DecoderOnlyBlock  # historical name → same GPT block


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):  # initialize Linear layers
        nn.init.normal_(module.weight, mean=0.0, std=0.02)  # small Gaussian for stable start
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # biases start at zero
    elif isinstance(module, nn.Embedding):  # initialize Embedding tables
        nn.init.normal_(module.weight, mean=0.0, std=0.02)  # same small Gaussian


class MiniGPT(nn.Module):
    """
    Tiny character-level language model (decoder-only).

      tokens → embed + position
            → N × DecoderOnlyBlock
            → LayerNorm → LM head → next-token logits
    """

    def __init__(
        self,
        vocab_size: int,     # number of distinct tokens (e.g. characters)
        n_embd: int = 128,   # embedding / hidden size
        n_head: int = 4,     # attention heads per block
        n_layer: int = 4,    # how many DecoderOnlyBlocks to stack
        block_size: int = 64,  # max context length
        dropout: float = 0.1,  # dropout rate
    ):
        super().__init__()  # register as nn.Module so parameters/submodules are tracked
        self.block_size = block_size  # store max context; generate crops to this, forward asserts against it

        # embeddings: tokens alone are permutation-invariant, so we add learned positions
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # map discrete token ids into continuous vectors the rest of the net can process
        self.pos_emb = nn.Embedding(block_size, n_embd)  # inject order info (attention has no inherent notion of position)
        self.drop = nn.Dropout(dropout)  # regularize early so the model can't overfit to specific token+pos combinations
        self.blocks = nn.ModuleList(
            [
                DecoderOnlyBlock(n_embd, n_head, block_size, dropout)  # one residual attn+MLP stage
                for _ in range(n_layer)  # depth lets each layer refine representations using earlier context
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # stabilize final hidden states before the large vocab projection
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # score every vocab token as the next-token prediction
        self.head.weight = self.tok_emb.weight  # weight tying: fewer params + forces input/output spaces to stay aligned
        self.apply(_init_weights)  # small-Gaussian init keeps early activations/grads from exploding or vanishing


    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape  # batch size and sequence length of token ids
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"  # refuse overlong context

        # --- embed ---
        pos = torch.arange(T, device=idx.device)  # position indices 0..T-1 on same device as idx
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # token embed + pos embed, then dropout

        # --- transform ---
        for block in self.blocks:
            x = block(x)  # run each DecoderOnlyBlock in sequence

        # --- project ---
        logits = self.head(self.ln_f(x))  # LayerNorm then project to vocab-sized logits per position

        # --- loss ---
        loss = None  # default: inference / no targets
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # flatten B*T for next-token CE loss
        return logits, loss  # always return logits; loss only when training

    @torch.no_grad()  # disable grad tracking for sampling (faster / less memory)
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):  # append one token at a time
            idx_cond = idx[:, -self.block_size :]  # crop to last block_size tokens (context window)
            logits, _ = self(idx_cond)  # forward pass → logits for every position
            logits = logits[:, -1, :] / max(temperature, 1e-8)  # take last position; divide by temperature (higher = flatter)
            probs = F.softmax(logits, dim=-1)  # convert logits → sampling distribution
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token id from that distribution
            idx = torch.cat([idx, next_id], dim=1)  # append sampled token to the sequence
        return idx  # return the full (prompt + generated) sequence


class EncoderDecoderLM(nn.Module):
    """
    Full encoder-decoder Transformer (original "Attention Is All You Need" shape).

    Encoder (source):
      - start with src tokens
      - add token embed + position embed
      - run through N EncoderBlocks
      - result is memory (encoder hidden states)

    Decoder (target, teacher-forced during training):
      - start with tgt tokens
      - add token embed + position embed
      - run through N DecoderBlocks (each attends to memory)
      - LM head → next-token logits

    Pipeline for one training step:
      - src  "hello"     → encoder builds memory of the whole word
      - tgt  "<bos>olle" → decoder predicts "olleh<eos>" one step at a time
                           using causal self-attn + cross-attn into memory
    """

    def __init__(
        self,
        vocab_size: int,      # shared vocab size for src and tgt (char-level here)
        n_embd: int = 128,    # embedding / hidden size
        n_head: int = 4,      # attention heads
        n_layer: int = 3,     # encoder depth AND decoder depth
        block_size: int = 32, # max src / tgt length
        dropout: float = 0.1, # dropout rate
    ):
        super().__init__()  # register as nn.Module
        self.block_size = block_size  # max sequence length for encode/decode/generate

        self.src_tok_emb = nn.Embedding(vocab_size, n_embd)  # source token embeddings
        self.tgt_tok_emb = nn.Embedding(vocab_size, n_embd)  # target token embeddings (separate table)
        self.pos_emb = nn.Embedding(block_size, n_embd)  # shared absolute position embeddings
        self.drop = nn.Dropout(dropout)  # dropout after embed+pos

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(n_embd, n_head, block_size, dropout)  # one encoder layer
                for _ in range(n_layer)  # stack n_layer encoder blocks
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(n_embd, n_head, block_size, dropout)  # one decoder layer (self + cross + ffn)
                for _ in range(n_layer)  # stack n_layer decoder blocks
            ]
        )
        self.ln_enc = nn.LayerNorm(n_embd)  # final norm on encoder memory
        self.ln_dec = nn.LayerNorm(n_embd)  # final norm on decoder states before LM head
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # decoder hidden → vocab logits

        self.apply(_init_weights)  # init all Linear / Embedding weights

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        # src: (B, S)
        S = src.size(1)  # source length
        assert S <= self.block_size  # refuse overlong source
        pos = torch.arange(S, device=src.device)  # position ids for the source
        x = self.drop(self.src_tok_emb(src) + self.pos_emb(pos))  # embed src tokens + positions
        for block in self.encoder:
            x = block(x)  # run each EncoderBlock (bidirectional self-attn + MLP)
        return self.ln_enc(x)  # return normalized encoder memory for the decoder to attend to

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # tgt: (B, T) — already-generated / teacher-forced prefix
        T = tgt.size(1)  # target length so far
        assert T <= self.block_size  # refuse overlong target
        pos = torch.arange(T, device=tgt.device)  # position ids for the target
        x = self.drop(self.tgt_tok_emb(tgt) + self.pos_emb(pos))  # embed tgt tokens + positions
        for block in self.decoder:
            x = block(x, memory)  # each DecoderBlock: causal self-attn + cross-attn into memory + MLP
        return self.head(self.ln_dec(x))  # LayerNorm then LM head → logits over vocab

    def forward(
        self,
        src: torch.Tensor,  # source token ids (B, S)
        tgt: torch.Tensor,  # target prefix token ids (B, T), teacher-forced in training
        targets: torch.Tensor | None = None,  # next-token labels aligned with tgt (or None at inference)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory = self.encode(src)  # run encoder once → fixed memory of the source
        logits = self.decode(tgt, memory)  # decode against that memory → (B, T, vocab)

        loss = None  # no loss unless labels provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # flatten predictions to (B*T, vocab)
                targets.view(-1),  # flatten labels to (B*T,)
                ignore_index=-1,  # pad positions (if any) use -1
            )
        return logits, loss  # logits always; loss when training

    @torch.no_grad()  # no gradients during generation
    def generate(
        self,
        src: torch.Tensor,  # source sequence to condition on
        bos_id: int,  # begin-of-sequence token id to start decoding
        eos_id: int,  # end-of-sequence token id that stops decoding
        max_new_tokens: int | None = None,  # optional cap on generated length
    ) -> torch.Tensor:
        """Encode src once, then autoregressively decode until EOS / max length."""
        if max_new_tokens is None:
            max_new_tokens = self.block_size - 1  # default: leave room for BOS within block_size
        memory = self.encode(src)  # encode source once; reuse for every decode step
        # Start every sequence with <bos>
        tgt = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)  # (B, 1) filled with BOS

        for _ in range(max_new_tokens):  # generate up to max_new_tokens tokens
            logits = self.decode(tgt[:, -self.block_size :], memory)  # decode with last block_size tgt tokens
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy: pick highest-logit token
            tgt = torch.cat([tgt, next_id], dim=1)  # append chosen token to the target sequence
            if (next_id == eos_id).all():
                break  # stop early when every sequence in the batch hit EOS
        return tgt  # return BOS + generated tokens (possibly including EOS)
