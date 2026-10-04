# Simple Attention Transformer

A tiny Transformer in PyTorch — **decoder-only** (GPT) and **encoder-decoder** (original paper) — implemented from scratch so you can see attention end to end. No `nn.MultiheadAttention`, no Hugging Face.

## What to read

| File | Role |
|------|------|
| `model.py` | All building blocks — heavily commented |
| `train.py` | Decoder-only char LM (Shakespeare) |
| `train_seq2seq.py` | Encoder-decoder on reverse-a-string |
| `data/tinyshakespeare.txt` | LM corpus |

## Two architectures

### 1. Decoder-only (`MiniGPT`) — how ChatGPT-style models work

```
tokens → embed + pos
      → [causal self-attn + MLP] × N
      → LM head → next token
```

Causal mask: position `i` only sees `≤ i`. There is **no encoder**.

### 2. Encoder-decoder (`EncoderDecoderLM`) — original Transformer / translation

```
src  → embed + pos → [bidirectional self-attn + MLP] × N  → memory
tgt  → embed + pos → [causal self-attn
                      + cross-attn(memory)     ← Q from decoder, K/V from encoder
                      + MLP] × N
                   → LM head → next target token
```

| Piece | Sees | Mask |
|-------|------|------|
| **Encoder self-attn** | whole source | none (bidirectional) |
| **Decoder self-attn** | generated prefix | causal |
| **Cross-attn** | encoder memory | none |

## Run it

```bash
cd ~/simple-transformer

# decoder-only language model
python3 train.py --steps 2000

# full encoder-decoder on reverse-string (learns fast, easy to verify)
python3 train_seq2seq.py
python3 train_seq2seq.py --steps 3000
```

## Suggested learning path

1. **`SelfAttention`** with `causal=True` vs `False` — same math, different mask.
2. **`CrossAttention`** — notice Q is from the decoder, K/V from encoder memory.
3. **`EncoderBlock` vs `DecoderBlock`** — decoder has one extra sub-layer.
4. Trace one reverse-string example in `train_seq2seq.py`:
   - src `"hello"` → encoder memory
   - decoder starts at `<bos>`, predicts `o`, then `l`, … until `<eos>`
5. Ablate: zero out cross-attn output — reverse-string accuracy should collapse.

## Scale

| | This toy | GPT-2 small | Original Transformer |
|--|----------|-------------|----------------------|
| Style | both | decoder-only | encoder-decoder |
| Params | ~0.3–1M | 124M | ~65M |
| Vocab | 65 chars / 29 toys | 50k BPE | 32k wordpieces |
