"""
Train the encoder-decoder Transformer on a toy seq2seq task: reverse a string.

  src:  "hello"       → encoder (bidirectional)
  tgt:  "<bos>olleh"  → decoder predicts "olleh<eos>"

This makes encoder vs decoder vs cross-attention obvious:
  - Encoder sees the whole source at once (no causal mask).
  - Decoder generates one char at a time (causal self-attn).
  - Cross-attn lets each decoder step read from encoder memory.

Usage:
  python train_seq2seq.py
  python train_seq2seq.py --steps 3000
"""

from __future__ import annotations

import argparse
import random
import string
import time
from pathlib import Path

import torch

from model import EncoderDecoderLM

# Special tokens (PAD kept in vocab for completeness; this toy uses fixed lengths)
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
ALPHABET = string.ascii_lowercase  # a-z only for a clean toy task


def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    tokens = [PAD, BOS, EOS] + list(ALPHABET)
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def make_example(stoi: dict[str, int], seq_len: int) -> tuple[list[int], list[int], list[int]]:
    """
    One reverse-string example → (src, tgt_in, tgt_out).

    Fixed length (no padding) keeps the learning example clear:
      src     = h e l l o
      tgt_in  = <bos> o l l e
      tgt_out =       o l l e h <eos>
    """
    chars = [random.choice(ALPHABET) for _ in range(seq_len)]
    src = [stoi[c] for c in chars]
    reversed_chars = list(reversed(chars))
    # Teacher forcing: decoder input is <bos> + target[:-1]
    #                  decoder target is everything after <bos> (ends with <eos>)
    tgt_full = [stoi[BOS]] + [stoi[c] for c in reversed_chars] + [stoi[EOS]]
    tgt_in = tgt_full[:-1]
    tgt_out = tgt_full[1:]
    return src, tgt_in, tgt_out


def get_batch(
    stoi: dict[str, int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [make_example(stoi, seq_len) for _ in range(batch_size)]
    srcs, tgts_in, tgts_out = zip(*examples)
    return (
        torch.tensor(srcs, dtype=torch.long, device=device),
        torch.tensor(tgts_in, dtype=torch.long, device=device),
        torch.tensor(tgts_out, dtype=torch.long, device=device),
    )


@torch.no_grad()
def accuracy(
    model: EncoderDecoderLM,
    stoi: dict[str, int],
    n: int,
    seq_len: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    for _ in range(n):
        src_ids, _, tgt_out = make_example(stoi, seq_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        pred = model.generate(src, bos_id=stoi[BOS], eos_id=stoi[EOS])
        # Drop leading <bos>; compare until <eos>
        pred_ids = pred[0, 1:].tolist()
        if stoi[EOS] in pred_ids:
            pred_ids = pred_ids[: pred_ids.index(stoi[EOS])]
        gold_ids = [t for t in tgt_out if t != stoi[EOS]]
        correct += int(pred_ids == gold_ids)
    model.train()
    return correct / n


def decode_ids(ids: list[int], itos: dict[int, str], stoi: dict[str, int]) -> str:
    out = []
    for i in ids:
        if i in (stoi[PAD], stoi[BOS]):
            continue
        if i == stoi[EOS]:
            break
        out.append(itos[i])
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder-decoder on reverse-string")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=8, help="source string length")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--ckpt", type=str, default="checkpoints/seq2seq.pt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # src len = seq_len; tgt len = seq_len + 1 (<bos>… or …<eos>)
    assert args.seq_len + 1 <= args.block_size, "need room for <bos>/<eos>"

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    stoi, itos = build_vocab()
    vocab_size = len(stoi)
    print(f"task: reverse a-z strings of length {args.seq_len} | vocab: {vocab_size}")

    model = EncoderDecoderLM(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        src, tgt_in, tgt_out = get_batch(stoi, args.batch_size, args.seq_len, device)
        _, loss = model(src, tgt_in, tgt_out)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            acc = accuracy(model, stoi, n=100, seq_len=args.seq_len, device=device)
            elapsed = time.time() - t0
            print(
                f"step {step:5d}/{args.steps} | loss {loss.item():.3f} | "
                f"exact-match {acc:.0%} | {elapsed:.1f}s"
            )

    root = Path(__file__).resolve().parent
    ckpt_path = root / args.ckpt
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": {
                "vocab_size": vocab_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": args.block_size,
            },
        },
        ckpt_path,
    )
    print(f"saved {ckpt_path}")

    # ---- show a few qualitative examples ----
    model.eval()
    print("\n===== examples (src → pred | gold) =====")
    for _ in range(8):
        src_ids, _, tgt_out = make_example(stoi, args.seq_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        pred = model.generate(src, bos_id=stoi[BOS], eos_id=stoi[EOS])
        src_str = decode_ids(src_ids, itos, stoi)
        pred_str = decode_ids(pred[0].tolist(), itos, stoi)
        gold_str = decode_ids(tgt_out, itos, stoi)
        mark = "OK" if pred_str == gold_str else "XX"
        print(f"  [{mark}] {src_str!r:14s} → {pred_str!r:14s} | {gold_str!r}")
    print("========================================")


if __name__ == "__main__":
    main()
