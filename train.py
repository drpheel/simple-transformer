"""
Train MiniGPT on character-level text and sample from it.

Usage:
  python train.py                  # train + generate a sample
  python train.py --steps 2000     # longer training
  python train.py --sample-only --ckpt checkpoints/model.pt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from model import MiniGPT


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Random starting indices; x is context, y is next-char targets (shifted by 1).
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character Transformer")
    parser.add_argument("--data", type=str, default="data/tinyshakespeare.txt")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model.pt")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    root = Path(__file__).resolve().parent
    data_path = root / args.data if not os.path.isabs(args.data) else Path(args.data)
    ckpt_path = root / args.ckpt if not os.path.isabs(args.ckpt) else Path(args.ckpt)

    text = data_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"corpus: {len(text):,} chars | vocab: {vocab_size}")

    data = encode(text, stoi)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    if args.sample_only:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = MiniGPT(
            vocab_size=ckpt["vocab_size"],
            n_embd=ckpt["n_embd"],
            n_head=ckpt["n_head"],
            n_layer=ckpt["n_layer"],
            block_size=ckpt["block_size"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        stoi, itos = ckpt["stoi"], ckpt["itos"]
    else:
        model = MiniGPT(
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
            x, y = get_batch(train_data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.eval_interval == 0 or step == 1:
                train_loss = estimate_loss(
                    model, train_data, args.batch_size, args.block_size, 20, device
                )
                val_loss = estimate_loss(
                    model, val_data, args.batch_size, args.block_size, 20, device
                )
                elapsed = time.time() - t0
                print(
                    f"step {step:5d}/{args.steps} | "
                    f"train {train_loss:.3f} | val {val_loss:.3f} | {elapsed:.1f}s"
                )

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": args.n_embd,
                "n_head": args.n_head,
                "n_layer": args.n_layer,
                "block_size": args.block_size,
                "stoi": stoi,
                "itos": itos,
            },
            ckpt_path,
        )
        print(f"saved {ckpt_path}")

    # ---- generate a sample ----
    model.eval()
    prompt = "\n"
    start = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=args.sample_tokens, temperature=args.temperature)
    print("\n===== sample =====")
    print(decode(out[0].tolist(), itos))
    print("==================")


if __name__ == "__main__":
    main()
