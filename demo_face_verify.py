#!/usr/bin/env python3
"""
Face encoding / verification demo using MTLFace AIResNet backbone.

Run with your conda env (e.g. moe):
  conda activate moe
  cd /path/to/MTLFace
  python demo_face_verify.py --img_a casia-webface/000000/00000001.jpg \\
      --img_b casia-webface/000000/00000002.jpg

Optional: load a trained backbone checkpoint from training output:
  python demo_face_verify.py --checkpoint output/save_models/backbone-36000 ...

Without --checkpoint, embeddings are from random initialization (similarity is not meaningful).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def add_repo_to_path() -> None:
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_backbone(backbone_name: str, image_size: int, checkpoint: str | None, device: torch.device):
    add_repo_to_path()
    from backbone.aifr import backbone_dict

    net = backbone_dict[backbone_name](input_size=image_size)
    net.eval()
    if checkpoint:
        from common.ops import load_network

        state = load_network(checkpoint)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print("Warning: missing keys when loading checkpoint:", len(missing))
        if unexpected:
            print("Warning: unexpected keys when loading checkpoint:", len(unexpected))
        print(f"Loaded backbone weights from: {checkpoint}")
    else:
        print(
            "No --checkpoint: using random weights. "
            "Cosine similarity is NOT meaningful for verification."
        )
    return net.to(device)


def preprocess(image_size: int) -> transforms.Compose:
    # Same as models/fr.py TrainImageDataset pipeline (without RandomHorizontalFlip)
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


@torch.inference_mode()
def encode(net: torch.nn.Module, path: str, tfm, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    emb = net(x)  # (1, 512)
    emb = F.normalize(emb, dim=1)
    return emb.squeeze(0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum().item())


def pick_demo_paths(data_root: str, list_path: str, same_identity: bool) -> tuple[str, str]:
    """Pick two image paths from casia-webface.txt (same or different id)."""
    import random

    rows = []
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            uid, rel = parts[0], parts[1]
            rows.append((uid, rel))
    if len(rows) < 2:
        raise RuntimeError(f"No usable rows in {list_path}")

    if same_identity:
        by_id: dict[str, list[str]] = {}
        for uid, rel in rows:
            by_id.setdefault(uid, []).append(rel)
        candidates = [u for u, paths in by_id.items() if len(paths) >= 2]
        if not candidates:
            raise RuntimeError("No identity with at least 2 images in list")
        uid = random.choice(candidates)
        p1, p2 = random.sample(by_id[uid], 2)
    else:
        uid_a, rel_a = random.choice(rows)
        uid_b, rel_b = random.choice(rows)
        for _ in range(5000):
            if uid_a != uid_b:
                break
            uid_b, rel_b = random.choice(rows)
        p1, p2 = rel_a, rel_b

    def full(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(data_root, p)

    return full(p1), full(p2)


def main() -> None:
    add_repo_to_path()
    from backbone.aifr import BACKBONE_CHOICES

    parser = argparse.ArgumentParser(
        description="MTLFace face verification demo (512-D embedding; IR / MobileNet / ViT backbones)"
    )
    parser.add_argument("--img_a", type=str, default=None, help="Path to first face image (112x112 RGB)")
    parser.add_argument("--img_b", type=str, default=None, help="Path to second face image")
    parser.add_argument(
        "--demo_random",
        action="store_true",
        help="Pick two random paths from --list_path (use with --same or default different id)",
    )
    parser.add_argument(
        "--same",
        action="store_true",
        help="With --demo_random: pick two images from the same identity",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Root for paths in list file")
    parser.add_argument(
        "--list_path",
        type=str,
        default=None,
        help="Annotation list (id path age gender). Default: dataset/casia-webface.txt",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Backbone state dict from training save_models/")
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="ir50",
        choices=list(BACKBONE_CHOICES),
        help="FR backbones: ir* (FAS-capable), mobilenet_v2, vit_b_32 (FR-only)",
    )
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold for SAME (tune with your model)")
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    root = repo_root()
    data_root = args.data_root or root
    list_path = args.list_path or os.path.join(root, "dataset", "casia-webface.txt")

    img_a, img_b = args.img_a, args.img_b
    if args.demo_random or img_a is None or img_b is None:
        if not os.path.isfile(list_path):
            parser.error(f"Need --img_a and --img_b, or a valid --list_path. Missing: {list_path}")
        img_a, img_b = pick_demo_paths(data_root, list_path, same_identity=args.same)
        print(f"Demo paths (same_identity={args.same}):")
        print("  A:", img_a)
        print("  B:", img_b)

    for p in (img_a, img_b):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    net = load_backbone(args.backbone_name, args.image_size, args.checkpoint, device)
    tfm = preprocess(args.image_size)

    e1 = encode(net, img_a, tfm, device)
    e2 = encode(net, img_b, tfm, device)
    sim = cosine_similarity(e1, e2)

    print(f"Device: {device}")
    print(f"Embedding dim: {e1.shape[0]}")
    print(f"Cosine similarity: {sim:.4f}")
    if args.checkpoint:
        verdict = "SAME person" if sim >= args.threshold else "DIFFERENT persons"
        print(f"Verdict (threshold={args.threshold}): {verdict}")
    else:
        print("Verdict: N/A (train or provide --checkpoint for real verification)")


if __name__ == "__main__":
    main()
