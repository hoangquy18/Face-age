#!/usr/bin/env python3
"""
Evaluate a trained MTLFace backbone on InsightFace-style test sets produced by
`dataset/convert_insightface.py --bin` (e.g. README: dest folder `arcface-test-set`).

Layout expected (one .bin -> one benchmark):
  arcface-test-set/
    agedb_30/
      00001.jpg ...
    agedb_30.txt     # lines: "00001.jpg 00002.jpg 1"  (1=same, -1=different)
    lfw.txt
    lfw/
    ...

Metrics: verification accuracy @ optimal threshold, TAR @ FAR=1e-4 (optional).

Example:
  python evaluate_arcface_test_set.py \\
      --test_root /path/to/arcface-test-set \\
      --checkpoint output/save_models/backbone-36000 \\
      --backbone_name ir50 --image_size 112
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def add_repo_to_path() -> None:
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_backbone(
    backbone_name: str, image_size: int, checkpoint: str, device: torch.device
):
    add_repo_to_path()
    from backbone.aifr import backbone_dict
    from common.ops import load_network

    net = backbone_dict[backbone_name](input_size=image_size)
    net.eval()
    state = load_network(checkpoint)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print("Warning: missing keys:", len(missing))
    if unexpected:
        print("Warning: unexpected keys:", len(unexpected))
    print(f"Loaded backbone from: {checkpoint}")
    return net.to(device)


def preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class PairListImages(Dataset):
    """Loads unique images referenced by an InsightFace pair list."""

    def __init__(self, image_dir: str, filenames: list[str], tfm):
        self.image_dir = image_dir
        self.filenames = filenames
        self.tfm = tfm

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        return self.tfm(img), idx


@torch.inference_mode()
def encode_all(
    net: torch.nn.Module,
    image_dir: str,
    unique_names: list[str],
    tfm,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Return map filename -> L2-normalized 512-D embedding on CPU."""
    ds = PairListImages(image_dir, unique_names, tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=device.type == "cuda",
    )
    out: dict[str, torch.Tensor] = {}
    for batch, indices in tqdm(loader, desc="encode", leave=False):
        batch = batch.to(device, non_blocking=True)
        emb = net(batch)
        emb = F.normalize(emb, dim=1).cpu()
        for i in range(emb.size(0)):
            out[unique_names[indices[i].item()]] = emb[i]
    return out


def parse_pair_file(pair_path: str) -> tuple[list[tuple[str, str, int]], list[str]]:
    pairs: list[tuple[str, str, int]] = []
    names: set[str] = set()
    with open(pair_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b, lab = parts[0], parts[1], int(parts[2])
            pairs.append((a, b, lab))
            names.add(a)
            names.add(b)
    return pairs, sorted(names)


def accuracy_at_best_threshold(
    scores: np.ndarray, y_true: np.ndarray
) -> tuple[float, float]:
    """y_true: 1 same, 0 different. Max accuracy over threshold sweep."""
    if len(np.unique(y_true)) < 2:
        return (
            float((scores > 0.5).mean() if y_true[0] == 1 else (scores <= 0.5).mean()),
            0.5,
        )
    u = np.sort(np.unique(scores))
    if len(u) > 1:
        mids = (u[:-1] + u[1:]) / 2.0
        cand = np.concatenate([u, mids])
    else:
        cand = u
    best_acc = 0.0
    best_t = float(u[0])
    for t in cand:
        pred = (scores > t).astype(np.int32)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_acc, best_t


def tar_at_far(
    scores: np.ndarray, y_true: np.ndarray, far: float = 1e-4
) -> tuple[float, float]:
    """
    y_true: 1 genuine (same), 0 impostor (different).
    Threshold from impostor distribution so FAR ~= far; report TAR on genuine.
    """
    gen = scores[y_true == 1]
    imp = scores[y_true == 0]
    if len(imp) == 0 or len(gen) == 0:
        return float("nan"), float("nan")
    # Threshold: (1-far) quantile of impostor scores (accept if score > t)
    t = float(np.quantile(imp, 1.0 - far))
    tar = float((gen > t).mean())
    return tar, t


def run_benchmark(
    net,
    pair_path: str,
    image_dir: str,
    tfm,
    device: torch.device,
    batch_size: int,
) -> dict:
    pairs, unique_names = parse_pair_file(pair_path)
    if not pairs:
        return {"name": os.path.basename(pair_path), "error": "no pairs", "n_pairs": 0}

    emb_map = encode_all(net, image_dir, unique_names, tfm, device, batch_size)

    scores_list = []
    y_list = []
    for a, b, lab in pairs:
        ea = emb_map[a].numpy()
        eb = emb_map[b].numpy()
        s = float(np.dot(ea, eb))
        scores_list.append(s)
        y_list.append(1 if lab == 1 else 0)

    scores = np.array(scores_list, dtype=np.float64)
    y_true = np.array(y_list, dtype=np.int32)

    acc, thr_acc = accuracy_at_best_threshold(scores, y_true)
    tar, thr_far = tar_at_far(scores, y_true, far=1e-4)

    return {
        "name": os.path.splitext(os.path.basename(pair_path))[0],
        "n_pairs": len(pairs),
        "accuracy_opt": acc * 100.0,
        "threshold_opt": thr_acc,
        "tar_far1e4": tar * 100.0,
        "threshold_far1e4": thr_far,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MTLFace backbone on arcface-test-set (InsightFace .bin converted)"
    )
    parser.add_argument(
        "--test_root",
        type=str,
        required=True,
        help="Folder containing <name>.txt pair lists and <name>/ image dirs (convert_insightface --bin output)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to backbone-* state dict"
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="ir50",
        choices=["ir34", "ir50", "ir64", "ir101", "irse101"],
    )
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated basenames without .txt (e.g. agedb_30,lfw). Default: all *.txt in test_root",
    )
    args = parser.parse_args()

    test_root = os.path.abspath(args.test_root)
    if not os.path.isdir(test_root):
        raise FileNotFoundError(test_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_backbone(args.backbone_name, args.image_size, args.checkpoint, device)
    tfm = preprocess(args.image_size)

    if args.only:
        names = [x.strip() for x in args.only.split(",") if x.strip()]
        pair_files = [os.path.join(test_root, f"{n}.txt") for n in names]
        for p in pair_files:
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
    else:
        pair_files = sorted(glob(os.path.join(test_root, "*.txt")))

    if not pair_files:
        raise RuntimeError(f"No *.txt pair lists under {test_root}")

    print(f"test_root={test_root}")
    print(f"device={device}, batch_size={args.batch_size}")
    print()

    rows = []
    for pair_path in pair_files:
        name = os.path.splitext(os.path.basename(pair_path))[0]
        image_dir = os.path.join(test_root, name)
        if not os.path.isdir(image_dir):
            print(f"[skip] {name}: missing image dir {image_dir}")
            continue
        print(f"=== {name} ===")
        r = run_benchmark(net, pair_path, image_dir, tfm, device, args.batch_size)
        if "error" in r:
            print(r)
            continue
        rows.append(r)
        print(f"  pairs: {r['n_pairs']}")
        print(f"  verification accuracy (optimal threshold): {r['accuracy_opt']:.2f}%")
        print(f"  TAR @ FAR=1e-4: {r['tar_far1e4']:.2f}%")
        print()

    if rows:
        print("--- Summary ---")
        for r in rows:
            print(
                f"{r['name']:16s}  acc={r['accuracy_opt']:.2f}%   TAR@FAR1e-4={r['tar_far1e4']:.2f}%"
            )


if __name__ == "__main__":
    main()
