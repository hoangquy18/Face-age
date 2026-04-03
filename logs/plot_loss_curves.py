#!/usr/bin/env python3
"""Parse training logs and plot id_loss, da_loss, age_loss for multiple models."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path(__file__).resolve().parent
# So sánh cùng số epoch (các log khác 20k; mobilenetv2.log có 30k epoch)
MAX_EPOCH = 20000
# Trung bình trượt (epoch) để làm mượt đường cong
SMOOTH_WINDOW = 500

# (filename stem -> legend label)
LOGS = [
    ("ir50", "IR-50"),
    ("ir64", "IR-64"),
    ("ir164", "IR-164"),
    ("mobilenetv2", "MobileNetV2"),
    ("vit", "ViT-B/32"),
    ("two_task", "Two-task + FAS"),
]

# Lines like: [2026-04-01 10:37:44] 00001, id_loss 35.53136, da_loss 312.31958, age_loss 340.76581, lr ...
# May be prefixed by tqdm: ...it/s][2026-04-01 ...
LINE_RE = re.compile(
    r"\]\s*(\d+)\s*,\s*id_loss\s+([\d.eE+-]+)\s*,\s*da_loss\s+([\d.eE+-]+)\s*,\s*age_loss\s+([\d.eE+-]+)",
)


def parse_log(path: Path):
    epochs, id_l, da_l, age_l = [], [], [], []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            try:
                i = float(m.group(2))
                d = float(m.group(3))
                a = float(m.group(4))
            except ValueError:
                continue
            if not all(map(lambda x: x == x, (i, d, a))):  # skip nan
                continue
            if ep > MAX_EPOCH:
                continue
            epochs.append(ep)
            id_l.append(i)
            da_l.append(d)
            age_l.append(a)
    return epochs, id_l, da_l, age_l


def rolling_mean_smooth(
    epochs: list[int], values: list[float], window: int
) -> tuple[list[float], list[float]]:
    """Trung bình trượt; trục x = tâm mỗi cửa sổ."""
    e = np.asarray(epochs, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < window or window < 2:
        return e.tolist(), v.tolist()
    kernel = np.ones(window) / window
    smoothed = np.convolve(v, kernel, mode="valid")
    e_centers = (e[window - 1 :] + e[: n - window + 1]) / 2.0
    return e_centers.tolist(), smoothed.tolist()


def plot_metric(
    series: list[tuple[str, list[int], list[float]]],
    title: str,
    ylabel: str,
    outfile: Path,
):
    plt.figure(figsize=(11, 6))
    for label, epochs, values in series:
        if not epochs:
            continue
        ex, vy = rolling_mean_smooth(epochs, values, SMOOTH_WINDOW)
        plt.plot(ex, vy, label=label, linewidth=1.1, alpha=0.95)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title + f" (smoothed, window={SMOOTH_WINDOW})")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def main():
    id_series = []
    da_series = []
    age_series = []

    for stem, label in LOGS:
        path = LOG_DIR / f"{stem}.log"
        if not path.exists():
            print(f"Skip missing: {path}")
            continue
        ep, id_l, da_l, age_l = parse_log(path)
        print(f"{label}: {len(ep)} points from {path.name}")
        id_series.append((label, ep, id_l))
        da_series.append((label, ep, da_l))
        age_series.append((label, ep, age_l))

    plot_metric(
        id_series,
        "Identity loss (id_loss)",
        "id_loss",
        LOG_DIR / "id_loss.png",
    )
    plot_metric(
        da_series,
        "Domain adaptation loss (da_loss)",
        "da_loss",
        LOG_DIR / "da_loss.png",
    )
    plot_metric(
        age_series,
        "Age loss (age_loss)",
        "age_loss",
        LOG_DIR / "age_loss.png",
    )
    print(
        f"Wrote: {LOG_DIR / 'id_loss.png'}, {LOG_DIR / 'da_loss.png'}, {LOG_DIR / 'age_loss.png'}"
    )


if __name__ == "__main__":
    main()
