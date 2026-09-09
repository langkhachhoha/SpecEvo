#!/usr/bin/env python3
"""Render a circle_packing_rect result stored in an .npz file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help=".npz file containing a circles array")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def minimum_circumscribing_rectangle(circles: np.ndarray) -> tuple[float, float, float, float]:
    min_x = float(np.min(circles[:, 0] - circles[:, 2]))
    max_x = float(np.max(circles[:, 0] + circles[:, 2]))
    min_y = float(np.min(circles[:, 1] - circles[:, 2]))
    max_y = float(np.max(circles[:, 1] + circles[:, 2]))
    return min_x, min_y, max_x - min_x, max_y - min_y


def render(circles: np.ndarray, output: Path, dpi: int) -> None:
    circles = np.asarray(circles, dtype=float)
    if circles.shape != (21, 3):
        raise ValueError(f"Expected circles shape (21, 3), got {circles.shape}")

    min_x, min_y, width, height = minimum_circumscribing_rectangle(circles)
    radii_sum = float(np.sum(circles[:, 2]))

    pad = max(0.04, 0.03 * max(width, height, 1.0))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x - pad, min_x + width + pad)
    ax.set_ylim(min_y - pad, min_y + height + pad)
    ax.grid(True, linewidth=0.7, alpha=0.35)

    rect = Rectangle((min_x, min_y), width, height, fill=False, linewidth=2.0, edgecolor="#303846")
    ax.add_patch(rect)

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(circles)))
    for idx, ((x, y, r), color) in enumerate(zip(circles, colors)):
        ax.add_patch(Circle((float(x), float(y)), float(r), facecolor=color, edgecolor="#202631", alpha=0.72))
        ax.text(float(x), float(y), str(idx), ha="center", va="center", fontsize=7, color="white")

    ax.set_title(
        f"circle_packing_rect n=21 | radii_sum={radii_sum:.6f} | width+height={width + height:.6f}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    if "circles" not in data:
        raise ValueError("Input .npz must contain a 'circles' array")
    render(data["circles"], args.output, args.dpi)


if __name__ == "__main__":
    main()
