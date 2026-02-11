from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm


@dataclass
class SampleAttention:
    index: int
    valid: bool
    attention: np.ndarray
    reasoning_start: int | None
    reasoning_end: int | None
    answer_start: int | None


def plot_head_attention(
    head: tuple[int, int],
    samples: Iterable[SampleAttention],
    output_path: Path,
) -> None:
    sample_list = list(samples)
    if not sample_list:
        return

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes_list = list(axes.flat)

    for ax, sample in zip(axes_list, sample_list):
        attn = sample.attention
        vmax = float(np.quantile(attn, 0.99))
        if vmax <= 0:
            vmax = float(attn.max()) if attn.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        norm = PowerNorm(gamma=0.3, vmin=0.0, vmax=vmax)
        ax.imshow(attn, cmap="bone", origin="lower", aspect="auto", norm=norm)
        ax.set_xlabel("Key token index")
        ax.set_ylabel("Query token index")

        if (
            sample.reasoning_start is not None
            and sample.reasoning_end is not None
            and sample.answer_start is not None
        ):
            reasoning_start = sample.reasoning_start
            reasoning_end = sample.reasoning_end
            answer_start = sample.answer_start
            answer_end = attn.shape[0]

            rect = Rectangle(
                (reasoning_start, answer_start),
                reasoning_end - reasoning_start,
                answer_end - answer_start,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        status = "valid" if sample.valid else "invalid"
        ax.set_title(f"Puzzle {sample.index:03d} ({status})")

    for ax in axes_list[len(sample_list) :]:
        ax.axis("off")

    layer, head_idx = head
    fig.suptitle(f"Attention L{layer}.H{head_idx}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
