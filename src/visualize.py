from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


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

    fig, axes = plt.subplots(1, len(sample_list), figsize=(5 * len(sample_list), 5))
    if len(sample_list) == 1:
        axes = [axes]

    for ax, sample in zip(axes, sample_list):
        attn = sample.attention
        ax.imshow(attn, cmap="bone", origin="lower", aspect="auto")
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

    layer, head_idx = head
    fig.suptitle(f"Attention L{layer}.H{head_idx}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
