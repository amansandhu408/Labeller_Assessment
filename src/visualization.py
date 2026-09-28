"""Small visualization helpers for model-vs-human annotation inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


ACTIONS = [
    "OPENS",
    "PICKS UP",
    "PUT DOWN",
    "POUR IN",
    "WALKS",
    "WASHING",
    "COOKING",
]


def plot_annotation_timeline(
    model_rows: Sequence[dict],
    human_rows: Sequence[dict] | None = None,
    title: str = "Annotation timeline",
    output_path: str | Path | None = None,
) -> None:
    """Plot model intervals, with dashed human-reference intervals."""
    action_to_y = {action: i for i, action in enumerate(ACTIONS)}
    fig, ax = plt.subplots(figsize=(12, 4))

    for row in model_rows:
        y = action_to_y.get(row.get("action", "OPENS"), 0)
        confidence = row.get("combined_confidence", row.get("action_confidence", 0.0))
        label = row.get("object") or "action-only"
        ax.plot([row["start"], row["end"]], [y, y], linewidth=8)
        ax.text(
            (row["start"] + row["end"]) / 2,
            y + 0.12,
            f"{row.get('action', '')} | {label} | {confidence:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for row in human_rows or []:
        y = action_to_y.get(row.get("action", "OPENS"), 0)
        ax.plot(
            [row["start"], row["end"]],
            [y - 0.18, y - 0.18],
            linewidth=3,
            linestyle="--",
        )

    ax.set_yticks(range(len(ACTIONS)))
    ax.set_yticklabels(ACTIONS)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    plt.show()
