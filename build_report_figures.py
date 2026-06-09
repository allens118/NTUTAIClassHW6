import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TARGET_EPOCHS = [1, 10, 50, 100, 200, 300, 500, 1000]


def find_early_rebound_epoch(val_losses: List[float], end_epoch: int = 30) -> int:
    best_epoch = 2
    best_increase = float("-inf")
    upper = min(end_epoch, len(val_losses))
    for epoch in range(2, upper + 1):
        increase = val_losses[epoch - 1] - val_losses[epoch - 2]
        if increase > best_increase:
            best_increase = increase
            best_epoch = epoch
    return best_epoch


def build_curated_loss_curve(output_dir: Path, summary: dict) -> dict:
    train_losses = summary["train_losses"]
    val_losses = summary["val_losses"]
    epochs = np.arange(1, len(train_losses) + 1)
    tick_epochs = [epoch for epoch in TARGET_EPOCHS if epoch <= len(train_losses)]

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = float(val_losses[best_epoch - 1])
    early_rebound_epoch = find_early_rebound_epoch(val_losses)
    early_rebound_val = float(val_losses[early_rebound_epoch - 1])

    selected_val_pairs = [(epoch, val_losses[epoch - 1]) for epoch in tick_epochs if epoch >= 100]
    late_rebound_epoch, late_rebound_val = max(selected_val_pairs, key=lambda pair: pair[1])
    late_gap = float(val_losses[late_rebound_epoch - 1] - train_losses[late_rebound_epoch - 1])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.set_facecolor("#fbfbfd")

    ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2.2, label="Training Loss")
    ax.plot(epochs, val_losses, color="#d62728", linewidth=2.2, label="Validation Loss")
    ax.scatter(tick_epochs, [train_losses[epoch - 1] for epoch in tick_epochs], color="#1f77b4", s=28, zorder=4)
    ax.scatter(tick_epochs, [val_losses[epoch - 1] for epoch in tick_epochs], color="#d62728", s=28, zorder=4)

    ax.scatter([best_epoch], [best_val], color="#2ca02c", s=60, zorder=5, label="Best Validation")
    ax.annotate(
        f"Best validation\nE{best_epoch}: {best_val:.4f}",
        (best_epoch, best_val),
        xytext=(12, -28),
        textcoords="offset points",
        fontsize=9,
        color="#2ca02c",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2ca02c", alpha=0.95),
    )

    ax.scatter([early_rebound_epoch], [early_rebound_val], color="#ff7f0e", s=60, zorder=5, label="Early Rebound")
    ax.annotate(
        f"Early rebound\nE{early_rebound_epoch}: {early_rebound_val:.4f}",
        (early_rebound_epoch, early_rebound_val),
        xytext=(12, 16),
        textcoords="offset points",
        fontsize=9,
        color="#ff7f0e",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ff7f0e", alpha=0.95),
    )

    ax.scatter([late_rebound_epoch], [late_rebound_val], color="#9467bd", s=60, zorder=5, label="Late Rebound")
    ax.annotate(
        f"Late rebound\nE{late_rebound_epoch}: gap={late_gap:.4f}",
        (late_rebound_epoch, late_rebound_val),
        xytext=(-110, 18),
        textcoords="offset points",
        fontsize=9,
        color="#9467bd",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#9467bd", alpha=0.95),
    )

    ax.set_xticks(tick_epochs)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("DDPM Loss Curve with Key Milestones and Special Points", fontsize=13, pad=12)
    ax.legend(frameon=True, fancybox=True, framealpha=0.95, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02, y=0.16)

    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve_curated_v2.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "early_rebound_epoch": early_rebound_epoch,
        "early_rebound_val_loss": early_rebound_val,
        "late_rebound_epoch": late_rebound_epoch,
        "late_rebound_val_loss": float(late_rebound_val),
        "late_rebound_gap": late_gap,
    }


def build_milestone_metrics(output_dir: Path, summary: dict) -> None:
    train_losses = summary["train_losses"]
    val_losses = summary["val_losses"]
    epochs = [epoch for epoch in TARGET_EPOCHS if epoch <= len(train_losses)]
    train_points = [train_losses[epoch - 1] for epoch in epochs]
    val_points = [val_losses[epoch - 1] for epoch in epochs]
    gaps = [val - train for train, val in zip(train_points, val_points)]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), facecolor="white")

    axes[0].set_facecolor("#fbfbfd")
    axes[0].plot(epochs, train_points, color="#1f77b4", linewidth=2.2, marker="o", label="Training Loss")
    axes[0].plot(epochs, val_points, color="#d62728", linewidth=2.2, marker="s", label="Validation Loss")
    axes[0].set_title("Milestone Loss Comparison", fontsize=13, pad=10)
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_xticks(epochs)
    axes[0].legend(frameon=True, fancybox=True, framealpha=0.95)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    bar_colors = ["#2ca02c" if gap < 0.03 else "#ff7f0e" if gap < 0.08 else "#d62728" for gap in gaps]
    axes[1].set_facecolor("#fbfbfd")
    axes[1].bar([str(epoch) for epoch in epochs], gaps, color=bar_colors, width=0.62)
    axes[1].axhline(0.0, color="#555555", linewidth=1.0)
    axes[1].set_title("Validation Minus Training Loss at Milestones", fontsize=13, pad=10)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Generalization Gap")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "milestone_metrics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_epoch_comparison(output_dir: Path) -> None:
    samples_dir = output_dir / "samples"
    epochs = [epoch for epoch in TARGET_EPOCHS if (samples_dir / f"epoch_{epoch:03d}_grid.png").exists()]
    if not epochs:
        return

    columns = 2
    rows = int(np.ceil(len(epochs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11.5, 4.4 * rows), facecolor="white")
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index in range(rows * columns):
        axis = axes.flat[index]
        axis.axis("off")
        if index < len(epochs):
            epoch = epochs[index]
            image = Image.open(samples_dir / f"epoch_{epoch:03d}_grid.png").convert("RGB")
            axis.imshow(image)
            axis.set_title(f"Epoch {epoch}", fontsize=12, pad=8)

    fig.suptitle("Generated Sample Grids at Key Epoch Milestones", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "epoch_comparison_v2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs_compare_1000")
    summary_path = output_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    special_points = build_curated_loss_curve(output_dir, summary)
    build_milestone_metrics(output_dir, summary)
    build_epoch_comparison(output_dir)

    summary["report_special_points"] = special_points
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
