import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Make training runs more reproducible. / 讓訓練結果更具有可重現性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PixivFacesDataset(Dataset):
    """Load cropped Pixiv face images from a folder. / 從資料夾載入 Pixiv 臉部裁切影像。"""

    def __init__(self, root: str, image_size: int, limit: Optional[int] = None) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.files = sorted(self.root.glob("*.jpg")) + sorted(self.root.glob("*.png"))
        if limit is not None:
            self.files = self.files[:limit]
        if not self.files:
            raise RuntimeError(f"No images found in {self.root}")

        # Diffusion models usually operate on normalized images in [-1, 1].
        # 擴散模型通常在 [-1, 1] 的正規化影像空間中訓練。
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.files[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode diffusion steps into continuous vectors. / 將擴散時間步編碼為連續向量。"""
    half_dim = dim // 2
    exponent = -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning. / 帶有時間步條件資訊的殘差區塊。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.block1(x)
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h)
        return h + residual


class DownBlock(nn.Module):
    """Encoder block with residual layers and downsampling. / 含殘差層與下採樣的編碼器區塊。"""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block with upsampling and skip fusion. / 含上採樣與跳接融合的解碼器區塊。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels + skip_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        return x


class SimpleUNet(nn.Module):
    """Compact U-Net for DDPM noise prediction. / 用於 DDPM 噪聲預測的輕量化 U-Net。"""

    def __init__(self, base_channels: int = 64, time_dim: int = 256) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_proj = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.down1 = DownBlock(base_channels, base_channels, time_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up3 = UpBlock(base_channels, base_channels, base_channels, time_dim)
        self.output_head = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = sinusoidal_time_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        x0 = self.input_proj(x)
        x1, skip1 = self.down1(x0, time_emb)
        x2, skip2 = self.down2(x1, time_emb)
        x3, skip3 = self.down3(x2, time_emb)
        x3 = self.mid1(x3, time_emb)
        x3 = self.mid2(x3, time_emb)
        x = self.up1(x3, skip3, time_emb)
        x = self.up2(x, skip2, time_emb)
        x = self.up3(x, skip1, time_emb)
        return self.output_head(x)


class DiffusionSchedule:
    """Precompute diffusion coefficients. / 預先計算前向與反向擴散所需的係數。"""

    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device) -> None:
        self.steps = steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bars[:-1]])
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Gather coefficients for each timestep. / 取出每個時間步對應的係數。"""
        out = values.gather(0, timesteps)
        return out.view(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Apply forward diffusion noise. / 對原始影像加入前向擴散噪聲。"""
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timesteps, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        """Estimate the original clean image. / 估計原始乾淨影像。"""
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, timesteps, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t.shape)
        return (x_t - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Run one reverse diffusion step. / 執行一次反向去噪步驟。"""
        betas_t = self.extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timesteps, x.shape)
        sqrt_recip_alpha = self.extract(self.sqrt_recip_alphas, timesteps, x.shape)

        pred_noise = model(x, timesteps)
        model_mean = sqrt_recip_alpha * (x - betas_t * pred_noise / sqrt_one_minus_alpha_bar)

        nonzero_mask = (timesteps != 0).float().view(-1, 1, 1, 1)
        posterior_var = self.extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        return model_mean + nonzero_mask * torch.sqrt(torch.clamp(posterior_var, min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int,
        sample_steps_to_keep: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate images from Gaussian noise. / 從高斯隨機噪聲逐步生成影像。"""
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        trajectory: List[torch.Tensor] = []
        keep_every = max(1, self.steps // max(sample_steps_to_keep, 1))

        for step in reversed(range(self.steps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timesteps)
            if step % keep_every == 0 or step == 0:
                trajectory.append(x.detach().cpu())

        return x, trajectory


@dataclass
class TrainingConfig:
    data_dir: str
    output_dir: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    diffusion_steps: int
    beta_start: float
    beta_end: float
    base_channels: int
    num_workers: int
    max_images: Optional[int]
    val_ratio: float
    sample_count: int
    keep_trajectory_steps: int
    seed: int
    device: str
    save_epochs: Optional[List[int]]


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] back to [0, 1]. / 將影像從 [-1, 1] 還原到 [0, 1]。"""
    return images.clamp(-1, 1).add(1).div(2)


def save_loss_curve(train_losses: List[float], val_losses: List[float], path: Path) -> None:
    """Save a report-ready loss chart. / 輸出可直接放入報告的損失曲線圖。"""
    epochs = np.arange(1, len(train_losses) + 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    ax.set_facecolor("#fbfbfd")

    ax.plot(
        epochs,
        train_losses,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=5.5,
        label="Training Loss",
    )
    ax.plot(
        epochs,
        val_losses,
        color="#d62728",
        linewidth=2.4,
        marker="s",
        markersize=5.0,
        label="Validation Loss",
    )

    ax.fill_between(epochs, train_losses, alpha=0.10, color="#1f77b4")
    ax.fill_between(epochs, val_losses, alpha=0.08, color="#d62728")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("DDPM Training and Validation Loss", fontsize=13, pad=14)
    ax.legend(frameon=True, fancybox=True, framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04, y=0.15)

    # Single-epoch runs need explicit annotation or the chart looks empty.
    # 若只有 1 個 epoch，必須直接標註數值，否則圖表看起來會像空白。
    if len(epochs) == 1:
        ax.annotate(
            f"train={train_losses[0]:.4f}",
            (epochs[0], train_losses[0]),
            xytext=(8, 10),
            textcoords="offset points",
            color="#1f77b4",
            fontsize=10,
            weight="bold",
        )
        ax.annotate(
            f"val={val_losses[0]:.4f}",
            (epochs[0], val_losses[0]),
            xytext=(8, -16),
            textcoords="offset points",
            color="#d62728",
            fontsize=10,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_curated_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    path: Path,
    tick_epochs: List[int],
) -> Dict[str, float]:
    """Save a curated report figure with selected ticks and special annotations."""
    epochs = np.arange(1, len(train_losses) + 1)
    tick_epochs = [epoch for epoch in tick_epochs if 1 <= epoch <= len(train_losses)]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val = float(val_losses[best_epoch - 1])

    early_window_end = min(len(val_losses), 20)
    early_fluctuation_epoch = int(np.argmax(val_losses[:early_window_end])) + 1
    early_fluctuation_val = float(val_losses[early_fluctuation_epoch - 1])

    plateau_reference_epoch = tick_epochs[-1] if tick_epochs else len(train_losses)
    plateau_gap = float(abs(train_losses[plateau_reference_epoch - 1] - val_losses[plateau_reference_epoch - 1]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.set_facecolor("#fbfbfd")

    ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2.3, label="Training Loss")
    ax.plot(epochs, val_losses, color="#d62728", linewidth=2.3, label="Validation Loss")

    ax.scatter(tick_epochs, [train_losses[epoch - 1] for epoch in tick_epochs], color="#1f77b4", s=26, zorder=4)
    ax.scatter(tick_epochs, [val_losses[epoch - 1] for epoch in tick_epochs], color="#d62728", s=26, zorder=4)

    ax.scatter([best_epoch], [best_val], color="#2ca02c", s=48, zorder=5, label="Best Validation")
    ax.annotate(
        f"Best val\nE{best_epoch}: {best_val:.4f}",
        (best_epoch, best_val),
        xytext=(12, -26),
        textcoords="offset points",
        fontsize=9,
        color="#2ca02c",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2ca02c", alpha=0.9),
    )

    ax.scatter([early_fluctuation_epoch], [early_fluctuation_val], color="#ff7f0e", s=48, zorder=5, label="Early Fluctuation")
    ax.annotate(
        f"Early fluctuation\nE{early_fluctuation_epoch}: {early_fluctuation_val:.4f}",
        (early_fluctuation_epoch, early_fluctuation_val),
        xytext=(14, 16),
        textcoords="offset points",
        fontsize=9,
        color="#ff7f0e",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ff7f0e", alpha=0.9),
    )

    ax.scatter(
        [plateau_reference_epoch],
        [val_losses[plateau_reference_epoch - 1]],
        color="#9467bd",
        s=48,
        zorder=5,
        label="Late-stage Plateau",
    )
    ax.annotate(
        f"Late-stage plateau\nE{plateau_reference_epoch} gap={plateau_gap:.4f}",
        (plateau_reference_epoch, val_losses[plateau_reference_epoch - 1]),
        xytext=(-110, 20),
        textcoords="offset points",
        fontsize=9,
        color="#9467bd",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#9467bd", alpha=0.9),
    )

    ax.set_xticks(tick_epochs)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("DDPM Loss Curve with Key Epoch Milestones", fontsize=13, pad=12)
    ax.legend(frameon=True, fancybox=True, framealpha=0.95, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02, y=0.15)

    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "early_fluctuation_epoch": early_fluctuation_epoch,
        "early_fluctuation_val_loss": early_fluctuation_val,
        "plateau_reference_epoch": plateau_reference_epoch,
        "plateau_gap": plateau_gap,
    }


def save_trajectory_grid(trajectory: List[torch.Tensor], path: Path) -> None:
    """Show one sample across denoising stages. / 顯示單一樣本在各去噪階段的結果。"""
    frames = [denormalize(batch[0]).permute(1, 2, 0).numpy() for batch in trajectory]
    total_frames = len(frames)
    fig, axes = plt.subplots(1, total_frames, figsize=(3.1 * total_frames, 3.6), facecolor="white")
    if total_frames == 1:
        axes = [axes]

    for index, (axis, frame) in enumerate(zip(axes, frames), start=1):
        axis.imshow(np.clip(frame, 0.0, 1.0))
        axis.set_title(f"Stage {index}", fontsize=11, pad=8)
        axis.axis("off")

    fig.suptitle("Reverse Diffusion Trajectory", fontsize=14, y=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sample_grid(samples: torch.Tensor, path: Path, title: str) -> None:
    """Save generated samples in a report-ready grid. / 以較正式的版面輸出生成樣本。"""
    samples = denormalize(samples).permute(0, 2, 3, 1).numpy()
    total = samples.shape[0]
    columns = min(4, total)
    rows = math.ceil(total / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows + 0.5), facecolor="white")
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index in range(rows * columns):
        axis = axes.flat[index]
        axis.axis("off")
        if index < total:
            axis.imshow(np.clip(samples[index], 0.0, 1.0))
            axis.set_title(f"Sample {index + 1}", fontsize=10, pad=6)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_epoch_comparison_figure(samples_dir: Path, target_epochs: List[int], output_path: Path) -> None:
    """Combine selected sample grids into one report-ready comparison figure."""
    target_epochs = sorted(target_epochs)
    existing_epochs = [epoch for epoch in target_epochs if (samples_dir / f"epoch_{epoch:03d}_grid.png").exists()]
    if not existing_epochs:
        return

    columns = 2
    rows = math.ceil(len(existing_epochs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 4.2 * rows), facecolor="white")
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for index in range(rows * columns):
        axis = axes.flat[index]
        axis.axis("off")
        if index < len(existing_epochs):
            epoch = existing_epochs[index]
            image = Image.open(samples_dir / f"epoch_{epoch:03d}_grid.png").convert("RGB")
            axis.imshow(image)
            axis.set_title(f"Epoch {epoch}", fontsize=12, pad=8)

    fig.suptitle("Generated Sample Comparison Across Key Epochs", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_style_metrics(dataset_samples: torch.Tensor, generated_samples: torch.Tensor) -> Dict[str, float]:
    """Compute simple style statistics. / 計算簡易風格統計指標。"""

    def stats(images: torch.Tensor) -> Dict[str, float]:
        # These statistics approximate color distribution and local texture strength.
        # 這些統計量可近似描述顏色分布與局部紋理強度。
        images = denormalize(images).cpu()
        mean_rgb = images.mean(dim=[0, 2, 3])
        std_rgb = images.std(dim=[0, 2, 3])
        saturation = (images.max(dim=1).values - images.min(dim=1).values).mean()
        gradients_x = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
        gradients_y = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
        return {
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
            "std_r": float(std_rgb[0]),
            "std_g": float(std_rgb[1]),
            "std_b": float(std_rgb[2]),
            "saturation": float(saturation),
            "texture": float((gradients_x + gradients_y) / 2),
        }

    dataset_stats = stats(dataset_samples)
    generated_stats = stats(generated_samples)
    metrics = {}
    for key in dataset_stats:
        metrics[f"dataset_{key}"] = dataset_stats[key]
        metrics[f"generated_{key}"] = generated_stats[key]
        metrics[f"abs_diff_{key}"] = abs(dataset_stats[key] - generated_stats[key])
    return metrics


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    diffusion: DiffusionSchedule,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average validation loss. / 計算平均驗證損失。"""
    model.eval()
    losses: List[float] = []
    for images in dataloader:
        images = images.to(device)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, diffusion.steps, (images.shape[0],), device=device)
        noisy_images = diffusion.q_sample(images, timesteps, noise)
        pred_noise = model(noisy_images, timesteps)
        losses.append(F.mse_loss(pred_noise, noise).item())
    return float(np.mean(losses)) if losses else 0.0


def create_data_loaders(config: TrainingConfig) -> Tuple[Dataset, DataLoader, DataLoader]:
    """Create train and validation loaders. / 建立訓練與驗證資料載入器。"""
    dataset = PixivFacesDataset(config.data_dir, config.image_size, config.max_images)
    if len(dataset) < 2:
        raise ValueError("The dataset must contain at least 2 images for train/validation splitting.")

    val_size = max(1, int(len(dataset) * config.val_ratio))
    val_size = min(val_size, len(dataset) - 1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= config.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, train_loader, val_loader


def train(config: TrainingConfig) -> None:
    """Run the full DDPM training pipeline. / 執行完整的 DDPM 訓練流程。"""
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    checkpoints_dir = output_dir / "checkpoints"
    samples_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    dataset, train_loader, val_loader = create_data_loaders(config)
    model = SimpleUNet(base_channels=config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    diffusion = DiffusionSchedule(
        steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    effective_sample_count = min(config.sample_count, len(dataset))
    save_epoch_set = set(config.save_epochs or [])

    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses: List[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for images in progress:
            images = images.to(device, non_blocking=True)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, config.diffusion_steps, (images.shape[0],), device=device)
            noisy_images = diffusion.q_sample(images, timesteps, noise)
            pred_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_validation_loss(model, diffusion, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Generate a small batch after each epoch for qualitative inspection.
        # 每個 epoch 後產生少量樣本，方便進行視覺品質檢查。
        if epoch in save_epoch_set or epoch == config.epochs:
            generated, trajectory = diffusion.sample(
                model=model,
                image_size=config.image_size,
                batch_size=effective_sample_count,
                sample_steps_to_keep=config.keep_trajectory_steps,
                device=device,
            )
            generated = generated.cpu()
            save_sample_grid(
                generated,
                samples_dir / f"epoch_{epoch:03d}_grid.png",
                title=f"Generated Samples After Epoch {epoch}",
            )
            save_trajectory_grid(trajectory, samples_dir / f"epoch_{epoch:03d}_trajectory.png")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": asdict(config),
        }
        if epoch in save_epoch_set or epoch == config.epochs:
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoints_dir / "best_model.pt")

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    save_loss_curve(train_losses, val_losses, output_dir / "loss_curve.png")
    curated_tick_epochs = [1, 10, 50, 100, 200, 300, 500, 1000]
    highlighted = save_curated_loss_curve(
        train_losses,
        val_losses,
        output_dir / "loss_curve_curated.png",
        curated_tick_epochs,
    )
    save_epoch_comparison_figure(samples_dir, curated_tick_epochs, output_dir / "epoch_comparison.png")

    # Save final evaluation metrics based on a small subset of the dataset.
    # 以少量資料樣本計算最終統計指標，供報告與比較使用。
    model.eval()
    eval_count = effective_sample_count
    reference_images = torch.stack([dataset[i] for i in range(eval_count)])
    generated, _ = diffusion.sample(
        model=model,
        image_size=config.image_size,
        batch_size=eval_count,
        sample_steps_to_keep=config.keep_trajectory_steps,
        device=device,
    )
    metrics = compute_style_metrics(reference_images, generated.cpu())

    summary = {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_size": len(dataset),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "highlighted_points": highlighted,
        "style_metrics": metrics,
        "config": asdict(config),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def parse_args() -> TrainingConfig:
    """Parse command-line arguments. / 解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="Train a DDPM on the Pixiv face dataset. / 在 Pixiv 臉部資料集上訓練 DDPM。"
    )
    parser.add_argument("--data-dir", type=str, default="crop_2020_img")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=300)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--keep-trajectory-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-epochs",
        type=str,
        default="",
        help="Comma-separated epochs to export samples/checkpoints, e.g. 1,10,50,100. / 需要輸出樣本與 checkpoint 的 epoch。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: auto, cuda, cpu, or cuda:0. / 執行裝置：auto、cuda、cpu 或 cuda:0。",
    )
    args = parser.parse_args()
    save_epochs = [int(token) for token in args.save_epochs.split(",") if token.strip()]
    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        diffusion_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        max_images=args.max_images,
        val_ratio=args.val_ratio,
        sample_count=args.sample_count,
        keep_trajectory_steps=args.keep_trajectory_steps,
        seed=args.seed,
        device=args.device,
        save_epochs=save_epochs,
    )


if __name__ == "__main__":
    train(parse_args())
