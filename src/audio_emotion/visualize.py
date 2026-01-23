from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from audio_emotion.data import AudioDataset
from audio_emotion.model import Model


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_indices(n_total: int, n_samples: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    n_samples = min(n_samples, n_total)
    perm = torch.randperm(n_total, generator=generator)
    return perm[:n_samples].tolist()


def _plot_grid(
    images: Iterable[torch.Tensor],
    titles: Iterable[str],
    out_path: Path,
    ncols: int = 4,
) -> Path:
    images_list = list(images)
    titles_list = list(titles)
    n_samples = len(images_list)
    ncols = max(1, ncols)
    nrows = (n_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, ax in enumerate(axes_list):
        if idx >= n_samples:
            ax.axis("off")
            continue
        img = images_list[idx].squeeze(0).cpu().numpy()
        ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(titles_list[idx])
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_class_distribution(dataset: AudioDataset, out_path: Path) -> Path:
    class_names = list(dataset.classes)
    counts = [0 for _ in class_names]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for audio_path in dataset.audio_files:
        label = audio_path.parent.name
        idx = class_to_idx.get(label)
        if idx is not None:
            counts[idx] += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    positions = range(len(class_names))
    ax.bar(positions, counts, color="#4c78a8")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: list[str],
    out_path: Path,
) -> Path:
    """Plot and save a confusion matrix heatmap."""

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    indices = list(range(n_classes))
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    max_val = cm.max().item() if cm.numel() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            value = int(cm[i, j].item())
            color = "white" if max_val > 0 and value > max_val * 0.5 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.no_grad()
def _compute_confusion_matrix(
    model: Model,
    dataset: AudioDataset,
    device: torch.device,
    channels_last: bool,
    max_samples: int,
    seed: int,
    batch_size: int,
) -> torch.Tensor:
    """Compute confusion matrix over a subset of the dataset."""

    n_classes = len(dataset.classes)
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long)

    indices = (
        _select_indices(len(dataset), max_samples, seed)
        if max_samples > 0
        else list(range(len(dataset)))
    )
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        if channels_last and batch_x.ndim == 4:
            batch_x = batch_x.contiguous(memory_format=torch.channels_last)
        batch_y = batch_y.to(device, non_blocking=True)

        logits = model(batch_x)
        preds = logits.argmax(dim=1)
        idx = (batch_y * n_classes + preds).to("cpu")
        cm += torch.bincount(idx, minlength=n_classes * n_classes).reshape(
            n_classes, n_classes
        )

    return cm


def _load_config(config_path: str, config_name: str) -> DictConfig:
    config_file = Path(config_path) / config_name
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    cfg = OmegaConf.load(config_file)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Loaded config is not a DictConfig")

    return cfg


def main(
    model_path: Path = Path("models/vgg16_audio.pt"),
    output_dir: Path = Path("reports/figures"),
    num_samples: int = 8,
    seed: int = 42,
    config_path: str = "configs",
    config_name: str = "config.yaml",
    show_class_distribution: bool = True,
    show_confusion_matrix: bool = True,
    confusion_matrix_path: Path = Path("models/confusion_matrix.pt"),
    recompute_confusion: bool = False,
    confusion_max_samples: int = 2048,
    confusion_batch_size: int = 32,
):
    """Visualize model predictions on a small batch of spectrograms."""

    cfg = _load_config(config_path, config_name)
    set_seed(seed)

    use_cuda = bool(cfg.environment.cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    channels_last = bool(getattr(cfg.training, "channels_last", False))

    dataset = AudioDataset(cfg, "data/raw", processed_dir="data/processed")
    dataset.send_to_processed("data/processed", overwrite=False, verbose=False)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = Model(cfg)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    indices = _select_indices(len(dataset), num_samples, seed)
    images: list[torch.Tensor] = []
    titles: list[str] = []

    for idx in indices:
        x, y = dataset[idx]
        x_batch = x.unsqueeze(0).to(device)
        if channels_last and x_batch.ndim == 4:
            x_batch = x_batch.contiguous(memory_format=torch.channels_last)
        logits = model(x_batch)
        pred_idx = int(logits.argmax(dim=1).item())
        true_idx = int(y.item())
        pred_label = (
            dataset.classes[pred_idx]
            if pred_idx < len(dataset.classes)
            else str(pred_idx)
        )
        true_label = (
            dataset.classes[true_idx]
            if true_idx < len(dataset.classes)
            else str(true_idx)
        )
        titles.append(f"pred: {pred_label} | true: {true_label}")
        images.append(x)

    out_path = output_dir / "prediction_grid.png"
    saved = _plot_grid(images, titles, out_path)
    typer.echo(f"Saved visualization to {saved}")

    if show_class_distribution:
        distribution_path = output_dir / "class_distribution.png"
        saved_distribution = _plot_class_distribution(dataset, distribution_path)
        typer.echo(f"Saved visualization to {saved_distribution}")

    if show_confusion_matrix:
        # Try to load pre-computed confusion matrix from evaluation
        cm = None
        if not recompute_confusion and confusion_matrix_path.exists():
            typer.echo(
                f"Loading pre-computed confusion matrix from {confusion_matrix_path}"
            )
            saved_data = torch.load(confusion_matrix_path, map_location="cpu")
            cm = saved_data["confusion_matrix"]
            saved_class_names = saved_data.get("class_names", list(dataset.classes))
            if saved_class_names != list(dataset.classes):
                typer.echo(
                    "Warning: Class names mismatch. Recomputing confusion matrix."
                )
                cm = None

        if cm is None:
            typer.echo(
                "Computing confusion matrix (run evaluate.py first to avoid this)"
            )
            cm = _compute_confusion_matrix(
                model,
                dataset,
                device,
                channels_last,
                confusion_max_samples,
                seed,
                confusion_batch_size,
            )

        cm_path = output_dir / "confusion_matrix.png"
        saved_cm = _plot_confusion_matrix(cm, list(dataset.classes), cm_path)
        typer.echo(f"Saved visualization to {saved_cm}")


if __name__ == "__main__":
    typer.run(main)
