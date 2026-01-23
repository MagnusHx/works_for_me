from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split

from audio_emotion.data import AudioDataset
from audio_emotion.model import Model


def set_seed(seed: int) -> None:
	"""Set RNG seeds for reproducibility."""

	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_model(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	channels_last: bool = False,
	show_progress: bool = True,
	num_classes: int = 0,
	class_names: list[str] | None = None,
	compute_confusion_matrix: bool = False,
) -> tuple[float, float, dict[str, float], torch.Tensor | None]:
	"""Compute average loss, accuracy, and optionally confusion matrix for a dataloader."""

	model.eval()
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	total = 0
	correct = 0
	total_batches = len(loader)

	class_names = class_names or []
	if num_classes <= 0:
		num_classes = len(class_names)
	class_correct = torch.zeros(num_classes, dtype=torch.long)
	class_total = torch.zeros(num_classes, dtype=torch.long)
	confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long) if compute_confusion_matrix else None

	if show_progress and total_batches > 0:
		typer.echo(f"Evaluating {total_batches} batches...", nl=True)

	for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
		batch_x = batch_x.to(device, non_blocking=True)
		if channels_last and batch_x.ndim == 4:
			batch_x = batch_x.contiguous(memory_format=torch.channels_last)
		batch_y = batch_y.to(device, non_blocking=True)

		logits = model(batch_x)
		loss = criterion(logits, batch_y)

		total_loss += loss.item() * batch_x.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == batch_y).sum().item()
		total += batch_y.size(0)

		if num_classes > 0:
			for class_idx in range(num_classes):
				mask = batch_y == class_idx
				class_total[class_idx] += mask.sum().item()
				class_correct[class_idx] += (preds[mask] == batch_y[mask]).sum().item()

		if confusion_matrix is not None:
			idx = (batch_y * num_classes + preds).cpu()
			confusion_matrix += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

		if show_progress and total_batches > 0:
			progress = min(batch_idx / total_batches, 1.0)
			bar_width = 24
			filled = int(bar_width * progress)
			bar = "=" * filled + "-" * (bar_width - filled)
			typer.echo(f"\r[{bar}] {batch_idx}/{total_batches}", nl=False)

	if show_progress and total_batches > 0:
		typer.echo("", nl=True)

	if total == 0:
		return 0.0, 0.0, {}, None

	per_class: dict[str, float] = {}
	if num_classes > 0:
		for class_idx in range(num_classes):
			name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
			denom = int(class_total[class_idx].item())
			per_class[name] = float(class_correct[class_idx].item() / denom) if denom > 0 else 0.0

	return total_loss / total, correct / total, per_class, confusion_matrix


def build_loader(dataset, cfg: DictConfig) -> DataLoader:
	"""Create a DataLoader using configuration values."""

	num_workers = int(getattr(cfg.dataloader, "num_workers", 0))
	loader_kwargs: dict[str, Any] = {
		"batch_size": int(cfg.dataloader.batch_size),
		"shuffle": False,
		"num_workers": num_workers,
		"pin_memory": bool(getattr(cfg.dataloader, "pin_memory", False)),
	}

	if num_workers > 0:
		loader_kwargs["prefetch_factor"] = int(getattr(cfg.dataloader, "prefetch_factor", 2))
		loader_kwargs["persistent_workers"] = bool(
			getattr(cfg.dataloader, "persistent_workers", num_workers > 0)
		)

	return DataLoader(dataset, **loader_kwargs)


def main(
	model_path: Path = Path("models/vgg16_audio.pt"),
	config_path: str = "configs",
	config_name: str = "config.yaml",
	split: str = "test",
	show_per_class: bool = False,
	save_confusion_matrix: bool = True,
	confusion_matrix_path: Path = Path("models/confusion_matrix.pt"),
):
	"""Evaluate a trained model checkpoint on the held-out test split."""

	config_file = Path(config_path) / config_name
	if not config_file.exists():
		raise FileNotFoundError(f"Config file not found: {config_file}")

	cfg = OmegaConf.load(config_file)
	if not isinstance(cfg, DictConfig):
		raise TypeError("Loaded config is not a DictConfig")

	set_seed(int(cfg.experiment.seed))

	use_cuda = bool(cfg.environment.cuda) and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	channels_last = bool(getattr(cfg.training, "channels_last", False))

	dataset = AudioDataset(cfg, "data/raw", processed_dir="data/processed")
	dataset.send_to_processed("data/processed", overwrite=False, verbose=False)

	n_total = len(dataset)
	n_train = int(cfg.splits.train * n_total)
	n_val = int(cfg.splits.val * n_total)
	n_test = n_total - n_train - n_val

	typer.echo(f"Dataset size: {n_total} | Split ratios: train={cfg.splits.train}, val={cfg.splits.val}, seed={cfg.experiment.seed}")
	typer.echo(f"Split sizes -> train: {n_train}, val: {n_val}, test: {n_test}")

	if n_test <= 0:
		raise RuntimeError("Test split size must be positive. Adjust split ratios in the config.")

	if split not in {"train", "val", "test"}:
		raise ValueError("split must be one of: train, val, test")

	# Create deterministic split - matches training exactly because:
	# 1. AudioDataset sorts files, ensuring consistent order
	# 2. Same seed and split ratios as train.py
	# 3. Evaluation remains independent - no coupling to training artifacts
	generator = torch.Generator().manual_seed(int(cfg.experiment.seed))
	train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
	selected_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
	loader = build_loader(selected_ds, cfg)

	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

	model = Model(cfg)
	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)
	model = model.to(device)
	if channels_last:
		model = model.to(memory_format=torch.channels_last)

	loss, acc, per_class, confusion_matrix = evaluate_model(
		model,
		loader,
		device,
		channels_last=channels_last,
		show_progress=True,
		num_classes=len(dataset.classes),
		class_names=list(dataset.classes),
		compute_confusion_matrix=save_confusion_matrix,
	)
	typer.echo(
		f"Evaluation complete | split: {split} | device: {device} | samples: {len(selected_ds)} | loss: {loss:.4f} | acc: {acc:.3f}"
	)
	if show_per_class and per_class:
		for name, value in per_class.items():
			typer.echo(f"class {name}: acc {value:.3f}")

	if save_confusion_matrix and confusion_matrix is not None:
		confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{"confusion_matrix": confusion_matrix, "class_names": list(dataset.classes)},
			confusion_matrix_path,
		)
		typer.echo(f"Saved confusion matrix to {confusion_matrix_path}")


if __name__ == "__main__":
	typer.run(main)
