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
) -> tuple[float, float]:
	"""Compute average loss and accuracy for a dataloader."""

	model.eval()
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	total = 0
	correct = 0
	total_batches = len(loader)

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

		if show_progress and total_batches > 0:
			progress = min(batch_idx / total_batches, 1.0)
			bar_width = 24
			filled = int(bar_width * progress)
			bar = "=" * filled + "-" * (bar_width - filled)
			typer.echo(f"\r[{bar}] {batch_idx}/{total_batches}", nl=False)

	if show_progress and total_batches > 0:
		typer.echo("", nl=True)

	if total == 0:
		return 0.0, 0.0

	return total_loss / total, correct / total


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

	if n_test <= 0:
		raise RuntimeError("Test split size must be positive. Adjust split ratios in the config.")

	generator = torch.Generator().manual_seed(int(cfg.experiment.seed))
	_, _, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
	test_loader = build_loader(test_ds, cfg)

	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

	model = Model(cfg)
	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)
	model = model.to(device)
	if channels_last:
		model = model.to(memory_format=torch.channels_last)

	loss, acc = evaluate_model(model, test_loader, device, channels_last=channels_last, show_progress=True)
	typer.echo(
		f"Evaluation complete | device: {device} | samples: {len(test_ds)} | loss: {loss:.4f} | acc: {acc:.3f}"
	)


if __name__ == "__main__":
	typer.run(main)
