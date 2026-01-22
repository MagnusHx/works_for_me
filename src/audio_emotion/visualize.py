from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import typer
from omegaconf import DictConfig, OmegaConf

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
		pred_label = dataset.classes[pred_idx] if pred_idx < len(dataset.classes) else str(pred_idx)
		true_label = dataset.classes[true_idx] if true_idx < len(dataset.classes) else str(true_idx)
		titles.append(f"pred: {pred_label} | true: {true_label}")
		images.append(x)

	out_path = output_dir / "prediction_grid.png"
	saved = _plot_grid(images, titles, out_path)
	typer.echo(f"Saved visualization to {saved}")


if __name__ == "__main__":
	typer.run(main)
