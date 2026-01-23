from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from audio_emotion import data as data_module
from audio_emotion.data import AudioDataset


def _write_sine_wav(path: Path, sample_rate: int, seconds: float = 0.1) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_audio_dataset_creates_and_uses_cache(tmp_path, monkeypatch):
    """AudioDataset should preprocess a wav once and reuse the cached npy."""

    cfg = OmegaConf.load("configs/config.yaml")
    data_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    label_dir = data_root / "happy"
    label_dir.mkdir(parents=True)
    wav_path = label_dir / "sample.wav"
    _write_sine_wav(wav_path, int(cfg.audio.sample_rate))

    monkeypatch.setattr(data_module, "download_audio_emotions", lambda _: data_root)

    ds = AudioDataset(cfg, data_root, processed_dir=processed_root)
    x1, y1 = ds[0]

    cached_file = processed_root / "happy" / "sample.npy"
    assert cached_file.exists()
    assert x1.shape[0] == 1 and x1.ndim == 3
    assert y1.item() == ds.class2idx["happy"]

    # Second access should load from cache without recomputing
    cached_mtime = cached_file.stat().st_mtime
    x2, _ = ds[0]
    assert cached_file.stat().st_mtime == cached_mtime
    assert torch.allclose(x1, x2)


def test_audio_dataset_send_to_processed(tmp_path, monkeypatch, capsys):
    """send_to_processed processes all files once and skips on repeat calls."""

    cfg = OmegaConf.load("configs/config.yaml")
    data_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    label_dir = data_root / "sad"
    label_dir.mkdir(parents=True)
    wav_path = label_dir / "sample.wav"
    _write_sine_wav(wav_path, int(cfg.audio.sample_rate))

    monkeypatch.setattr(data_module, "download_audio_emotions", lambda _: data_root)

    ds = AudioDataset(cfg, data_root, processed_dir=processed_root)
    ds.send_to_processed(processed_root, overwrite=False, verbose=True)

    first_output = capsys.readouterr().out
    assert "Processed" in first_output or "Skipping" in first_output

    # Call again should skip without errors
    ds.send_to_processed(processed_root, overwrite=False, verbose=True)
    second_output = capsys.readouterr().out
    assert "Skipping" in second_output
