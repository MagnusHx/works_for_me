from pathlib import Path
from torch.utils.data import Dataset
import librosa
import numpy as np
import torch
from omegaconf import DictConfig

from audio_emotion.download import download_audio_emotions


class AudioDataset(Dataset):
    def __init__(self, cfg: DictConfig, data_dir: str | Path, processed_dir: str | Path = "data/processed") -> None:
        self.cfg = cfg

        self.data_path = download_audio_emotions(Path(data_dir))
        self.audio_files = sorted(self.data_path.rglob("*.wav"))
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found after download in {self.data_path}")

        # label -> int mapping (e.g. "happy" -> 0)
        self.classes = sorted({p.parent.name for p in self.audio_files})
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        self.processed_dir = Path(processed_dir)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int):
        audio_path = self.audio_files[index]
        label_str = audio_path.parent.name
        y = self.class2idx[label_str]

        # Load precomputed spectrogram (.npy)
        npy_path = self.processed_dir / label_str / f"{audio_path.stem}.npy"

        # If not found, compute it on-the-fly (safer than crashing)
        if not npy_path.exists():
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            (self.processed_dir / label_str).mkdir(parents=True, exist_ok=True)
            spec = self.preprocess(audio_path)
            np.save(npy_path, spec)

        x = torch.from_numpy(np.load(npy_path)).float()  # shape: (1, F, T)
        return x, torch.tensor(y, dtype=torch.long)

    #

    def preprocess(
        self,
        wav_path: str | Path,
        clip_seconds: float = 4.0,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int | None = 400,
        center: bool = True,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Preprocess an audio file into a log-magnitude STFT spectrogram for the CNN."""
        sample_rate = self.cfg.audio.sample_rate
        y, sr = librosa.load(
            wav_path,
            sr=sample_rate,
            mono=True,
            offset=0.0,
            duration=clip_seconds,
            dtype=np.float32,
        )

        target_len = int(sr * clip_seconds)
        if y.shape[0] < target_len:
            y = np.pad(y, (0, target_len - y.shape[0]), mode="constant")
        elif y.shape[0] > target_len:
            y = y[:target_len]

        S = librosa.stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=center,
        )

        mag = np.abs(S)
        log_mag = np.log(mag + epsilon)

        return log_mag[np.newaxis, :, :].astype(np.float32)

    def send_to_processed(
        self,
        output_folder: str | Path = "data/processed",
        overwrite: bool = False,
        verbose: bool = True,
    ) -> Path:
        """Preprocess all audio files and save spectrograms, skipping ones already processed."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: keep dataset pointing to the folder we write to
        self.processed_dir = output_folder

        missing: list[Path] = []
        for audio_file in self.audio_files:
            label = audio_file.parent.name
            output_path = output_folder / label / f"{audio_file.stem}.npy"
            if overwrite or not output_path.exists():
                missing.append(audio_file)

        if not missing:
            if verbose:
                print(f"All {len(self.audio_files)} files already processed in {output_folder}. Skipping.")
            return output_folder

        processed_count = 0
        skipped_count = len(self.audio_files) - len(missing)

        for audio_file in missing:
            label = audio_file.parent.name
            label_folder = output_folder / label
            label_folder.mkdir(parents=True, exist_ok=True)

            spectrogram = self.preprocess(audio_file)

            output_path = label_folder / f"{audio_file.stem}.npy"
            np.save(output_path, spectrogram)
            processed_count += 1

        if verbose:
            print(f"Processed {processed_count} files to {output_folder} (skipped {skipped_count} already-processed).")

        return output_folder
