from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

from audio_emotion.download import download_audio_emotions


class AudioDataset(Dataset):
    def __init__(self, data_dir: Path) -> None:
        self.data_path = download_audio_emotions(data_dir)
        self.audio_files = sorted(self.data_path.rglob("*.wav"))

        if not self.audio_files:
            raise RuntimeError(
                f"No .wav files found after download in {self.data_path}"
            )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int):
        audio_path = self.audio_files[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        label = audio_path.parent.name
        return waveform, label

    def preprocess(self, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing data from {self.data_path} to {output_folder}")
        
        train_ratio: float = 0.8,
        seed: int = 42,