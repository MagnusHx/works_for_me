from pathlib import Path
import typer
from torch.utils.data import Dataset
import torchaudio
import torch


class AudioDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Run the download step first."
            )
        
        self.audio_files = sorted(list(self.data_path.rglob("*.wav")))
        
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.audio_files)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        audio_path = self.audio_files[index]
        waveform, sample_rate = torchaudio.load(audio_path)
        # Extract label from parent directory name
        label = audio_path.parent.name
        return waveform, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing data from {self.data_path} to {output_folder}")
        

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = AudioDataset(data_path)
    dataset.preprocess(output_folder)


app = typer.Typer()

@app.command()
def download(dataset_id: str = "ejlok1/toronto-emotional-speech-set-tess", 
             output_dir: Path = Path("data/raw")):
    """Download dataset from Kaggle."""
    from audio_emotion.dataset import download_kaggle_dataset
    path = download_kaggle_dataset(dataset_id, output_dir)
    print(f"Downloaded to: {path}")

@app.command()
def process(raw_dir: Path = Path("data/raw"), 
            processed_dir: Path = Path("data/processed")):
    """Preprocess raw data."""
    preprocess(raw_dir, processed_dir)


if __name__ == "__main__":
    app()
