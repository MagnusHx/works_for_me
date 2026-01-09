from pathlib import Path
import kagglehub
import typer
from torch.utils.data import Dataset

<<<<<<< HEAD

class AudioDataset(Dataset):
=======
class MyDataset(Dataset):
>>>>>>> f0deec7 (Remove unnecessary blank line in data.py)
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:

    # Download latest version
        path = kagglehub.dataset_download("uldisvalainis/audio-emotions")

        print("Path to dataset files:", path)
        
        self.data_path = path
        

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = AudioDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(AudioDataset)
