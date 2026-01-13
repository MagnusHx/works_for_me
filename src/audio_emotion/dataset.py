from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def download_audio_emotions(output_dir: Path) -> Path:
    dataset = "uldisvalainis/audio-emotions"

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Dataset already exists at {output_dir}")
        return output_dir

    api = KaggleApi()
    api.authenticate()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset} to {output_dir}...")
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)

    return output_dir
