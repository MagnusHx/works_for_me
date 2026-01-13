from pathlib import Path
import typer

from audio_emotion.dataset import download_audio_emotions
from audio_emotion.data import AudioDataset

app = typer.Typer()


@app.command()
def download(
    output_dir: Path = Path("data/raw/audio-emotions"),
):
    """Download the audio-emotions dataset from Kaggle."""
    path = download_audio_emotions(output_dir)
    typer.echo(f"Dataset available at: {path}")


@app.command()
def process(
    raw_dir: Path = Path("data/raw/audio-emotions"),
    processed_dir: Path = Path("data/processed"),
):
    """Preprocess raw audio data."""
    dataset = AudioDataset(raw_dir)
    dataset.preprocess(processed_dir)


if __name__ == "__main__":
    app()
