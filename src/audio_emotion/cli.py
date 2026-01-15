from pathlib import Path
import typer

from audio_emotion.download import download_audio_emotions

app = typer.Typer()


@app.command()
def download(
    output_dir: Path = Path("data/raw/audio-emotions"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download the dataset even if it already exists.",
    ),
):
    """
    Download the audio-emotions dataset from Kaggle.
    """

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        typer.echo(
            f"Dataset already exists at {output_dir}. "
            "Use --force to re-download."
        )
        raise typer.Exit(code=0)

    output_dir.mkdir(parents=True, exist_ok=True)

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
