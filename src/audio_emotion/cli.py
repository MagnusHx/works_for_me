from pathlib import Path
import typer

from audio_emotion.download import download_audio_emotions

app = typer.Typer()


@app.command()
def download(
    output_dir: Path = Path("data/raw/audio-emotions"),
):
    """
    Download the audio-emotions dataset from Kaggle.
    """

    if output_dir.exists() and any(output_dir.iterdir()):
        typer.echo(f"Dataset already exists at {output_dir}.")
        raise typer.Exit(code=0)

    output_dir.mkdir(parents=True, exist_ok=True)

    path = download_audio_emotions(output_dir)
    typer.echo(f"Dataset available at: {path}")
