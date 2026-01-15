import typer
import hydra
from omegaconf import DictConfig
from pathlib import Path
from audio_emotion.model import Model
from audio_emotion.data import AudioDataset

app = typer.Typer()


@app.command()
def train(
    config_path: str = "configs",
    config_name: str = "config.yaml",
):
    config_path = str(Path(config_path).resolve())

    @hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name,
    )
    def _train(cfg: DictConfig):
        dataset = AudioDataset(cfg, "data/raw")
        print(len(dataset))

        dataset.send_to_processed("data/processed")


        model = Model(cfg)
        typer.echo("Training initialized")

        # resten af træningskoden her

    _train()


if __name__ == "__main__":
    app()
