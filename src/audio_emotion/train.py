import hydra
from omegaconf import DictConfig
from audio_emotion.model import Model
from audio_emotion.data import AudioDataset

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config.yaml",
)

def train(cfg: DictConfig):
    dataset = AudioDataset("data/raw")
    print(len(dataset))
    print(len(dataset.shape()))
    waveform, label = dataset[0]
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
