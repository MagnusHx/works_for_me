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
    dataset = AudioDataset(cfg,"data/raw")
    print(len(dataset))
    
    dataset.preprocess("data/processed")
    
    model = Model()
    

if __name__ == "__main__":
    train()
