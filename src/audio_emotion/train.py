import hydra
from omegaconf import DictConfig
from audio_emotion.model import Model
from audio_emotion.data import AudioDataset
import torch

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config.yaml",
)

def train(cfg: DictConfig):
    dataset = AudioDataset(cfg,"data/raw")
    print(len(dataset))
    
    dataset.preprocess("data/processed")

    # ---- Model fra Hydra cfg ----
    model = Model(cfg)

    # Dette blev brugt som et sikkerhedstjekt. Det virkede første hug (:
    # x = torch.randn(2, cfg.model.in_channels, 128, 256)  # dummy spectrogram batch
    # y = model(x)
    # print("Output shape:", y.shape)  # forvent (2, cfg.model.num_classes)
    

    #waveform, label = dataset[0]
    #model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
