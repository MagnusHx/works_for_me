from torch.utils.data import Dataset
from tests import _PATH_DATA
from audio_emotion.data import AudioDataset
from omegaconf import OmegaConf

def test_my_dataset():
    """Test the MyDataset class."""
    cfg = OmegaConf.load("configs/config.yaml")
    dataset = AudioDataset(cfg,"data/raw")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 12798
    