from torch.utils.data import Dataset
from tests import _PATH_DATA
from audio_emotion.data import AudioDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = AudioDataset("data/raw")
    assert isinstance(dataset, Dataset)
