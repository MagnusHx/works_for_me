import pytest
import torch
from omegaconf import OmegaConf
from audio_emotion.model import Model


@pytest.fixture
def cfg():
    """Load config file."""
    return OmegaConf.load("configs/config.yaml")


@pytest.fixture
def model(cfg):
    """Create a model instance."""
    return Model(cfg)


def test_model_initialization(cfg):
    """Test if the model initializes correctly with config.yaml"""
    model = Model(cfg)
    assert model is not None
    assert isinstance(model, torch.nn.Module)
