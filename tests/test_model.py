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


def test_model_forward_shape(cfg):
    """Model forward produces expected batch and class dimensions."""

    model = Model(cfg)
    x = torch.randn(2, int(cfg.model.in_channels), 64, 64)
    out = model(x)
    assert out.shape == (2, int(cfg.model.num_classes))


def test_model_rejects_invalid_dropout(cfg):
    """Dropout outside [0,1] should raise."""

    bad_cfg = cfg.copy()
    bad_cfg.model.dropout = 1.5
    with pytest.raises(ValueError):
        Model(bad_cfg)
