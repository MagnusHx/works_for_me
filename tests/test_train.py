import torch
from omegaconf import OmegaConf

from audio_emotion import train as train_mod


def test_train_make_block_shapes():
	cfg = OmegaConf.load("configs/config.yaml")
	block = train_mod.Model._make_block(
		in_channels=1,
		out_channels=2,
		conv_layers=2,
		kernel_size=3,
		stride=1,
		padding=1,
		pool_kernel=2,
		pool_stride=2,
	)
	# Two conv + bn + relu, then pool
	assert len(block) == 2 * 3 + 1


def test_train_set_seed_deterministic():
	train_mod.set_seed(123)
	a = torch.rand(3)
	train_mod.set_seed(123)
	b = torch.rand(3)
	assert torch.allclose(a, b)
