import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audio_emotion.evaluate import evaluate_model, build_loader


def test_evaluate_model_computes_metrics_and_confusion():
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    model = nn.Linear(4, 2)

    loss, acc, per_class, cm = evaluate_model(
        model,
        loader,
        device=torch.device("cpu"),
        channels_last=False,
        show_progress=False,
        num_classes=2,
        class_names=["a", "b"],
        compute_confusion_matrix=True,
    )

    assert loss >= 0
    assert 0 <= acc <= 1
    assert set(per_class.keys()) == {"a", "b"}
    assert cm is not None and cm.shape == (2, 2)
    assert int(cm.sum().item()) == len(y)


def test_build_loader_uses_cfg_settings():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "batch_size": 3,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": 2,
                "persistent_workers": False,
            }
        }
    )

    dataset = TensorDataset(torch.randn(5, 2), torch.zeros(5, dtype=torch.long))
    loader = build_loader(dataset, cfg)

    batches = list(loader)
    assert len(batches) == 2  # ceil(5/3)
    for xb, yb in batches:
        assert xb.shape[0] <= 3
        assert yb.dtype == torch.long


def test_evaluate_main_smoke(tmp_path, monkeypatch):
    """Run evaluate.main against a tiny fake dataset and model."""

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "experiment": {"seed": 0},
            "environment": {"cuda": False},
            "training": {"channels_last": False},
            "dataloader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "splits": {"train": 0.5, "val": 0.25, "test": 0.25},
        }
    )

    # Fake dataset with required attributes and send_to_processed
    x = torch.randn(8, 1, 4, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    class FakeDataset(TensorDataset):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.classes = ["a", "b"]

        def send_to_processed(self, *_a, **_k):
            return None

    dataset = FakeDataset(x, y)

    # Monkeypatch AudioDataset and Model
    monkeypatch.setattr(
        "audio_emotion.evaluate.AudioDataset", lambda *_a, **_k: dataset
    )

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 2)

        def forward(self, x):
            return self.lin(x.flatten(1))

    dummy_model = DummyModel()
    monkeypatch.setattr("audio_emotion.evaluate.Model", lambda _cfg: dummy_model)

    # Monkeypatch loader builder to use our dataset and cfg
    monkeypatch.setattr(
        "audio_emotion.evaluate.build_loader",
        lambda ds, _cfg: DataLoader(ds, batch_size=2),
    )

    # Save dummy model state
    model_path = tmp_path / "m.pt"
    torch.save(dummy_model.state_dict(), model_path)

    # Write config to disk for OmegaConf.load
    config_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, config_path)

    # Import main lazily to avoid typer.run; call directly
    from audio_emotion import evaluate as eval_mod

    eval_mod.main(
        model_path=model_path,
        config_path=str(config_path.parent),
        config_name=config_path.name,
        split="test",
        show_per_class=True,
        save_confusion_matrix=False,
    )
