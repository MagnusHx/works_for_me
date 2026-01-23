import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from audio_emotion import visualize as viz


def test_visualize_cm_uses_cached_file(tmp_path, monkeypatch, capsys):
    """Confusion matrix should load from cache when available."""

    cm = torch.tensor([[2, 1], [0, 3]], dtype=torch.long)
    cache_path = tmp_path / "cm.pt"
    torch.save({"confusion_matrix": cm, "class_names": ["a", "b"]}, cache_path)

    # Minimal config
    cfg = OmegaConf.create(
        {
            "audio": {"sample_rate": 16000},
            "environment": {"cuda": False},
            "training": {"channels_last": False},
        }
    )

    # Fake dataset with required attributes/methods
    x = torch.randn(4, 1, 4, 4)
    y = torch.tensor([0, 1, 0, 1])

    class FakeDataset(TensorDataset):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.classes = ["a", "b"]
            self.audio_files = []

        def send_to_processed(self, *_a, **_k):
            return None

    dataset = FakeDataset(x, y)

    # Monkeypatch dataset/model and config loader
    monkeypatch.setattr(viz, "AudioDataset", lambda *_a, **_k: dataset)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 2)

        def forward(self, xb):
            return self.lin(xb.flatten(1))

    dummy_model = DummyModel()
    monkeypatch.setattr(viz, "Model", lambda _cfg: dummy_model)
    monkeypatch.setattr(viz, "_load_config", lambda *_a, **_k: cfg)

    # Save a dummy checkpoint to satisfy file existence/load
    model_path = tmp_path / "m.pt"
    torch.save(dummy_model.state_dict(), model_path)

    # Avoid recomputation; expect load path used
    viz.main(
        model_path=model_path,
        output_dir=tmp_path / "reports",
        num_samples=2,
        seed=0,
        show_confusion_matrix=True,
        confusion_matrix_path=cache_path,
        recompute_confusion=False,
    )

    out = capsys.readouterr().out
    assert "Saved visualization to" in out
