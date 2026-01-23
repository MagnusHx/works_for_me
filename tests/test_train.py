import torch

from audio_emotion import train as train_mod


def test_train_make_block_shapes():
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


def test_save_learning_curve_creates_outputs(tmp_path, monkeypatch):
    """save_learning_curve should write metrics.json and plot."""

    # Redirect outputs directory
    monkeypatch.setattr(
        train_mod, "Path", lambda p=".": __import__("pathlib").Path(tmp_path) / p
    )

    history = {
        "train_acc": [0.5, 0.6],
        "val_acc": [0.4, 0.55],
    }

    plot_path = train_mod.save_learning_curve(history)
    metrics_path = plot_path.parent / "metrics.json"
    assert plot_path.exists()
    assert metrics_path.exists()


def test_train_one_epoch_and_evaluate_smoke():
    """Basic smoke for train_one_epoch and evaluate using tiny data and model."""

    device = torch.device("cpu")
    model = torch.nn.Linear(4, 2)
    data_x = torch.randn(6, 4)
    data_y = torch.tensor([0, 1, 0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_x, data_y), batch_size=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_loss, train_acc = train_mod.train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
        log_every=0,
        scaler=None,
        autocast_kwargs={"enabled": False},
    )
    assert train_loss >= 0
    assert 0 <= train_acc <= 1

    eval_loss, eval_acc = train_mod.evaluate(
        model, loader, criterion, device, autocast_kwargs={"enabled": False}
    )
    assert eval_loss >= 0
    assert 0 <= eval_acc <= 1
