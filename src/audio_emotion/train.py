import time
import typer
import hydra
from omegaconf import DictConfig
from pathlib import Path
from audio_emotion.model import Model
from audio_emotion.data import AudioDataset

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

app = typer.Typer()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    log_every: int = 50,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    n_batches = len(loader)
    t0 = time.perf_counter()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)  # already long from dataset; safe either way

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if log_every > 0 and (step % log_every == 0 or step == 1 or step == n_batches):
            elapsed = time.perf_counter() - t0
            typer.echo(
                f"  batch {step:>5}/{n_batches} | "
                f"loss {loss.item():.4f} | "
                f"seen {total} samples | "
                f"{elapsed:.1f}s elapsed"
            )

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


@app.command()
def train(
    config_path: str = "configs",
    config_name: str = "config.yaml",
):
    config_path = str(Path(config_path).resolve())

    @hydra.main(
        version_base=None,
        config_path=config_path,
        config_name=config_name,
    )
    def _train(cfg: DictConfig):
        # ---------- Reproducibility ----------
        set_seed(int(cfg.experiment.seed))

        # ---------- Dataset ----------
        processed_dir = "data/processed"
        dataset = AudioDataset(cfg, "data/raw", processed_dir=processed_dir)
        typer.echo(f"Dataset size: {len(dataset)}")

        # ------- AB TEST -------
        typer.echo("A) Before send_to_processed")
        t0 = time.perf_counter()
        dataset.send_to_processed(processed_dir)
        typer.echo(f"B) After send_to_processed ({time.perf_counter()-t0:.1f}s)")


        # Precompute all spectrograms (so training doesn't do slow preprocessing)
        #dataset.send_to_processed(processed_dir)

        # ---------- Device ----------
        use_cuda = bool(cfg.environment.cuda) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        typer.echo(f"Using device: {device} | batch_size: {cfg.dataloader.batch_size}")

        # ---------- Split (train/val) ----------
        n_total = len(dataset)
        n_train = int(cfg.splits.train * n_total)
        n_val = int(cfg.splits.val * n_total)
        n_test = n_total - n_train - n_val

        typer.echo(f"Splits -> train: {n_train}, val: {n_val}, test: {n_test}")

        # ---------- Random split ----------
        generator = torch.Generator().manual_seed(int(cfg.experiment.seed))
        train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test], generator=generator)

        # ----- C TEST ------
        typer.echo("C) Before creating loaders")

        # ---------- Dataloaders ----------
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=bool(cfg.dataloader.shuffle),
            num_workers=0,
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
        # ------ D TEST -------
        typer.echo("D) After creating loaders")

        typer.echo(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # ---------- Smoke test (forces first batch load) ----------
        if len(train_loader) == 0:
            raise RuntimeError(
                "Train loader has 0 batches. Your train split or batch_size is causing empty training data."
            )

        #xb, yb = next(iter(train_loader))
        #typer.echo(f"First batch x shape: {tuple(xb.shape)} | y shape: {tuple(yb.shape)}")

        # ----- EF TEST ------
        typer.echo("E) Before first batch fetch")
        t1 = time.perf_counter()
        xb, yb = next(iter(train_loader))
        typer.echo(f"F) After first batch fetch ({time.perf_counter()-t1:.1f}s) | x {tuple(xb.shape)} y {tuple(yb.shape)}")

        # ---------- Model ----------
        model = Model(cfg).to(device)
        typer.echo("Training initialized")

        # ---------- Loss + Optimizer ----------
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optimizer.lr),
            weight_decay=float(cfg.optimizer.weight_decay),
        )

        # ---------- Training loop ----------
        epochs = int(cfg.training.epochs)
        typer.echo(f"Epochs: {epochs}")

        for epoch in range(1, epochs + 1):
            typer.echo(f"\n=== Epoch {epoch:02d}/{epochs} ===")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, log_every=50
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            typer.echo(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
            )

        # ---------- Save model ----------
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "models/vgg16_audio.pt")
        typer.echo("Saved model to models/vgg16_audio.pt")

    _train()


if __name__ == "__main__":
    app()
