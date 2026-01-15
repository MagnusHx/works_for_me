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

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss= running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)

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
        dataset = AudioDataset(cfg, "data/raw")
        typer.echo(f"Dataset size: {len(dataset)}")

        dataset.send_to_processed("data/processed")

        # ---------- Device ----------
        use_cuda = bool(cfg.environment.cuda) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        typer.echo(f"Using device: {device}")

        # ---------- Split (train/val) ----------
        n_total = len(dataset)
        n_train = int(cfg.splits.train * n_total)
        n_val = int(cfg.splits.val * n_total)
        n_test = n_total - n_train - n_val 

        # ---------- Random split ----------
        generator = torch.Generator().manual_seed(int(cfg.experiment.seed))
        train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test], generator=generator)

        # ---------- Dataloaders ----------
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=bool(cfg.dataloader.shuffle),
            num_workers=0,      # sæt evt >0 senere
            pin_memory=use_cuda,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg.dataloader.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )

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
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            typer.echo(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
            )

        # ---------- Save model ----------
        # Gem i models/ (lav mappen hvis den ikke findes)
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "models/vgg16_audio.pt")
        typer.echo("Saved model to models/vgg16_audio.pt")

    _train()



if __name__ == "__main__":
    app()
