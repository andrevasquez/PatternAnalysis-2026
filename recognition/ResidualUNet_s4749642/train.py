"""Train, validate, test and save a segmentation model for the HipMRI prostate task.

Example:
    python train.py --model-name residual_unet --data-dir ./data/keras_slices_data --epochs 20
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

# Setup constants
NUM_CLASSES = 6
PROSTATE_LABEL = 5  # Other labels can be found in dataset notebook


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, validate, test and save a segmentation model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/keras_slices_data"),
        help="Directory containing the dataset (default: ./data/keras_slices_data)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name used for the saved model file, e.g. residual_unet",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory where the trained model is saved (default: ./checkpoints)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def set_seed(seed):
    # Fix random seeds to make experiments more reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, batch_size, device):
    """Return train, validation and test DataLoaders."""
    # TODO: build HipMRIDataset splits from data_dir (see dataset.ipynb)
    raise NotImplementedError


def build_model(device):
    """Create the model to train."""
    # TODO: instantiate the model (see modules.ipynb)
    raise NotImplementedError


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return (loss, prostate dice)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a validation or test loader and return (loss, prostate dice)."""
    # TODO
    raise NotImplementedError


def save_model(model, checkpoint_dir, model_name):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{model_name}.pt"
    torch.save(model.state_dict(), path)
    return path


def main():
    args = parse_args()
    set_seed(args.seed)

    # Normal device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.data_dir)

    train_loader, val_loader, test_loader = build_loaders(args.data_dir, args.batch_size, device)
    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Ignore artificial padding added by the dataset loader.
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Training and validation
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f}"
        )

    # Testing
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} dice {test_dice:.4f}")

    # Saving
    path = save_model(model, args.checkpoint_dir, args.model_name)
    print("Saved model to", path)


if __name__ == "__main__":
    main()
