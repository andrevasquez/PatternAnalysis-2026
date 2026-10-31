"""Train, validate, test and save a segmentation model for the HipMRI prostate task.

Example:
    python train.py --model-type residual --model-name residual_unet --data-dir ./data/keras_slices_data --epochs 20
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
        description=(
            "Train, validate, test and save a segmentation model for prostate\n"
            "segmentation on the HipMRI 2D slices.\n"
            "\n"
            "The model is trained on the training split and checked against the\n"
            "validation split after every epoch. After the final epoch it is\n"
            "evaluated once on the test split, and its weights are saved to\n"
            "<checkpoint-dir>/<model-name>.pt."
        ),
        epilog=(
            "examples:\n"
            "  python train.py --model-type baseline --model-name baseline_unet\n"
            "  python train.py --model-type residual --model-name residual_unet \\\n"
            "      --epochs 30 --batch-size 8 --lr 5e-4"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["baseline", "residual", "improved"],
        help=(
            "Model architecture to train:\n"
            "  baseline  standard U-Net\n"
            "  residual  U-Net with residual blocks\n"
            "  improved  improved U-Net variant"
        ),
    )
    required.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Name for the saved model file, without extension.\n"
            "e.g. 'residual_unet' saves to <checkpoint-dir>/residual_unet.pt.\n"
            "An existing file with the same name is overwritten."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/keras_slices_data"),
        help=(
            "Directory containing the dataset split folders\n"
            "(keras_slices_train, keras_slices_seg_train, etc.).\n"
            "Relative paths are resolved from the current directory.\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help=(
            "Directory where the trained model is saved.\n"
            "Created if it does not exist.\n"
            "(default: %(default)s)"
        ),
    )

    hyper = parser.add_argument_group("training hyperparameters")
    hyper.add_argument(
        "--epochs",
        type=int,
        default=20,
        help=(
            "Number of full passes over the training split.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of slices per batch. Lower this if the GPU\n"
            "runs out of memory.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help=(
            "Learning rate for the Adam optimiser.\n"
            "(default: %(default)s)"
        ),
    )
    hyper.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for Python, NumPy and PyTorch so runs\n"
            "can be reproduced.\n"
            "(default: %(default)s)"
        ),
    )
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


def build_model(model_type, device):
    """Create the model to train: "baseline", "residual" or "improved"."""
    # TODO: instantiate the model for model_type (see modules.ipynb)
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
    model = build_model(args.model_type, device)
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
