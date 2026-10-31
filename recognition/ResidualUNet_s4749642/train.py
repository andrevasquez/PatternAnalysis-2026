"""Train, validate, test and save a segmentation model for the HipMRI prostate task.

Example:
    python train.py --model-type residual --model-name residual_unet --data-dir ./data/keras_slices_data --epochs 20
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch

from tqdm.auto import tqdm

from dataset import get_dataloaders
from modules import get_model

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
            "Name for the saved model and output files, without extension.\n"
            "e.g. 'residual_unet' saves to <checkpoint-dir>/residual_unet.pt\n"
            "and <output-dir>/residual_unet_training_profile.csv.\n"
            "Existing files with the same name are overwritten."
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
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help=(
            "Directory for run outputs such as the training profile,\n"
            "saved as <output-dir>/<model-name>_training_profile.csv.\n"
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
    return get_dataloaders(data_dir, batch_size, device)


def build_model(model_type, device):
    """Create the model to train: "baseline", "residual" or "improved"."""
    return get_model(model_type).to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, training=True):
    """Run one training epoch and return (loss, prostate dice)."""
    model.train(training)
    loss_sum, pixel_count = 0.0, 0
    intersection, predicted_count, target_count = 0, 0, 0
    
    # Iterate through the dataset in batches, showing a progress bar.
    for images, masks in tqdm(loader, leave=False, mininterval=5):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
        valid = masks != -100 # Remove padded pixels
        pixels = valid.sum().item() # Get batches total loss by multiplying average loss with pixels. Convert from tensor to number.
        loss_sum += loss.item() * pixels
        pixel_count += pixels
        prediction = logits.detach().argmax(dim=1)
        pred_prostate = (prediction == PROSTATE_LABEL) & valid # Criterion for predicted prostate label.
        true_prostate = (masks == PROSTATE_LABEL) & valid # Criterion for true prostate label
        intersection += (pred_prostate & true_prostate).sum().item() # Prediction and truth both positive
        predicted_count += pred_prostate.sum().item()
        target_count += true_prostate.sum().item()
    denominator = predicted_count + target_count
    dice = 2 * intersection / denominator if denominator else float("nan")
    return loss_sum / pixel_count, dice


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a validation or test loader and return (loss, prostate dice)."""
    return train_one_epoch(model, loader, None, criterion, device, training=False)


def training_profile(device, epoch_seconds, training_start):
    """Return training time and peak GPU memory, as in the training notebooks."""
    profile = {
        "Device": str(device),
        "Completed epochs": len(epoch_seconds),
        "Recorded epoch time (min)": sum(epoch_seconds) / 60,
        "Wall time (min)": (time.perf_counter() - training_start) / 60,
    }
    if device.type == "cuda":
        profile.update({
            "GPU": torch.cuda.get_device_name(device),
            "Peak allocated VRAM (GiB)": torch.cuda.max_memory_allocated(device) / 1024**3,
            "Peak reserved VRAM (GiB)": torch.cuda.max_memory_reserved(device) / 1024**3,
        })
    return profile


def print_profile(profile):
    print("Training profile")
    width = max(len(name) for name in profile)
    for name, value in profile.items():
        # Round floats for display only; the CSV keeps full precision.
        shown = f"{value:.3f}" if isinstance(value, float) else value
        print(f"  {name:<{width}}  {shown}")


def save_profile(profile, output_dir, model_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_training_profile.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(profile.items())
    return path


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

    # Start timing and track peak PyTorch GPU memory during training and validation.
    training_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Training and validation
    epoch_seconds = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        epoch_seconds.append(time.perf_counter() - epoch_start)
        elapsed = time.perf_counter() - training_start
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f} dice {val_dice:.4f} | "
            f"{epoch_seconds[-1]:.1f}s | elapsed {elapsed / 60:.1f} min",
            flush=True,
        )

    profile = training_profile(device, epoch_seconds, training_start)
    print_profile(profile)
    print("Saved training profile to", save_profile(profile, args.output_dir, args.model_name))

    # Testing
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} dice {test_dice:.4f}")

    # Saving
    path = save_model(model, args.checkpoint_dir, args.model_name)
    print("Saved model to", path)


if __name__ == "__main__":
    main()
