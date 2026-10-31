"""Load a model trained by train.py and run prostate segmentation predictions.

Example:
    python predict.py --model-name residual_unet
"""

import argparse
import sys
from pathlib import Path

import torch

from modules import get_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a model saved by train.py and use it to predict prostate\n"
            "segmentations on the HipMRI 2D slices.\n"
            "\n"
            "The model is loaded from <checkpoint-dir>/<model-name>.pt."
        ),
        epilog=(
            "examples:\n"
            "  python predict.py --model-name residual_unet\n"
            "  python predict.py --model-name baseline_unet --checkpoint-dir ./checkpoints"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Name of the model to load, as given to train.py --model-name,\n"
            "without extension. e.g. 'residual_unet' loads\n"
            "<checkpoint-dir>/residual_unet.pt."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help=(
            "Directory containing models saved by train.py.\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/keras_slices_data"),
        help=(
            "Directory containing the dataset split folders\n"
            "(keras_slices_test, keras_slices_seg_test, etc.).\n"
            "(default: %(default)s)"
        ),
    )
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help=(
            "Directory for prediction outputs.\n"
            "Created if it does not exist.\n"
            "(default: %(default)s)"
        ),
    )

    options = parser.add_argument_group("prediction options")
    options.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of slices per batch.\n"
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def load_model(checkpoint_dir, model_name, device):
    """Rebuild a model from a checkpoint saved by train.py."""
    path = checkpoint_dir / f"{model_name}.pt"
    if not path.is_file():
        sys.exit(f"Model not found: {path}")

    # The checkpoint stores settings as well as tensors, so weights_only=False.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_type" not in checkpoint:
        sys.exit(f"{path} was not saved by train.py (no model_type found).")

    model = get_model(checkpoint["model_type"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def predict(model, data_dir, batch_size, output_dir, device):
    """Run the model on the data and save the predictions."""
    # TODO
    raise NotImplementedError


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, checkpoint = load_model(args.checkpoint_dir, args.model_name, device)
    print(
        f"Loaded {args.model_name} ({checkpoint['model_type']}, "
        f"epoch {checkpoint['epoch']}, "
        f"val prostate Dice {checkpoint['val_prostate_dice']:.4f})"
    )

    predict(model, args.data_dir, args.batch_size, args.output_dir, device)


if __name__ == "__main__":
    main()
