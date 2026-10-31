"""Load a model trained by train.py and run prostate segmentation predictions.

Example:
    python predict.py --model-name residual_unet
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import time

from dataset import get_dataloaders
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


def build_test_loader(data_dir, batch_size, device):
    """Return the test DataLoader, built the same way as in train.py."""
    _, _, test_loader = get_dataloaders(data_dir, batch_size, device)
    return test_loader

def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def predict(model, test_loader, output_dir, device, checkpoint):
    """Run the model on the test data and save the predictions."""

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load inputs
    indices = np.linspace(0, len(test_loader.dataset)-1,
                      num=min(100, len(test_loader.dataset)), dtype=int)
    inputs = [test_loader.dataset[int(i)][0].unsqueeze(0) for i in indices]

    times_ms = []
    with torch.inference_mode():
        # Warmup the model, not timed
        warmup = inputs[0].to(device)
        for _ in range(10):
            output = model(warmup).argmax(dim=1)
        del output, warmup
        
        # Reset time and memory stats
        synchronise(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Iterate through each input and predict
        for cpu_image in inputs:
            image = cpu_image.to(device)
            synchronise(device)
            start = time.perf_counter()
            prediction = model(image).argmax(dim=1)
            synchronise(device)
            times_ms.append((time.perf_counter()-start)*1000)
            del prediction, image
    
    # Gather performance metrics
    results = {"Model": checkpoint['model_type'], "Checkpoint epoch": checkpoint['epoch'],
           "Validation prostate Dice": checkpoint['val_prostate_dice'],
           "Parameters": sum(p.numel() for p in model.parameters()),
           "Inference batch size": 1, "Slices measured": len(times_ms),
           "Mean inference time (ms/slice)": np.mean(times_ms),
           "Median inference time (ms/slice)": np.median(times_ms),
           "95th percentile time (ms/slice)": np.percentile(times_ms, 95)}
    if device.type == "cuda":
        results.update({"GPU": torch.cuda.get_device_name(device),
                        "Peak inference allocated VRAM (GiB)": torch.cuda.max_memory_allocated(device)/1024**3,
                        "Peak inference reserved VRAM (GiB)": torch.cuda.max_memory_reserved(device)/1024**3})
    
    profile = pd.Series(results, name="Resource profile").to_frame()
    print(profile)

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.data_dir)

    test_loader = build_test_loader(args.data_dir, args.batch_size, device)
    print(f"Test loader: {len(test_loader.dataset)} slices, {len(test_loader)} batches")

    model, checkpoint = load_model(args.checkpoint_dir, args.model_name, device)
    print(
        f"Loaded {args.model_name} ({checkpoint['model_type']}, "
        f"epoch {checkpoint['epoch']}, "
        f"val prostate Dice {checkpoint['val_prostate_dice']:.4f})"
    )

    predict(model, test_loader, args.output_dir, device, checkpoint)


if __name__ == "__main__":
    main()
