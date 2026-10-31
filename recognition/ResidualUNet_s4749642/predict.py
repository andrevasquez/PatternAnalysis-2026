"""Load a model trained by train.py and run prostate segmentation predictions.

Example:
    python predict.py --model-name residual_unet
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import time
import matplotlib

import matplotlib.pyplot as plt

from dataset import get_dataloaders
from modules import get_model

PROSTATE_LABEL = 5  # Same label as train.py; also stored in the checkpoint

matplotlib.use("Agg")

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

    preview = parser.add_argument_group("preview options")
    preview.add_argument(
        "--patient",
        type=str,
        default=None,
        help=(
            "Test patient ID to preview, as three digits (040, 041, 042).\n"
            "If omitted, a random test patient is chosen.\n"
            "(default: random)"
        ),
    )
    preview.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Scan week to preview. Patient 040 has weeks 0-5 and 7;\n"
            "041 and 042 have week 0 only.\n"
            "If omitted, a random week for the patient is chosen.\n"
            "(default: random)"
        ),
    )
    preview.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help=(
            "Number of slices containing prostate to show, evenly\n"
            "spaced through the scan.\n"
            "(default: %(default)s)"
        ),
    )
    preview.add_argument(
        "--open",
        action="store_true",
        help=(
            "Open the saved preview image with the system's default\n"
            "application (VS Code when run from its terminal).\n"
            "(default: off)"
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

def select_preview_slices(dataset, patient_id=None, week=None, count=5):
    """Pick `count` evenly spaced slices containing prostate from one patient's scan."""
    prefix = f"case_{patient_id}_week_{week}_slice_"
    slice_number = lambda i: int(re.search(r"_slice_(\d+)", dataset.pairs[i][0].name).group(1))

    scans = sorted({
        re.match(r"case_(\d+)_week_(\d+)_", image_path.name).groups()
        for image_path, _ in dataset.pairs
    })                 

    if patient_id is None or week is None:
        options = [(p, w) for p, w in scans
                if (patient_id is None or p == patient_id)
                and (week is None or w == str(week))]
        patient_id, week = options[np.random.randint(len(options))]

    prefix = f"case_{patient_id}_week_{week}_slice_"
    scan = sorted((i for i, (image_path, _) in enumerate(dataset.pairs)
                if image_path.name.startswith(prefix)), key=slice_number)

    if not scan:
        sys.exit(f"No test slices for patient {patient_id}, week {week}.")

    with_prostate = [i for i in scan if (dataset[i][1] == PROSTATE_LABEL).any()]
    candidates = with_prostate or scan          # fall back if no prostate is visible
    picks = np.linspace(0, len(candidates) - 1, num=min(count, len(candidates)), dtype=int)
    # Return the patient and week, which may have been chosen at random.
    return [candidates[p] for p in picks], patient_id, week

def save_preview(model, dataset, indices, output_dir, model_name, device, title):
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)), squeeze=False)
    colours = dict(cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    for row, i in enumerate(indices):
        image, mask = dataset[i]
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        image, mask = image[0].numpy(), mask.numpy()

        # Crop the -100 padding so only the real slice is shown.
        rows, cols = np.where(mask != -100)
        region = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        image, mask, prediction = image[region], mask[region], prediction[region]

        slice_number = re.search(r"_slice_(\d+)", dataset.pairs[i][0].name).group(1)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 1].imshow(mask, **colours)
        axes[row, 2].imshow(prediction, **colours)
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(np.ma.masked_where(prediction == 0, prediction), alpha=0.4, **colours)
        for ax, name in zip(axes[row], ["MRI", "Ground truth", "Prediction", "Prediction overlay"]):
            ax.set_title(f"Slice {slice_number}: {name}")
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_preview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

import os, shutil, subprocess

def open_file(path):
    """Open a file with the system's default application, if one is available."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif shutil.which("code"):                      # VS Code, including remote sessions
        subprocess.run(["code", path], check=False)
    elif shutil.which("xdg-open") and os.environ.get("DISPLAY"):
        subprocess.run(["xdg-open", path], check=False)
    else:
        print("No viewer found; open the file manually:", path)


def predict(model, test_loader, output_dir, device, checkpoint):
    """Run the model on the test data and save the predictions."""

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Load inputs and their ground-truth masks
    indices = np.linspace(0, len(test_loader.dataset) - 1,
                        num=min(100, len(test_loader.dataset)), dtype=int)
    samples = [test_loader.dataset[int(i)] for i in indices]
    inputs = [image.unsqueeze(0) for image, _ in samples]
    masks = [mask for _, mask in samples]

    predictions = []
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

        # Iterate through each input and predict, saving prediction
        for cpu_image in inputs:
            image = cpu_image.to(device)
            synchronise(device)
            start = time.perf_counter()
            prediction = model(image).argmax(dim=1)
            synchronise(device)
            times_ms.append((time.perf_counter()-start)*1000)
            predictions.append(prediction.cpu())
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

    indices, patient_id, week = select_preview_slices(
        test_loader.dataset, args.patient, args.week, args.num_slices
    )
    print(f"Preview: patient {patient_id}, week {week}, {len(indices)} slices")
    path = save_preview(model, test_loader.dataset, indices, args.output_dir, args.model_name, device,
                        f"{checkpoint['model_type']} — patient {patient_id}, week {week}")
    print("Saved preview to", path)
    if args.open:
        open_file(path)


if __name__ == "__main__":
    main()
