import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from data import ValidationDataset, descriptors, read_label, training_field_paths
from models import Generator3D


def volume_statistics(volume, lags):
    fractions = np.array([np.count_nonzero(volume == phase) for phase in range(3)]) / volume.size
    size = volume.shape[0]
    indices = np.linspace(0, size - 1, min(40, size), dtype=int)
    correlations = np.zeros((2, lags))
    for index in indices:
        for section in (volume[:, index, :], volume[:, :, index]):
            _, curve = descriptors(section, lags)
            correlations += curve
    return fractions, correlations / (2 * len(indices))


def report_group(group, items, phases, curves):
    def write(unit, statistic, fractions, s2):
        phases.writerow([group, unit, statistic, *fractions])
        for lag, (horizontal, vertical) in enumerate(s2.T):
            curves.writerow([group, unit, statistic, lag, horizontal, vertical])

    for unit, fractions, s2 in items:
        write(unit, "individual", fractions, s2)
    fractions = np.stack([entry[1] for entry in items])
    curves_by_unit = np.stack([entry[2] for entry in items])
    mean_curve = curves_by_unit.mean(axis=0)
    write("", "mean", fractions.mean(axis=0), mean_curve)
    if group == "generated":
        write("", "std", fractions.std(axis=0), curves_by_unit.std(axis=0))
    return mean_curve


@torch.inference_mode()
def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    size = checkpoint["size"]
    side = args.gen_size if args.gen_size is not None else size
    factor = size // 4
    if side < factor or side % factor:
        raise ValueError(f"Generation size must be a positive multiple of {factor}.")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    lags = min(30, size // 2, side // 2)
    training = training_field_paths(args.fields, checkpoint["fields"])
    validation = ValidationDataset(args.fields, checkpoint["fields"])
    train_stats = [(p.stem, *descriptors(read_label(p), lags)) for p in training]
    val_stats = [(name, *descriptors(label, lags)) for name, label in validation]

    generator = Generator3D(size).to(args.device).eval()
    generator.load_state_dict(checkpoint["generator"])
    torch.manual_seed(args.seed)
    latent = side // factor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "results"
        staged.mkdir()
        generated = []
        for run in range(1, args.realizations + 1):
            z = torch.randn(1, 32, latent, latent, latent, device=args.device)
            volume = generator(z)[0].argmax(0).byte().cpu().numpy()
            name = f"volume_{run:03d}"
            np.save(staged / f"{name}.npy", volume)
            generated.append((name, *volume_statistics(volume, lags)))

        with (staged / "phase_fractions.csv").open("w", newline="") as phase_file, (
            staged / "s2_void.csv"
        ).open("w", newline="") as curve_file:
            phases, curves = csv.writer(phase_file), csv.writer(curve_file)
            phases.writerow(["group", "unit", "statistic", "crack", "pore", "matrix"])
            curves.writerow(["group", "unit", "statistic", "lag_px", "horizontal", "vertical"])
            report_group("train", train_stats, phases, curves)
            reference = report_group("validation", val_stats, phases, curves)
            reconstructed = report_group("generated", generated, phases, curves)

        rmse = np.sqrt(np.mean((reconstructed - reference) ** 2, axis=1))
        summary = dict(training_fields=len(training), validation_fields=len(validation),
                       realizations=len(generated), volume_side=side, seed=args.seed,
                       lags=lags, s2_rmse_horizontal=float(rmse[0]), s2_rmse_vertical=float(rmse[1]))
        (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        staged.rename(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gen-size", type=int)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.realizations < 1:
        parser.error("Realizations must be positive.")
    evaluate(args)
