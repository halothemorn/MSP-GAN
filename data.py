from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_EXTENSIONS = {".npy", ".png", ".tif", ".tiff"}


def field_files(directory):
    return sorted(path for path in directory.glob("*")
                  if path.is_file() and path.suffix.lower() in LABEL_EXTENSIONS)


def training_field_paths(root, patch_fields):
    directory = Path(root) / "train"
    maps = {}
    for path in field_files(directory):
        if path.stem in maps:
            raise ValueError(f"Multiple full maps for training field {path.stem} in {directory}")
        maps[path.stem] = path
    expected = set(patch_fields)
    missing = expected.difference(maps)
    extra = set(maps).difference(expected)
    if missing or extra:
        raise ValueError(f"Field IDs in {directory} must match the training patch folders. "
                         f"Missing maps: {sorted(missing)}. Maps without patches: {sorted(extra)}.")
    heldout = {path.stem.casefold() for path in field_files(Path(root) / "validation")}
    overlap = heldout.intersection(key.casefold() for key in expected)
    if overlap:
        raise ValueError(f"Full field IDs assigned to both splits: {sorted(overlap)}")
    return [maps[key] for key in sorted(expected)]


def read_label(path):
    if path.suffix.lower() == ".npy":
        label = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            label = np.array(image)
    if label.ndim != 2 or label.dtype.kind not in "iu" or not label.size:
        raise ValueError(f"Expected a 2D integer label map in {path}")
    if label.min() < 0 or label.max() > 2:
        raise ValueError(f"Expected phase IDs 0, 1 and 2 in {path}")
    return label


class TrainDataset(Dataset):
    def __init__(self, root):
        directory = Path(root) / "train"
        self.paths = sorted(p for p in directory.glob("*/*") if p.is_file()
                            and p.suffix.lower() in LABEL_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No label patches found under {directory}/FIELD_ID/")
        self.fields = sorted({path.parent.name for path in self.paths})
        self.size = read_label(self.paths[0]).shape[0]

        validation = Path(root) / "validation"
        overlap = {f.casefold() for f in self.fields} & {
            path.name.casefold() for path in validation.glob("*") if path.is_dir()}
        if overlap:
            raise ValueError(f"Fields assigned to both splits: {sorted(overlap)}")

    def label(self, index):
        path = self.paths[index]
        label = read_label(path)
        if label.shape != (self.size, self.size):
            raise ValueError(f"Expected a square patch of width {self.size} in {path}")
        return label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        label = self.label(index)
        return torch.from_numpy(np.eye(3, dtype=np.float32)[label].transpose(2, 0, 1))


class ValidationDataset(Dataset):
    def __init__(self, root, training_fields):
        self.paths = field_files(Path(root) / "validation")
        if not self.paths:
            raise ValueError("Expected full held-out label maps under fields/validation/.")
        seen = {field.casefold() for field in training_fields}
        for path in self.paths:
            field = path.stem.casefold()
            if field in seen:
                raise ValueError(f"Repeated training or validation field ID: {path.stem}")
            seen.add(field)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return path.stem, read_label(path)


def descriptors(label, lags):
    if min(label.shape) < lags:
        raise ValueError(f"Field shape {label.shape} is too small for {lags} lags.")
    fractions = np.bincount(label.ravel().astype(np.int64), minlength=3) / label.size
    void = (label < 2).astype(np.float32)
    spectrum = np.fft.rfft2(void)
    s2 = np.fft.irfft2(abs(spectrum) ** 2, s=void.shape, axes=(0, 1)) / void.size
    return fractions, np.stack([s2[0, :lags], s2[:lags, 0]])


def full_field_targets(paths, patch_size):
    lags = min(30, patch_size // 2)
    fractions = np.zeros(3)
    correlations = np.zeros((2, lags))
    for path in paths:
        label = read_label(path)
        if min(label.shape) < patch_size:
            raise ValueError(f"Full field {path} is smaller than patch width {patch_size}.")
        frac, corr = descriptors(label, lags)
        fractions += frac
        correlations += corr
    return fractions / len(paths), correlations.mean(axis=0) / len(paths)
