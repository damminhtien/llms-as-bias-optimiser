"""Pinned-format loaders for the handwriting transfer and negative-control sets."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

_EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
_KMNIST_BASE = "http://codh.rois.ac.jp/kmnist/dataset/kmnist"
_KMNIST_OPENML_URL = "https://openml.org/data/v1/download/21388379/Kuzushiji-MNIST.arff"
_KMNIST_OPENML_MD5 = "e907c57cd14de74c228fdf82f3e5e7b1"
_FASHION_BASE = "https://fashion-mnist.s3.eu-central-1.amazonaws.com"

_DATASETS = {
    "emnist_digits": {
        "kind": "emnist",
        "split": "digits",
        "train_count": 240_000,
        "test_count": 40_000,
        "classes": 10,
    },
    "emnist_letters": {
        "kind": "emnist",
        "split": "letters",
        "train_count": 124_800,
        "test_count": 20_800,
        "classes": 26,
    },
    "kmnist": {
        "kind": "idx",
        "prefix": "kmnist",
        "base_url": _KMNIST_BASE,
        "train_count": 60_000,
        "test_count": 10_000,
        "classes": 10,
    },
    "fashion_mnist": {
        "kind": "idx",
        "prefix": "fashion_mnist",
        "base_url": _FASHION_BASE,
        "train_count": 60_000,
        "test_count": 10_000,
        "classes": 10,
    },
}

_FASHION_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


@dataclass(frozen=True, slots=True)
class TransferDataset:
    """Transfer arrays with source checksums and explicit partition provenance."""

    name: str
    train_images: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    test_images: NDArray[np.float32]
    test_labels: NDArray[np.int64]
    source: str
    source_sha256: str
    preprocessing: str
    test_partition: str

    def __post_init__(self) -> None:
        for split in ("train", "test"):
            images = np.asarray(getattr(self, f"{split}_images"), dtype=np.float32)
            labels = np.asarray(getattr(self, f"{split}_labels"), dtype=np.int64)
            if images.ndim != 3 or images.shape[1:] != (28, 28):
                raise ValueError(f"{split} images must have shape (n, 28, 28)")
            if labels.shape != (len(images),):
                raise ValueError(f"{split} labels must align with images")
            if not np.isfinite(images).all() or np.any((images < 0) | (images > 1)):
                raise ValueError(f"{split} images must be finite and normalized")
            if labels.min(initial=0) < 0:
                raise ValueError(f"{split} labels must be non-negative")
            images.setflags(write=False)
            labels.setflags(write=False)
            object.__setattr__(self, f"{split}_images", images)
            object.__setattr__(self, f"{split}_labels", labels)
        if not self.name or not self.source or len(self.source_sha256) != 64:
            raise ValueError("dataset provenance fields are invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, *, referer: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; bias-optimizer/0.1)",
            **({"Referer": referer} if referer else {}),
        },
    )
    temporary: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as output,
        ):
            temporary = Path(output.name)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _ensure_emnist(root: Path, split: str) -> tuple[dict[str, Path], str]:
    archive = root / "emnist" / "gzip.zip"
    if not archive.exists():
        _download(_EMNIST_URL, archive, referer="https://www.nist.gov/")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"NIST EMNIST archive is invalid: {archive}")

    target = root / "emnist" / split
    target.mkdir(parents=True, exist_ok=True)
    names = {
        f"{part}-{kind}": (
            f"emnist-{split}-{'train' if part == 'train' else 'test'}-"
            f"{kind}-idx{3 if kind == 'images' else 1}-ubyte.gz"
        )
        for part in ("train", "t10k")
        for kind in ("images", "labels")
    }
    result: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as zipped:
        for key, member in names.items():
            output = target / f"{key}.gz"
            if not output.exists():
                try:
                    payload = zipped.read(f"gzip/{member}")
                except KeyError as exc:
                    raise ValueError(f"NIST archive is missing {member}") from exc
                output.write_bytes(payload)
            result[key] = output
    return result, _sha256(archive)


def _ensure_idx_files(
    root: Path, name: str, config: dict[str, object]
) -> tuple[dict[str, Path], str]:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    base_url = str(config["base_url"])
    files = {
        "train-images": "train-images-idx3-ubyte.gz",
        "train-labels": "train-labels-idx1-ubyte.gz",
        "t10k-images": "t10k-images-idx3-ubyte.gz",
        "t10k-labels": "t10k-labels-idx1-ubyte.gz",
    }
    result: dict[str, Path] = {}
    hashes: list[tuple[str, str]] = []
    for key, filename in files.items():
        path = directory / filename
        if not path.exists():
            _download(f"{base_url}/{filename}", path)
        if name == "fashion_mnist":
            actual_md5 = hashlib.md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest()
            if actual_md5 != _FASHION_MD5[filename]:
                raise ValueError(f"Fashion-MNIST checksum failed for {filename}")
        hashes.append((filename, _sha256(path)))
        result[key] = path
    manifest = directory / "source-checksums.json"
    manifest.write_text(
        json.dumps(dict(hashes), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    combined = hashlib.sha256(
        json.dumps(hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, combined


def _read_idx_images(path: Path) -> NDArray[np.float32]:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051 or (rows, columns) != (28, 28):
            raise ValueError(f"invalid IDX image header in {path}")
        payload = stream.read()
    if len(payload) != count * rows * columns:
        raise ValueError(f"IDX image payload size does not match its header: {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns).astype(
        np.float32
    ) / np.float32(255)


def _read_idx_labels(path: Path) -> NDArray[np.int64]:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"invalid IDX label header in {path}")
        payload = stream.read()
    if len(payload) != count:
        raise ValueError(f"IDX label payload size does not match its header: {path}")
    return np.frombuffer(payload, dtype=np.uint8).astype(np.int64)


def _load_openml_kmnist(
    root: Path,
) -> tuple[NDArray, NDArray, str, NDArray, NDArray]:
    """Use the OpenML mirror when CODH's original binary host is unreachable."""
    directory = root / "kmnist" / "openml"
    path = directory / "Kuzushiji-MNIST.arff"
    if not path.exists():
        _download(_KMNIST_OPENML_URL, path)
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    md5 = digest.hexdigest()
    if md5 != _KMNIST_OPENML_MD5:
        raise ValueError("OpenML KMNIST mirror checksum does not match dataset 41982")
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip().lower() == "@data":
                break
        else:
            raise ValueError("OpenML KMNIST ARFF file has no data section")
        values = np.loadtxt(stream, delimiter=",", dtype=np.float32, ndmin=2)
    if values.shape != (70_000, 785):
        raise ValueError(f"OpenML KMNIST data has unexpected shape {values.shape}")
    pixels = values[:, :784]
    labels = values[:, 784].astype(np.int64)
    if np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("OpenML KMNIST pixels fall outside [0, 255]")
    images = (pixels / np.float32(255)).reshape(-1, 28, 28)
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        train_size=60_000,
        random_state=42,
        stratify=labels,
    )
    checksum = _sha256(path)
    return (
        images[train_indices],
        labels[train_indices],
        checksum,
        images[test_indices],
        labels[test_indices],
    )


def load_transfer_dataset(
    name: str,
    *,
    data_dir: Path = Path("data/transfer"),
) -> TransferDataset:
    """Load one official domain, downloading and validating its IDX source files."""
    if name not in _DATASETS:
        raise ValueError(f"unknown transfer dataset {name!r}")
    root = Path(data_dir)
    config = _DATASETS[name]
    test_partition = "publisher's official test split"
    if config["kind"] == "emnist":
        split = str(config["split"])
        paths, checksum = _ensure_emnist(root, split)
        source = _EMNIST_URL
        preprocessing = "transpose image axes to correct EMNIST's published orientation"
    else:
        try:
            paths, checksum = _ensure_idx_files(root, name, config)
            source = str(config["base_url"])
            train_images = _read_idx_images(paths["train-images"])
            train_labels = _read_idx_labels(paths["train-labels"])
            test_images = _read_idx_images(paths["t10k-images"])
            test_labels = _read_idx_labels(paths["t10k-labels"])
        except OSError:
            if name != "kmnist":
                raise
            (
                train_images,
                train_labels,
                checksum,
                test_images,
                test_labels,
            ) = _load_openml_kmnist(root)
            source = _KMNIST_OPENML_URL
            test_partition = (
                "stratified 60,000/10,000 split from OpenML dataset 41982, seed 42"
            )
        preprocessing = "none; original 28x28 grayscale orientation"

    if config["kind"] == "emnist":
        train_images = _read_idx_images(paths["train-images"])
        train_labels = _read_idx_labels(paths["train-labels"])
        test_images = _read_idx_images(paths["t10k-images"])
        test_labels = _read_idx_labels(paths["t10k-labels"])

    if config["kind"] == "emnist":
        train_images = train_images.transpose(0, 2, 1).copy()
        test_images = test_images.transpose(0, 2, 1).copy()
        if name == "emnist_letters":
            train_labels -= 1
            test_labels -= 1

    if (
        len(train_images) != config["train_count"]
        or len(train_labels) != config["train_count"]
    ):
        raise ValueError(f"unexpected training split size for {name}")
    if (
        len(test_images) != config["test_count"]
        or len(test_labels) != config["test_count"]
    ):
        raise ValueError(f"unexpected test split size for {name}")
    class_count = int(config["classes"])
    if (
        train_labels.min(initial=0) < 0
        or test_labels.min(initial=0) < 0
        or train_labels.max(initial=-1) >= class_count
        or test_labels.max(initial=-1) >= class_count
    ):
        raise ValueError(f"labels fall outside the expected {class_count}-class range")

    return TransferDataset(
        name=name,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        source=source,
        source_sha256=checksum,
        preprocessing=preprocessing,
        test_partition=test_partition,
    )
