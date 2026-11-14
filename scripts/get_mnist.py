"""Download MNIST and cache it as a single .npz.

Pulled from a GitHub mirror of the original idx files, because the canonical
yann.lecun.com host is often unreachable.

    python scripts/get_mnist.py
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://raw.githubusercontent.com/fgnt/mnist/master"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
DATA = Path(__file__).resolve().parent.parent / "data"


def _read_idx(raw: bytes) -> np.ndarray:
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2051:                       # images
        rows, cols = struct.unpack(">II", raw[8:16])
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows * cols)
    if magic == 2049:                       # labels
        return np.frombuffer(raw, np.uint8, offset=8)
    raise ValueError(f"unexpected magic {magic}")


def load(cache=True) -> dict[str, np.ndarray]:
    DATA.mkdir(exist_ok=True)
    npz = DATA / "mnist.npz"
    if cache and npz.exists():
        with np.load(npz) as f:
            return {k: f[k] for k in f.files}

    out = {}
    for key, fname in FILES.items():
        gz = DATA / fname
        if not gz.exists():
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(f"{BASE}/{fname}", gz)
        out[key] = _read_idx(gzip.decompress(gz.read_bytes()))

    np.savez_compressed(npz, **out)
    print(f"cached to {npz}")
    return out


if __name__ == "__main__":
    d = load()
    for k, v in d.items():
        print(f"{k:<8} {v.shape} {v.dtype}")
