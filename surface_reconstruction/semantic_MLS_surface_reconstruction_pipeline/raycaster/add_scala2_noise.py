#!/usr/bin/env python3
"""Add SCALA2 measurement noise to already-rendered NPZ point clouds without reraycasting."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from raycast_mls_scala2 import save_npz
from scala2_noise import apply_scala2_measurement_noise


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, type=Path, help="Directory containing frame NPZ files, e.g. <raycast>/<sensor>/points")
    p.add_argument("--output-dir", type=Path, default=None, help="Default: sibling directory named points_noisy")
    p.add_argument("--range-sigma-m", type=float, default=0.05)
    p.add_argument("--azimuth-sigma-deg", type=float, default=0.1)
    p.add_argument("--polar-sigma-deg", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--npz-compression", choices=["stored", "compressed"], default="stored")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if min(a.range_sigma_m, a.azimuth_sigma_deg, a.polar_sigma_deg) < 0: p.error("Noise standard deviations must be non-negative")
    a.input_dir = a.input_dir.resolve()
    a.output_dir = (a.input_dir.parent / "points_noisy" if a.output_dir is None else a.output_dir.resolve())
    return a


def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(a.input_dir.glob("*.npz"))
    if not files: raise FileNotFoundError(f"No NPZ files found in {a.input_dir}")
    for src in files:
        dst = a.output_dir / src.name
        if dst.exists() and not a.overwrite:
            print(f"reuse {dst}"); continue
        with np.load(src, allow_pickle=False) as data: arrays = {k: np.asarray(data[k]) for k in data.files}
        noisy = apply_scala2_measurement_noise(arrays, a.range_sigma_m, a.azimuth_sigma_deg, a.polar_sigma_deg, a.seed)
        save_npz(dst, noisy, a.npz_compression); print(f"{src.name}: {len(noisy['xyz']):,} points -> {dst}")


if __name__ == "__main__": main()
