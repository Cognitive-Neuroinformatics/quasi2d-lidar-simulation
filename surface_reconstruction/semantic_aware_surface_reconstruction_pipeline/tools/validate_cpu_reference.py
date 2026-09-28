#!/usr/bin/env python3
"""Compare raycast outputs from two roots and fail on any requested mismatch."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

DEFAULT_ARRAYS = (
    "ray_index", "range_m", "xyz", "xyz_world", "semantic_id", "ground_id",
    "instance_id", "source_type", "source_object_id", "intensity",
)


def compare_array(a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {a.shape}/{a.dtype} != {b.shape}/{b.dtype}"
    if np.issubdtype(a.dtype, np.floating):
        equal = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        if a.size:
            finite = np.isfinite(a) & np.isfinite(b)
            max_abs = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))) if np.any(finite) else 0.0
        else:
            max_abs = 0.0
        return bool(equal), f"max_abs={max_abs:.9g}"
    return bool(np.array_equal(a, b)), "exact"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference_root", type=Path)
    ap.add_argument("candidate_root", type=Path)
    ap.add_argument("--sensors", nargs="*", default=None)
    ap.add_argument("--arrays", nargs="+", default=list(DEFAULT_ARRAYS))
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    if args.sensors:
        sensors = args.sensors
    else:
        sensors = sorted(
            item.name for item in args.reference_root.iterdir()
            if item.is_dir() and (item / "points").is_dir()
        )

    failures = 0
    compared = 0
    for sensor in sensors:
        ref_dir = args.reference_root / sensor / "points"
        cand_dir = args.candidate_root / sensor / "points"
        for ref_path in sorted(ref_dir.glob("*.npz")):
            cand_path = cand_dir / ref_path.name
            if not cand_path.is_file():
                print(f"MISSING {sensor}/{ref_path.name}")
                failures += 1
                continue
            with np.load(ref_path, allow_pickle=False) as a, np.load(cand_path, allow_pickle=False) as b:
                frame_ok = True
                for name in args.arrays:
                    if name not in a.files or name not in b.files:
                        print(f"MISSING ARRAY {sensor}/{ref_path.name}: {name}")
                        failures += 1
                        frame_ok = False
                        continue
                    ok, detail = compare_array(np.asarray(a[name]), np.asarray(b[name]), args.atol, args.rtol)
                    if not ok:
                        print(f"DIFF {sensor}/{ref_path.name} {name}: {detail}")
                        failures += 1
                        frame_ok = False
                if frame_ok:
                    print(f"OK {sensor}/{ref_path.name}")
                compared += 1

    print(f"Compared frames: {compared}, failures: {failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
