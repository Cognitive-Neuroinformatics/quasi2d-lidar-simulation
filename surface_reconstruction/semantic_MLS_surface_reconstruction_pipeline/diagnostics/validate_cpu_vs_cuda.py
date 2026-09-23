#!/usr/bin/env python3
"""Compare CPU-reference and CUDA SCALA2 raycast NPZ outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-root", required=True, type=Path)
    p.add_argument("--cuda-root", required=True, type=Path)
    p.add_argument("--sensors", nargs="+", default=["front_center"])
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--end-frame", type=int, required=True, help="Exclusive")
    p.add_argument("--range-atol", type=float, default=1e-5)
    p.add_argument("--xyz-atol", type=float, default=1e-5)
    return p.parse_args()


def compare_file(cpu_path: Path, cuda_path: Path, args):
    with np.load(cpu_path, allow_pickle=False) as a, np.load(cuda_path, allow_pickle=False) as b:
        out = {
            "cpu_hits": len(a["xyz"]),
            "cuda_hits": len(b["xyz"]),
            "same_ray_index": np.array_equal(a["ray_index"], b["ray_index"]),
            "same_semantic": np.array_equal(a["semantic_id"], b["semantic_id"]),
            "same_instance": np.array_equal(a["instance_id"], b["instance_id"]),
            "same_source_type": np.array_equal(a["source_type"], b["source_type"]),
            "same_source_object_id": np.array_equal(a["source_object_id"], b["source_object_id"]),
            "same_ground_id": np.array_equal(a["ground_id"], b["ground_id"]),
            "same_intensity": np.array_equal(a["intensity"], b["intensity"], equal_nan=True),
        }
        if len(a["range_m"]) == len(b["range_m"]):
            dr = np.abs(a["range_m"].astype(np.float64) - b["range_m"].astype(np.float64))
            dxyz = np.linalg.norm(a["xyz"].astype(np.float64) - b["xyz"].astype(np.float64), axis=1)
            out.update({
                "range_max_diff": float(dr.max(initial=0.0)),
                "range_mean_diff": float(dr.mean()) if len(dr) else 0.0,
                "range_within_tolerance": bool(np.all(dr <= args.range_atol)),
                "xyz_max_diff": float(dxyz.max(initial=0.0)),
                "xyz_mean_diff": float(dxyz.mean()) if len(dxyz) else 0.0,
                "xyz_within_tolerance": bool(np.all(dxyz <= args.xyz_atol)),
            })
        else:
            out.update({
                "range_max_diff": float("inf"),
                "range_mean_diff": float("inf"),
                "range_within_tolerance": False,
                "xyz_max_diff": float("inf"),
                "xyz_mean_diff": float("inf"),
                "xyz_within_tolerance": False,
            })

        strict = (
            out["cpu_hits"] == out["cuda_hits"]
            and out["same_ray_index"]
            and out["same_semantic"]
            and out["same_instance"]
            and out["same_source_type"]
            and out["same_source_object_id"]
            and out["same_ground_id"]
            and out["same_intensity"]
            and out["range_within_tolerance"]
            and out["xyz_within_tolerance"]
        )
        out["pass"] = strict
        return out


def main():
    args = parse_args()
    all_ok = True
    print("=" * 96)
    print("CPU vs CUDA SCALA2 RAYCAST VALIDATION")
    print("=" * 96)
    for sensor in args.sensors:
        for frame in range(args.start_frame, args.end_frame):
            cpu = args.cpu_root / sensor / "points" / f"{frame:03d}.npz"
            cuda = args.cuda_root / sensor / "points" / f"{frame:03d}.npz"
            if not cpu.is_file() or not cuda.is_file():
                print(f"{sensor} frame {frame:03d}: MISSING cpu={cpu.is_file()} cuda={cuda.is_file()}")
                all_ok = False
                continue
            r = compare_file(cpu, cuda, args)
            all_ok &= r["pass"]
            print(
                f"{sensor} frame {frame:03d}: {'PASS' if r['pass'] else 'DIFF'} "
                f"hits={r['cpu_hits']}/{r['cuda_hits']} ray={r['same_ray_index']} "
                f"sem={r['same_semantic']} inst={r['same_instance']} source={r['same_source_type']} "
                f"range_max={r['range_max_diff']:.9g} xyz_max={r['xyz_max_diff']:.9g}"
            )
    print("=" * 96)
    print("OVERALL:", "PASS" if all_ok else "DIFFERENCES FOUND")
    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
