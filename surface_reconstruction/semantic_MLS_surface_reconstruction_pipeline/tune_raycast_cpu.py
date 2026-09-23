#!/usr/bin/env python3
"""Benchmark exact six-sensor CPU raycasting with several sensor-worker counts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ALL_SENSORS = ["front_left", "front_center", "front_right", "rear_left", "rear_center", "rear_right"]


def main():
    project = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--caseid", required=True)
    ap.add_argument("--reconstruction-root", required=True, type=Path)
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 3, 6])
    ap.add_argument("--start-frame", type=int, default=17)
    ap.add_argument("--end-frame", type=int, default=20)
    ap.add_argument("--point-batch-size", type=int, default=500000)
    ap.add_argument("--patch-radius", type=float, default=0.03)
    ap.add_argument("--max-range", type=float, default=80.0)
    ap.add_argument("--keep-outputs", action="store_true")
    args = ap.parse_args()

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })

    results = []
    tuning_root = args.reconstruction_root / "cpu_raycast_tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)

    for workers in args.workers:
        if workers < 1:
            continue
        output_root = tuning_root / f"workers_{workers}"
        shutil.rmtree(output_root, ignore_errors=True)
        cmd = [
            sys.executable, str(project / "raycaster" / "raycast_mls_scala2.py"),
            "--dataset-root", str(args.dataset_root),
            "--caseid", args.caseid,
            "--reconstruction-root", str(args.reconstruction_root),
            "--output-root", str(output_root),
            "--sensors", *ALL_SENSORS,
            "--start-frame", str(args.start_frame),
            "--end-frame", str(args.end_frame),
            "--first-mirror-side", "0",
            "--minimum-range", "0.5",
            "--max-range", str(args.max_range),
            "--intersection-mode", "tangent_patch",
            "--patch-radius", str(args.patch_radius),
            "--hit-radius", str(args.patch_radius),
            "--ray-neighbor-count", "0",
            "--candidate-lookup", "structured",
            "--point-batch-size", str(args.point_batch_size),
            "--static-tile-cache", "64",
            "--property-cache", "64",
            "--npz-compression", "stored",
            "--sensor-workers", str(workers),
            "--ckdtree-workers", "1",
            "--overwrite",
        ]
        print("\n" + "=" * 80)
        print(f"sensor-workers={workers}")
        print("=" * 80)
        started = time.perf_counter()
        completed = subprocess.run(cmd, env=env, check=False)
        wall = time.perf_counter() - started
        record = {"sensor_workers": workers, "wall_seconds": wall, "exit_code": completed.returncode}
        summary_path = output_root / "raycast_summary.json"
        if summary_path.is_file():
            with summary_path.open() as stream:
                summary = json.load(stream)
            frame_records = [r for frames in summary.get("sensors", {}).values() for r in frames if r.get("status") == "completed"]
            record["sensor_frames"] = len(frame_records)
            record["mean_internal_sensor_frame_seconds"] = (
                sum(float(r.get("seconds", 0.0)) for r in frame_records) / len(frame_records)
                if frame_records else None
            )
        results.append(record)
        print(f"wall={wall:.2f}s exit={completed.returncode}")
        if completed.returncode != 0:
            break
        if not args.keep_outputs:
            shutil.rmtree(output_root, ignore_errors=True)

    result_path = tuning_root / "raycast_cpu_tuning.json"
    with result_path.open("w") as stream:
        json.dump(results, stream, indent=2)

    successful = [r for r in results if r["exit_code"] == 0]
    if successful:
        best = min(successful, key=lambda r: r["wall_seconds"])
        print(f"\nFastest tested sensor-workers: {best['sensor_workers']} ({best['wall_seconds']:.2f}s)")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
