#!/usr/bin/env python3
"""Generate controlled SCALA-2 noise ablations from an already-raycast clean points/ tree."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from raycast_mls_scala2 import save_npz
from scala2_noise import apply_scala2_measurement_noise

PROFILES = {
    "range_only": {
        "range": True,
        "azimuth": False,
        "polar": False,
        "model": "datasheet_gaussian",
        "dirname": "points_noise_range_only",
    },

    "azimuth_only": {
        "range": False,
        "azimuth": True,
        "polar": False,
        "model": "datasheet_gaussian",
        "dirname": "points_noise_azimuth_only",
    },

    "polar_only": {
        "range": False,
        "azimuth": False,
        "polar": True,
        "model": "datasheet_gaussian",
        "dirname": "points_noise_polar_only",
    },

    "angular_only": {
        "range": False,
        "azimuth": True,
        "polar": True,
        "model": "datasheet_gaussian",
        "dirname": "points_noise_angular_only",
    },

    "datasheet": {
        "range": True,
        "azimuth": True,
        "polar": True,
        "model": "datasheet_gaussian",
        "dirname": "points_noise_datasheet",
    },

    "incidence": {
        "range": True,
        "azimuth": True,
        "polar": True,
        "model": "incidence_secant",
        "dirname": "points_noise_incidence",
    },
}

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raycast-root", required=True, type=Path, help="Root containing <sensor>/points/*.npz")
    ap.add_argument("--sensors", nargs="+", default=None, help="Default: every subdirectory containing points/")
    ap.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=list(PROFILES))
    ap.add_argument("--range-sigma-m", type=float, default=0.05)
    ap.add_argument("--azimuth-sigma-deg", type=float, default=0.1)
    ap.add_argument("--polar-sigma-deg", type=float, default=0.6)
    ap.add_argument("--incidence-max-angle-deg", type=float, default=75.0, help="Cap beta in sigma_r/cos(beta); avoids divergence near 90 degrees")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--npz-compression", choices=["stored", "compressed"], default="stored")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def scalar(arrays, name, default=""):
    if name not in arrays: return default
    x = np.asarray(arrays[name]).reshape(-1)
    return x[0].item() if len(x) else default


def frame_metrics(clean: dict[str, np.ndarray], noisy: dict[str, np.ndarray], profile: str, sensor: str, file_name: str):
    clean_xyz = np.asarray(clean["xyz_sensor"], dtype=np.float64)
    noisy_xyz = np.asarray(noisy["xyz_sensor"], dtype=np.float64)
    displacement = np.linalg.norm(noisy_xyz - clean_xyz, axis=1)
    abs_dr = np.abs(np.asarray(noisy["noise_delta_range_m"], dtype=np.float64))
    effective_sigma = np.asarray(noisy["noise_effective_range_sigma_m"], dtype=np.float64)
    incidence = np.asarray(noisy["incidence_angle_deg"], dtype=np.float64)
    def stat(values, fn, default=0.0): return float(fn(values)) if values.size else float(default)
    return {
        "sensor": sensor,
        "file": file_name,
        "frame_index": int(scalar(clean, "output_frame_index", -1)),
        "profile": profile,
        "points": int(len(clean_xyz)),
        "mean_displacement_m": stat(displacement, np.mean),
        "median_displacement_m": stat(displacement, np.median),
        "p95_displacement_m": stat(displacement, lambda x: np.percentile(x, 95)),
        "mean_abs_range_error_m": stat(abs_dr, np.mean),
        "p95_abs_range_error_m": stat(abs_dr, lambda x: np.percentile(x, 95)),
        "mean_effective_range_sigma_m": stat(effective_sigma, np.mean),
        "p95_effective_range_sigma_m": stat(effective_sigma, lambda x: np.percentile(x, 95)),
        "mean_incidence_angle_deg": stat(incidence, np.mean),
        "p95_incidence_angle_deg": stat(incidence, lambda x: np.percentile(x, 95)),
    }


def main():
    args = parse_args(); root = args.raycast_root.resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    sensors = args.sensors or sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "points").is_dir())
    if not sensors: raise RuntimeError(f"No sensor points/ directories found under {root}")
    rows = []
    for sensor in sensors:
        input_dir = root / sensor / "points"
        if not input_dir.is_dir(): raise FileNotFoundError(input_dir)
        files = sorted(input_dir.glob("*.npz"))
        print(f"[{sensor}] clean frames={len(files)}")
        for profile in args.profiles: (root / sensor / PROFILES[profile]["dirname"]).mkdir(parents=True, exist_ok=True)
        for i, input_path in enumerate(files, 1):
            with np.load(input_path, allow_pickle=False) as data: clean = {k: data[k] for k in data.files}
            for profile in args.profiles:
                cfg = PROFILES[profile]; output_path = root / sensor / cfg["dirname"] / input_path.name
                if output_path.exists() and not args.overwrite:
                    with np.load(output_path, allow_pickle=False) as data: noisy = {k: data[k] for k in data.files}
                else:
                    noisy = apply_scala2_measurement_noise(
                                clean,
                                args.range_sigma_m if cfg["range"] else 0.0,
                                args.azimuth_sigma_deg if cfg["azimuth"] else 0.0,
                                args.polar_sigma_deg if cfg["polar"] else 0.0,
                                args.seed,
                                model=cfg["model"],
                                incidence_max_angle_deg=args.incidence_max_angle_deg,
                            )
                    noisy["noise_ablation_profile"] = np.asarray([profile])
                    save_npz(output_path, noisy, args.npz_compression)
                rows.append(frame_metrics(clean, noisy, profile, sensor, input_path.name))
            if i % 25 == 0 or i == len(files): print(f"[{sensor}] processed {i}/{len(files)}")

    csv_path = root / "noise_ablation_per_frame.csv"
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = {}
    for profile in args.profiles:
        subset = [r for r in rows if r["profile"] == profile]
        total_points = sum(r["points"] for r in subset)
        weighted = lambda key: float(sum(r[key] * r["points"] for r in subset) / total_points) if total_points else 0.0
        aggregate[profile] = {"frames": len(subset), "points": total_points, "weighted_mean_displacement_m": weighted("mean_displacement_m"), "weighted_mean_abs_range_error_m": weighted("mean_abs_range_error_m"), "weighted_mean_effective_range_sigma_m": weighted("mean_effective_range_sigma_m"), "mean_frame_p95_displacement_m": float(np.mean([r["p95_displacement_m"] for r in subset])) if subset else 0.0}
    summary = {"schema_version": 1, "raycast_root": str(root), "profiles": args.profiles, "parameters": {"range_sigma_m": args.range_sigma_m, "azimuth_sigma_deg": args.azimuth_sigma_deg, "polar_sigma_deg": args.polar_sigma_deg, "incidence_max_angle_deg": args.incidence_max_angle_deg, "seed": args.seed}, "output_directories": {name: PROFILES[name]["dirname"] for name in args.profiles}, "aggregate": aggregate, "per_frame_csv": str(csv_path)}
    summary_path = root / "noise_ablation_summary.json"; summary_path.write_text(json.dumps(summary, indent=2))
    print("\nNoise ablation complete")
    for profile, values in aggregate.items(): print(f"  {profile:14s} points={values['points']:9d} mean_xyz_shift={values['weighted_mean_displacement_m']:.4f} m mean_abs_dr={values['weighted_mean_abs_range_error_m']:.4f} m mean_sigma_r={values['weighted_mean_effective_range_sigma_m']:.4f} m")
    print(f"Summary: {summary_path}"); print(f"Per-frame metrics: {csv_path}")


if __name__ == "__main__": main()
