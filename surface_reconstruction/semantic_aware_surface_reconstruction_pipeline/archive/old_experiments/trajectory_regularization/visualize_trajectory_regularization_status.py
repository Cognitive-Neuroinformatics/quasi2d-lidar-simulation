#!/usr/bin/env python3
"""
Visualize where trajectory-ground regularization accepted/rejected ROAD/LANE support.

This diagnostic reproduces the V2 classification using the same parameters saved
in trajectory_ground_regularization_v2_report.json, then writes:

  regularization_status_summary.json
  regularization_status_full.npz
  regularization_status_colored.ply
  01_world_xy_status.png
  02_trajectory_sd_status.png
  03_trajectory_sz_status.png
  04_residual_vs_s.png

Color convention
----------------
green   = trusted observed ROAD/LANE, unchanged
blue    = generated ROAD/LANE accepted/projected
orange  = generated rejected, residual 0.20-0.50 m
red     = generated rejected, residual >0.50 m
purple  = observed ROAD/LANE fit outlier
gray    = selected ROAD/LANE outside the trajectory fitting corridor / untouched

The script does NOT modify the reconstruction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    0: np.array([150, 150, 150], dtype=np.uint8),  # outside corridor / untouched
    1: np.array([0, 90, 255], dtype=np.uint8),     # generated projected
    2: np.array([255, 165, 0], dtype=np.uint8),    # generated rejected moderate
    3: np.array([255, 0, 0], dtype=np.uint8),      # generated rejected severe
    4: np.array([0, 180, 0], dtype=np.uint8),      # trusted observed
    5: np.array([150, 0, 200], dtype=np.uint8),    # observed outlier
}

NAMES = {
    0: "outside_corridor_or_untouched",
    1: "generated_projected",
    2: "generated_rejected_moderate",
    3: "generated_rejected_severe",
    4: "observed_trusted_unchanged",
    5: "observed_fit_outlier_unchanged",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--caseid", required=True)
    ap.add_argument(
        "--regularizer-script",
        type=Path,
        default=Path("trajectory_regularize_ground_v2.py"),
        help="The exact V2 script used to make the regularized NPZ.",
    )
    ap.add_argument("--report-json", type=Path, default=None)
    ap.add_argument("--input-npz", type=Path, default=None)
    ap.add_argument("--family", choices=["road_lane", "other_ground", "walkable_sidewalk"], default="road_lane")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--ply-max-points", type=int, default=800000)
    ap.add_argument("--plot-max-points", type=int, default=250000)
    ap.add_argument("--full-ply", action="store_true")
    return ap.parse_args()


def import_regularizer(path: Path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("trajectory_regularize_ground_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def namespace_from_report(report):
    p = dict(report["parameters"])
    # build_surface / load_trajectory only need numeric/bool parameters, but keep everything.
    for key, value in list(p.items()):
        if key.endswith("_root") or key.endswith("_npz") or key.endswith("_dir"):
            if value is not None:
                p[key] = Path(value)
    return SimpleNamespace(**p)


def deterministic_stratified_indices(status, max_points):
    n = len(status)
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)

    out = []
    unique, counts = np.unique(status, return_counts=True)
    # Ensure every class is visible while preserving approximate proportions.
    for code, count in zip(unique, counts):
        idx = np.flatnonzero(status == code)
        quota = max(1000, int(round(max_points * count / n)))
        quota = min(quota, len(idx))
        if quota == len(idx):
            out.append(idx)
        else:
            sel = np.linspace(0, len(idx) - 1, quota).astype(np.int64)
            out.append(idx[sel])

    out = np.concatenate(out)
    if len(out) > max_points:
        sel = np.linspace(0, len(out) - 1, max_points).astype(np.int64)
        out = out[sel]
    return np.sort(out)


def write_ascii_ply(path, xyz, rgb, status, residual, s, d):
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar status\n")
        f.write("property float residual_m\n")
        f.write("property float trajectory_s_m\n")
        f.write("property float lateral_d_m\n")
        f.write("end_header\n")
        for p, c, st, r, ss, dd in zip(xyz, rgb, status, residual, s, d):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{int(st)} {float(r):.6f} {float(ss):.6f} {float(dd):.6f}\n"
            )


def scatter_plot(path, x, y, status, xlabel, ylabel, title, max_points):
    idx = deterministic_stratified_indices(status, max_points)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot gray first, then diagnostic classes on top.
    order = [0, 4, 1, 2, 3, 5]
    for code in order:
        m = status[idx] == code
        if not np.any(m):
            continue
        color = COLORS[code].astype(np.float64) / 255.0
        ax.scatter(
            x[idx][m],
            y[idx][m],
            s=2,
            alpha=0.50 if code in (0, 4) else 0.75,
            c=[color],
            label=f"{code}: {NAMES[code]}",
            rasterized=True,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(markerscale=4, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    dataset = args.dataset_root.resolve()
    caseid = args.caseid

    report_json = (
        args.report_json.resolve()
        if args.report_json
        else dataset / "recon_related" / caseid /
        "trajectory_ground_regularization_v2" /
        "trajectory_ground_regularization_v2_report.json"
    )
    input_npz = (
        args.input_npz.resolve()
        if args.input_npz
        else dataset / "recon_related" / caseid / "static_recon_labels.npz"
    )
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else dataset / "recon_related" / caseid /
        "trajectory_ground_regularization_v2" /
        "status_visualization"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not report_json.is_file():
        raise FileNotFoundError(report_json)
    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)

    report = json.loads(report_json.read_text())
    reg = import_regularizer(args.regularizer_script)
    cfg = namespace_from_report(report)

    with np.load(input_npz, allow_pickle=False) as dset:
        data = {k: np.asarray(dset[k]) for k in dset.files}

    xyz_all = np.asarray(data["xyz"], dtype=np.float64)
    sem_all = np.asarray(data["semantic_id"], dtype=np.int16)
    generated_all, generated_field = reg.detect_generated(data, cfg.generated_field)

    family_ids = reg.FAMILIES[args.family]
    global_idx = np.flatnonzero(np.isin(sem_all, family_ids))
    xyz = xyz_all[global_idx]
    generated = generated_all[global_idx]

    traj = reg.load_trajectory(
        dataset,
        caseid,
        cfg.trajectory_sample_m,
        cfg.trajectory_z_median_window_m,
        cfg.trajectory_z_smooth_window_m,
    )
    s, d = reg.trajectory_coords(xyz, traj)
    corridor = np.abs(d) <= cfg.fit_lateral_max_m
    observed = ~generated
    observed_fit = corridor & observed

    surface, trusted_fit_mask, refit_used = reg.fit_surface(
        s,
        d,
        xyz[:, 2],
        observed_fit,
        cfg,
        traj,
    )
    z_surface = reg.evaluate(surface, s, d)
    residual_signed = xyz[:, 2] - z_surface
    residual = np.abs(residual_signed)

    # Use the exact trusted observed mask returned by the V2 regularizer's
    # second-pass fit, rather than recomputing an approximation.
    trusted_observed = np.asarray(trusted_fit_mask, dtype=bool)
    observed_outlier = corridor & observed & ~trusted_observed

    gen_in_corridor = corridor & generated
    gen_project = (
        gen_in_corridor
        & (residual <= cfg.generated_project_max_residual_m)
    )
    gen_reject_moderate = (
        gen_in_corridor
        & (residual > cfg.generated_project_max_residual_m)
        & (residual <= cfg.generated_severe_residual_m)
    )
    gen_reject_severe = (
        gen_in_corridor
        & (residual > cfg.generated_severe_residual_m)
    )

    status = np.zeros(len(xyz), dtype=np.uint8)
    status[gen_project] = 1
    status[gen_reject_moderate] = 2
    status[gen_reject_severe] = 3
    status[trusted_observed] = 4
    status[observed_outlier] = 5

    rgb = np.vstack([COLORS[int(code)] for code in status])

    # Full diagnostic NPZ: compact enough to preserve exact locations/status.
    np.savez_compressed(
        out_dir / "regularization_status_full.npz",
        xyz=xyz.astype(np.float32),
        rgb=rgb,
        status=status,
        status_name=np.asarray([NAMES[int(v)] for v in status]),
        semantic_id=sem_all[global_idx],
        is_generated=generated.astype(np.uint8),
        residual_signed_m=residual_signed.astype(np.float32),
        residual_abs_m=residual.astype(np.float32),
        trajectory_s_m=s.astype(np.float32),
        lateral_d_m=d.astype(np.float32),
        fitted_surface_z=z_surface.astype(np.float32),
        source_index=global_idx.astype(np.int64),
    )

    # PLY for Open3D / CloudCompare. Downsample deterministically unless --full-ply.
    ply_idx = (
        np.arange(len(xyz), dtype=np.int64)
        if args.full_ply
        else deterministic_stratified_indices(status, args.ply_max_points)
    )
    write_ascii_ply(
        out_dir / "regularization_status_colored.ply",
        xyz[ply_idx],
        rgb[ply_idx],
        status[ply_idx],
        residual[ply_idx],
        s[ply_idx],
        d[ply_idx],
    )

    # Graphs.
    scatter_plot(
        out_dir / "01_world_xy_status.png",
        xyz[:, 0], xyz[:, 1], status,
        "world x [m]", "world y [m]",
        f"{args.family}: regularization status in world XY",
        args.plot_max_points,
    )
    scatter_plot(
        out_dir / "02_trajectory_sd_status.png",
        s, d, status,
        "trajectory arc length s [m]", "lateral d [m]",
        f"{args.family}: status in trajectory coordinates",
        args.plot_max_points,
    )
    scatter_plot(
        out_dir / "03_trajectory_sz_status.png",
        s, xyz[:, 2], status,
        "trajectory arc length s [m]", "world z [m]",
        f"{args.family}: status along road elevation profile",
        args.plot_max_points,
    )

    idx = deterministic_stratified_indices(status, args.plot_max_points)
    fig, ax = plt.subplots(figsize=(14, 8))
    for code in [0, 4, 1, 2, 3, 5]:
        m = status[idx] == code
        if not np.any(m):
            continue
        color = COLORS[code].astype(np.float64) / 255.0
        ax.scatter(
            s[idx][m],
            residual[idx][m],
            s=2,
            alpha=0.55 if code in (0, 4) else 0.75,
            c=[color],
            label=f"{code}: {NAMES[code]}",
            rasterized=True,
        )
    ax.axhline(
        cfg.generated_project_max_residual_m,
        linestyle="--",
        label=f"project threshold {cfg.generated_project_max_residual_m:.2f} m",
    )
    ax.axhline(
        cfg.generated_severe_residual_m,
        linestyle="--",
        label=f"severe threshold {cfg.generated_severe_residual_m:.2f} m",
    )
    ax.set_xlabel("trajectory arc length s [m]")
    ax.set_ylabel("|point - fitted surface| [m]")
    ax.set_ylim(
        0,
        min(
            max(1.0, float(np.percentile(residual[np.isfinite(residual)], 99.5))),
            10.0,
        ),
    )
    ax.set_title(f"{args.family}: surface residual vs trajectory position")
    ax.grid(True)
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(out_dir / "04_residual_vs_s.png", dpi=180)
    plt.close(fig)

    counts = {NAMES[code]: int(np.count_nonzero(status == code)) for code in sorted(NAMES)}
    summary = {
        "caseid": caseid,
        "family": args.family,
        "input_npz": str(input_npz),
        "report_json": str(report_json),
        "regularizer_script": str(args.regularizer_script.resolve()),
        "generated_field": generated_field,
        "refit_used": bool(refit_used),
        "points": int(len(xyz)),
        "counts": counts,
        "thresholds": {
            "observed_refit_max_residual_m": float(cfg.refit_observed_max_residual_m),
            "generated_project_max_residual_m": float(cfg.generated_project_max_residual_m),
            "generated_severe_residual_m": float(cfg.generated_severe_residual_m),
            "fit_lateral_max_m": float(cfg.fit_lateral_max_m),
        },
        "outputs": {
            "full_npz": str(out_dir / "regularization_status_full.npz"),
            "colored_ply": str(out_dir / "regularization_status_colored.ply"),
        },
    }
    (out_dir / "regularization_status_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("=" * 78)
    print("REGULARIZATION STATUS VISUALIZATION")
    print("=" * 78)
    for code in sorted(NAMES):
        print(f"{code} {NAMES[code]:40s}: {counts[NAMES[code]]:,}")
    print()
    print(f"Output folder: {out_dir}")
    print(f"PLY          : {out_dir / 'regularization_status_colored.ply'}")
    print(f"Full NPZ     : {out_dir / 'regularization_status_full.npz'}")


if __name__ == "__main__":
    main()