#!/usr/bin/env python3
"""Run and profile one complete CPU scene: densification -> MLS -> SCALA2 raycasting.

This launcher intentionally keeps the scientific parameters unchanged while
using the exact CPU optimizations in this package.  It writes one benchmark
folder containing terminal logs, /usr/bin/time -v resource reports, detailed
stage timings and a machine-readable summary for later CPU-vs-CUDA comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ALL_SENSORS = [
    "front_left", "front_center", "front_right",
    "rear_left", "rear_center", "rear_right",
]


def run_stage(name: str, command: list[str], env: dict[str, str], log_dir: Path):
    log_path = log_dir / f"{name}.log"
    resource_path = log_dir / f"{name}.resources.txt"
    wrapped = ["/usr/bin/time", "-v", "-o", str(resource_path), *command]
    print("\n" + "=" * 80)
    print(f"STAGE: {name}")
    print("=" * 80)
    print("COMMAND:")
    print(" ".join(command))
    print("=" * 80)
    started = time.perf_counter()
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(
            wrapped,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    wall = time.perf_counter() - started
    print(f"[{name}] wall time: {wall:.2f} s, exit={return_code}")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return {
        "wall_seconds": wall,
        "command": command,
        "log": str(log_path),
        "resource_report": str(resource_path),
    }


def load_json_if_exists(path: Path):
    if not path.is_file():
        return None
    with path.open() as stream:
        return json.load(stream)


def summarize_raycast(path: Path):
    data = load_json_if_exists(path)
    if data is None:
        return None
    completed = []
    for sensor, frames in data.get("sensors", {}).items():
        for record in frames:
            if record.get("status") == "completed":
                completed.append((sensor, record))
    if not completed:
        return {"completed_sensor_frames": 0}

    records = [record for _, record in completed]
    seconds = [float(record.get("seconds", 0.0)) for record in records]
    hits = [int(record.get("hits", 0)) for record in records]
    timing_keys = sorted({
        key
        for record in records
        for key in record.get("timing_breakdown_s", {})
    })
    timing_sum = {
        key: float(sum(record.get("timing_breakdown_s", {}).get(key, 0.0) for record in records))
        for key in timing_keys
    }
    timing_mean = {
        key: value / len(records)
        for key, value in timing_sum.items()
    }
    return {
        "completed_sensor_frames": len(records),
        "sum_sensor_frame_seconds": float(sum(seconds)),
        "mean_sensor_frame_seconds": float(sum(seconds) / len(seconds)),
        "min_sensor_frame_seconds": float(min(seconds)),
        "max_sensor_frame_seconds": float(max(seconds)),
        "mean_hits": float(sum(hits) / len(hits)),
        "timing_breakdown_sum_s": timing_sum,
        "timing_breakdown_mean_s": timing_mean,
        "static_geometry_io_sum_s": float(sum(record.get("static_geometry_io_seconds", 0.0) for record in records)),
        "output_write_sum_s": float(sum(record.get("output_write_seconds", 0.0) for record in records)),
        "points_examined_sum": int(sum(record.get("points_examined", 0) for record in records)),
        "candidate_pairs_sum": int(sum(record.get("candidate_pairs", 0) for record in records)),
    }


def main():
    project = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset-root", type=Path,
        default=Path("/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study"),
    )
    ap.add_argument(
        "--caseid",
        default="segment-17791493328130181905_1480_000_1500_000_with_camera_labels",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--clean", action="store_true", help="Delete only this benchmark's optimized reconstruction/raycast outputs before the run")
    ap.add_argument("--reconstruction-root", type=Path, default=None, help="Optimized MLS output root. Default: <dataset>/semantic_aware_mls/semantic_static_mls_cpu_optimized/<case>")
    ap.add_argument("--densified-npz", type=Path, default=None, help="Optimized densified static NPZ. Default: <static_filter>/static_recon_labels_strict_densified_cpu_optimized.npz")
    ap.add_argument("--support-npz", type=Path, default=None, help="Optimized generated-support NPZ. Default: <static_filter>/scene_ground_support_cpu_optimized.npz")
    ap.add_argument("--skip-densify", action="store_true")
    ap.add_argument("--skip-reconstruct", action="store_true")
    ap.add_argument("--skip-raycast", action="store_true")
    ap.add_argument("--write-densified-pcd", action="store_true", help="PCD is not consumed by MLS; disabled by default for timing")
    ap.add_argument("--npz-compression", choices=["stored", "compressed"], default="stored")
    ap.add_argument("--fill-spacing-mode", choices=["fixed", "local_ring_density", "along_ring_density"], default="local_ring_density")
    ap.add_argument("--pcl-threads", type=int, default=8)
    ap.add_argument("--mls-workers", type=int, default=3)
    ap.add_argument("--attribute-workers", type=int, default=1)
    ap.add_argument("--pcl-work-root", type=Path, default=None, help="Explicit temporary PCL PCD directory. If omitted, /dev/shm is used automatically when enough space is available.")
    ap.add_argument("--no-auto-shm", action="store_true", help="Do not automatically use /dev/shm for temporary PCL input/output PCD files")
    ap.add_argument("--shm-min-free-gb", type=float, default=8.0, help="Minimum free /dev/shm space required for automatic RAM-disk PCL work files")
    ap.add_argument("--raycast-sensors", nargs="+", choices=ALL_SENSORS, default=ALL_SENSORS)
    ap.add_argument("--sensor-workers", type=int, default=3)
    ap.add_argument("--raycast-start-frame", type=int, default=0)
    ap.add_argument("--raycast-end-frame", type=int, default=None)
    ap.add_argument("--point-batch-size", type=int, default=500000)
    ap.add_argument("--patch-radius", type=float, default=0.03)
    ap.add_argument("--maximum-range", type=float, default=80.0)
    args = ap.parse_args()

    dataset = args.dataset_root.resolve()
    case = args.caseid
    static_dir = dataset / "recon_related" / case / "static_filter"
    dynamic_root = dataset / "temp" / case / "occ" / "preproc" / "dynamic" / "objects"
    strict_npz = static_dir / "static_recon_labels_strict.npz"
    support_npz = (args.support_npz.resolve() if args.support_npz is not None else static_dir / "scene_ground_support_cpu_optimized.npz")
    densified_npz = (args.densified_npz.resolve() if args.densified_npz is not None else static_dir / "static_recon_labels_strict_densified_cpu_optimized.npz")
    densified_pcd = densified_npz.with_suffix(".pcd")
    reconstruction_root = (
        args.reconstruction_root.resolve() if args.reconstruction_root is not None else
        dataset / "semantic_aware_mls" / "semantic_static_mls_cpu_optimized" / case
    )
    raycast_root = reconstruction_root / "scala2_raycast_cpu_optimized"

    # PCL currently exchanges binary PCD files with the Python wrapper for every
    # independent MLS job. On Linux, a sufficiently large /dev/shm removes the
    # physical-disk round trip while preserving identical PCL input/output.
    selected_pcl_work_root = args.pcl_work_root.resolve() if args.pcl_work_root is not None else None
    auto_shm = False
    if selected_pcl_work_root is None and not args.no_auto_shm:
        shm = Path("/dev/shm")
        if shm.is_dir():
            free_bytes = shutil.disk_usage(shm).free
            if free_bytes >= int(args.shm_min_free_gb * (1024 ** 3)):
                selected_pcl_work_root = shm / f"semantic_static_mls_{os.getpid()}"
                selected_pcl_work_root.mkdir(parents=True, exist_ok=True)
                auto_shm = True
                print(
                    f"Using RAM disk for temporary PCL PCD files: {selected_pcl_work_root} "
                    f"({free_bytes / (1024 ** 3):.1f} GiB free)"
                )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = dataset / "semantic_aware_mls" / "cpu_benchmarks" / case / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        if reconstruction_root.exists():
            print(f"Removing previous optimized reconstruction: {reconstruction_root}")
            shutil.rmtree(reconstruction_root)
        # The strict source is never touched. Optimized densification outputs
        # use separate filenames and are overwritten by the densification stage.

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "case": case,
        "dataset_root": str(dataset),
        "project_root": str(project),
        "settings": vars(args) | {
            "dataset_root": str(dataset),
            "reconstruction_root": str(reconstruction_root),
            "densified_npz": str(densified_npz),
            "support_npz": str(support_npz),
            "pcl_work_root": None if selected_pcl_work_root is None else str(selected_pcl_work_root),
            "pcl_work_root_auto_shm": bool(auto_shm),
        },
        "paths": {
            "strict_npz": str(strict_npz),
            "support_npz": str(support_npz),
            "densified_npz": str(densified_npz),
            "reconstruction_root": str(reconstruction_root),
            "raycast_root": str(raycast_root),
            "log_dir": str(log_dir),
        },
        "stages": {},
    }

    # Snapshot machine information for the CPU-vs-CUDA paper table later.
    for command, filename in [
        (["lscpu"], "lscpu.txt"),
        (["free", "-h"], "memory.txt"),
        (["nvidia-smi"], "nvidia_smi.txt"),
    ]:
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            (log_dir / filename).write_text(result.stdout)
        except FileNotFoundError:
            pass

    common_env = os.environ.copy()
    common_env["PYTHONUNBUFFERED"] = "1"

    pipeline_started = time.perf_counter()

    if not args.skip_densify:
        densify_timing = log_dir / "densify_timing.json"
        command = [
            args.python, str(project / "preprocessing" / "reconstruct_static_densified_pointcloud.py"),
            "-s", str(dataset), "--caseid", case,
            "--strict_path", str(strict_npz), "-o", str(support_npz),
            "--generic_coverage",
            "--analysis_voxel", "0.05",
            "--fill_spacing", "0.03",
            "--fill_spacing_mode", args.fill_spacing_mode,
            "--adaptive_density_voxel", "0.01",
            "--adaptive_density_neighbors", "3",
            "--adaptive_query_neighbors", "16",
            "--adaptive_fill_scale", "1.0",
            "--adaptive_fill_min", "0.01",
            "--adaptive_fill_max", "0.08",
            "--min_separation", "0.01",
            "--deduplicate", "--dedup_spacing", "0.005",
            "--max_candidates_per_family", "0",
            "--max_support", "0",
            "--densified_npz", str(densified_npz),
            "--npz_compression", args.npz_compression,
            "--timing_json", str(densify_timing),
        ]
        if args.write_densified_pcd:
            command.extend(["--densified_pcd", str(densified_pcd)])
        env = common_env.copy()
        # cKDTree uses its own workers=-1. Keep BLAS libraries single-threaded
        # so they do not compete with SciPy's CPU pool on tiny 3x3 operations.
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        summary["stages"]["densify"] = run_stage("densify", command, env, log_dir)
        summary["stages"]["densify"]["detail"] = load_json_if_exists(densify_timing)

    if not args.skip_reconstruct:
        command = [
            args.python, str(project / "reconstruct_semantic_static_mls.py"),
            "--dataset-root", str(dataset),
            "--caseid", case,
            "--static-input", str(densified_npz),
            "--dynamic-input-root", str(dynamic_root),
            "--pcl-executable", str(project / "build_pcl_mls" / "pcl_mls_reconstruct"),
            "--config", str(project / "semantic_static_mls_v1.json"),
            "--output-root", str(reconstruction_root),
            "--stages", "background", "static_objects", "dynamic_objects",
            "--tile-size", "25", "--tile-halo", "0.5",
            "--pcl-threads", str(args.pcl_threads),
            "--mls-workers", str(args.mls_workers),
            "--attribute-workers", str(args.attribute_workers),
            "--minimum-label-confidence", "0.66",
            "--no-exclude-points-in-tracked-boxes",
            "--require-no-voxel-downsampling",
            "--npz-compression", args.npz_compression,
            "--overwrite",
        ]
        if selected_pcl_work_root is not None:
            selected_pcl_work_root.mkdir(parents=True, exist_ok=True)
            command.extend(["--work-root", str(selected_pcl_work_root)])
        env = common_env.copy()
        env.update({
            "OMP_NUM_THREADS": str(args.pcl_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        summary["stages"]["reconstruct"] = run_stage("reconstruct", command, env, log_dir)
        report_path = reconstruction_root / "reconstruction_report.json"
        report = load_json_if_exists(report_path)
        if report is not None:
            summary["stages"]["reconstruct"]["detail"] = {
                "elapsed_seconds": report.get("elapsed_seconds"),
                "stage_timings_seconds": report.get("stage_timings_seconds"),
                "pcl_call_timing_sums": report.get("pcl_call_timing_sums"),
            }

    if not args.skip_raycast:
        command = [
            args.python, str(project / "raycaster" / "raycast_mls_scala2.py"),
            "--dataset-root", str(dataset),
            "--caseid", case,
            "--reconstruction-root", str(reconstruction_root),
            "--output-root", str(raycast_root),
            "--sensors", *args.raycast_sensors,
            "--start-frame", str(args.raycast_start_frame),
            "--first-mirror-side", "0",
            "--minimum-range", "0.5",
            "--max-range", str(args.maximum_range),
            "--intersection-mode", "tangent_patch",
            "--patch-radius", str(args.patch_radius),
            "--hit-radius", str(args.patch_radius),
            "--ray-neighbor-count", "0",
            "--candidate-lookup", "structured",
            "--point-batch-size", str(args.point_batch_size),
            "--static-tile-cache", "64",
            "--property-cache", "64",
            "--npz-compression", args.npz_compression,
            "--sensor-workers", str(args.sensor_workers),
            "--ckdtree-workers", "1",
            "--overwrite",
        ]
        if args.raycast_end_frame is not None:
            command.extend(["--end-frame", str(args.raycast_end_frame)])
        env = common_env.copy()
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        summary["stages"]["raycast"] = run_stage("raycast", command, env, log_dir)
        summary["stages"]["raycast"]["detail"] = summarize_raycast(
            raycast_root / "raycast_summary.json"
        )

    if auto_shm and selected_pcl_work_root is not None:
        shutil.rmtree(selected_pcl_work_root, ignore_errors=True)

    summary["pipeline_wall_seconds"] = time.perf_counter() - pipeline_started
    summary_path = log_dir / "cpu_pipeline_benchmark.json"
    with summary_path.open("w") as stream:
        json.dump(summary, stream, indent=2, default=str)

    print("\n" + "=" * 80)
    print("CPU PIPELINE BENCHMARK COMPLETE")
    print("=" * 80)
    for stage, record in summary["stages"].items():
        print(f"{stage:14s}: {record['wall_seconds']:10.2f} s")
    print(f"{'TOTAL':14s}: {summary['pipeline_wall_seconds']:10.2f} s")

    densify_detail = summary["stages"].get("densify", {}).get("detail")
    if isinstance(densify_detail, dict):
        ranked = sorted(
            (
                (name, float(value))
                for name, value in densify_detail.items()
                if name.endswith("_s") and isinstance(value, (int, float)) and name != "total_s"
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            print("\nLargest densification stages:")
            for name, seconds in ranked[:8]:
                print(f"  {name:34s} {seconds:10.2f} s")

    reconstruction_detail = summary["stages"].get("reconstruct", {}).get("detail")
    if isinstance(reconstruction_detail, dict):
        print("\nMLS stage timings:")
        for name, seconds in sorted(
            (reconstruction_detail.get("stage_timings_seconds") or {}).items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            print(f"  {name:34s} {float(seconds):10.2f} s")
        sums = reconstruction_detail.get("pcl_call_timing_sums") or {}
        if sums:
            print("  PCL-call parallel sums:")
            for name, seconds in sorted(sums.items(), key=lambda item: float(item[1]), reverse=True):
                print(f"    {name:30s} {float(seconds):10.2f} s")

    raycast_detail = summary["stages"].get("raycast", {}).get("detail")
    if isinstance(raycast_detail, dict) and raycast_detail.get("completed_sensor_frames", 0):
        print("\nRaycast mean per sensor-frame:")
        print(f"  wall recorded inside renderer       {raycast_detail['mean_sensor_frame_seconds']:10.2f} s")
        for name, seconds in sorted(
            raycast_detail.get("timing_breakdown_mean_s", {}).items(),
            key=lambda item: float(item[1]),
            reverse=True,
        ):
            print(f"  {name:34s} {float(seconds):10.2f} s")

    print(f"Summary       : {summary_path}")
    print(f"Logs          : {log_dir}")


if __name__ == "__main__":
    main()
