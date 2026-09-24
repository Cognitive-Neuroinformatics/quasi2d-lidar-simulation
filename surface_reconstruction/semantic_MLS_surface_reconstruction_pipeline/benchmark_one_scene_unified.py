#!/usr/bin/env python3
"""Benchmark semantic MLS reconstruction + selectable CPU/CUDA SCALA-2 raycasting.

The static preprocessed input is in:
    <dataset-root>/recon_related/<case>/static_recon_labels.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ALL_SENSORS = ["front_left", "front_center", "front_right", "rear_left", "rear_center", "rear_right"]


def run_stage(name: str, command: list[str], env: dict[str, str], log_dir: Path):
    log_path = log_dir / f"{name}.log"
    resource_path = log_dir / f"{name}.resources.txt"
    wrapped = ["/usr/bin/time", "-v", "-o", str(resource_path), *command]
    print("\n" + "=" * 80); print(f"STAGE: {name}"); print("=" * 80); print("COMMAND:"); print(" ".join(command)); print("=" * 80)
    started = time.perf_counter()
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(wrapped, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); log.write(line)
        return_code = process.wait()
    wall = time.perf_counter() - started
    print(f"[{name}] wall time: {wall:.2f} s, exit={return_code}")
    if return_code != 0: raise subprocess.CalledProcessError(return_code, command)
    return {"wall_seconds": wall, "command": command, "log": str(log_path), "resource_report": str(resource_path)}


def load_json(path: Path):
    if not path.is_file(): raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as stream: return json.load(stream)


def load_json_if_exists(path: Path):
    if not path.is_file(): return None
    with path.open() as stream: return json.load(stream)


def require_dict(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict): raise ValueError(f"Config section '{key}' is missing or is not an object")
    return value


def project_path(project: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def summarize_raycast(path: Path):
    data = load_json_if_exists(path)
    if data is None: return None
    completed = []
    for sensor, frames in data.get("sensors", {}).items():
        for record in frames:
            if record.get("status") == "completed": completed.append((sensor, record))
    if not completed: return {"completed_sensor_frames": 0}
    records = [record for _, record in completed]
    seconds = [float(record.get("seconds", 0.0)) for record in records]
    hits = [int(record.get("hits", 0)) for record in records]
    timing_keys = sorted({key for record in records for key in record.get("timing_breakdown_s", {})})
    timing_sum = {key: float(sum(record.get("timing_breakdown_s", {}).get(key, 0.0) for record in records)) for key in timing_keys}
    timing_mean = {key: value / len(records) for key, value in timing_sum.items()}
    return {"completed_sensor_frames": len(records), "sum_sensor_frame_seconds": float(sum(seconds)), "mean_sensor_frame_seconds": float(sum(seconds) / len(seconds)), "min_sensor_frame_seconds": float(min(seconds)), "max_sensor_frame_seconds": float(max(seconds)), "mean_hits": float(sum(hits) / len(hits)), "timing_breakdown_sum_s": timing_sum, "timing_breakdown_mean_s": timing_mean, "static_geometry_io_sum_s": float(sum(record.get("static_geometry_io_seconds", 0.0) for record in records)), "output_write_sum_s": float(sum(record.get("output_write_seconds", 0.0) for record in records)), "points_examined_sum": int(sum(record.get("points_examined", 0) for record in records)), "candidate_pairs_sum": int(sum(record.get("candidate_pairs", 0) for record in records))}


def parse_args(project: Path):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=project / "configs" / "baseline_default.json")
    pre_args, _ = pre.parse_known_args(); config_path = pre_args.config.resolve(); config = load_json(config_path)
    run_cfg = require_dict(config, "run"); ray_cfg = require_dict(config, "raycast"); shared_ray_cfg = require_dict(ray_cfg, "shared"); noise_cfg = require_dict(ray_cfg, "noise"); cuda_cfg = require_dict(ray_cfg, "cuda")
    ap = argparse.ArgumentParser(description=__doc__, parents=[pre])
    ap.add_argument("--dataset-root", type=Path, default=Path(run_cfg["dataset_root"]))
    ap.add_argument("--caseid", default=run_cfg["caseid"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--rasterizer", choices=["cpu", "cuda"], default=run_cfg["rasterizer"], help="Final SCALA-2 raycaster only; preprocessing backend is selected in waymo_preprocessing_cpu_cuda.py.")
    ap.add_argument("--static-input", type=Path, default=None, help="Final preprocessed static NPZ. Default: <dataset>/recon_related/<case>/static_recon_labels.npz")
    ap.add_argument("--clean", action="store_true", help="Delete this pipeline's reconstruction root before the run")
    ap.add_argument("--reconstruction-root", type=Path, default=None)
    ap.add_argument("--skip-reconstruct", action="store_true")
    ap.add_argument("--skip-raycast", action="store_true")
    ap.add_argument("--npz-compression", choices=["stored", "compressed"], default=run_cfg["npz_compression"])
    ap.add_argument("--pcl-threads", type=int, default=int(run_cfg["pcl_threads"]))
    ap.add_argument("--mls-workers", type=int, default=int(run_cfg["mls_workers"]))
    ap.add_argument("--attribute-workers", type=int, default=int(run_cfg["attribute_workers"]))
    ap.add_argument("--pcl-work-root", type=Path, default=None)
    ap.add_argument("--no-auto-shm", action="store_true")
    ap.add_argument("--shm-min-free-gb", type=float, default=float(run_cfg["shm_min_free_gb"]))
    ap.add_argument("--raycast-sensors", nargs="+", choices=ALL_SENSORS, default=list(run_cfg["raycast_sensors"]))
    ap.add_argument("--sensor-workers", type=int, default=int(run_cfg["sensor_workers"]))
    ap.add_argument("--raycast-start-frame", type=int, default=int(run_cfg["raycast_start_frame"]))
    ap.add_argument("--raycast-end-frame", type=int, default=run_cfg.get("raycast_end_frame"))
    ap.add_argument("--point-batch-size", type=int, default=None)
    ap.add_argument("--intersection-mode", default=shared_ray_cfg["intersection_mode"])
    ap.add_argument("--patch-radius", type=float, default=float(shared_ray_cfg["patch_radius_m"]))
    ap.add_argument("--hit-radius", type=float, default=float(shared_ray_cfg["hit_radius_m"]))
    ap.add_argument("--minimum-range", type=float, default=float(shared_ray_cfg["minimum_range_m"]))
    ap.add_argument("--maximum-range", type=float, default=float(shared_ray_cfg["maximum_range_m"]))
    ap.add_argument("--first-mirror-side", type=int, default=int(shared_ray_cfg["first_mirror_side"]))
    ap.add_argument("--cuda-devices", nargs="+", default=[str(v) for v in cuda_cfg["devices"]])
    ap.add_argument("--cuda-precision", choices=["float32", "float64"], default=cuda_cfg["precision"])
    ap.add_argument("--cuda-gpu-cache-gb", type=float, default=float(cuda_cfg["gpu_cache_gb_per_gpu"]))
    ap.add_argument("--cuda-detailed-stats", action="store_true", default=bool(cuda_cfg.get("detailed_stats", False)))
    ap.add_argument("--noise-output", choices=["clean", "noisy", "both"], default=noise_cfg["output"])
    ap.add_argument("--noise-range-sigma-m", type=float, default=float(noise_cfg["range_sigma_m"]))
    ap.add_argument("--noise-azimuth-sigma-deg", type=float, default=float(noise_cfg["azimuth_sigma_deg"]))
    ap.add_argument("--noise-polar-sigma-deg", type=float, default=float(noise_cfg["polar_sigma_deg"]))
    ap.add_argument("--noise-seed", type=int, default=int(noise_cfg["seed"]))
    args = ap.parse_args(); args.config = config_path
    return args, config


def main():
    project = Path(__file__).resolve().parent; args, config = parse_args(project)
    run_cfg = require_dict(config, "run"); mls_cfg = require_dict(config, "mls"); ray_cfg = require_dict(config, "raycast"); shared_ray_cfg = require_dict(ray_cfg, "shared"); cpu_cfg = require_dict(ray_cfg, "cpu"); cuda_cfg = require_dict(ray_cfg, "cuda"); env_cfg = require_dict(config, "environment")
    raycast_point_batch_size = args.point_batch_size if args.point_batch_size is not None else int(cpu_cfg["point_batch_size"] if args.rasterizer == "cpu" else cuda_cfg["point_batch_size"])
    dataset = args.dataset_root.resolve(); case = args.caseid
    static_npz = args.static_input.resolve() if args.static_input is not None else dataset / "recon_related" / case / "static_recon_labels.npz"
    dynamic_root = dataset / "temp" / case / "occ" / "preproc" / "dynamic" / "objects"
    reconstruction_root = args.reconstruction_root.resolve() if args.reconstruction_root is not None else dataset / "semantic_aware_mls" / "semantic_static_mls_cpu_optimized" / case
    raycast_root = reconstruction_root / ("scala2_raycast_cpu_optimized" if args.rasterizer == "cpu" else f"scala2_raycast_cuda_{args.cuda_precision}")
    if not static_npz.is_file(): raise FileNotFoundError(f"Final preprocessed static input not found: {static_npz}\nRun waymo_preprocessing_cpu_cuda.py first.")

    selected_pcl_work_root = args.pcl_work_root.resolve() if args.pcl_work_root is not None else None; auto_shm = False
    if selected_pcl_work_root is None and not args.no_auto_shm:
        shm = Path("/dev/shm")
        if shm.is_dir():
            free_bytes = shutil.disk_usage(shm).free
            if free_bytes >= int(args.shm_min_free_gb * (1024 ** 3)):
                selected_pcl_work_root = shm / f"semantic_static_mls_{os.getpid()}"; selected_pcl_work_root.mkdir(parents=True, exist_ok=True); auto_shm = True
                print(f"Using RAM disk for temporary PCL PCD files: {selected_pcl_work_root} ({free_bytes / (1024 ** 3):.1f} GiB free)")

    run_id = time.strftime("%Y%m%d_%H%M%S"); log_dir = dataset / "semantic_aware_mls" / "pipeline_benchmarks" / case / f"{run_id}_{args.rasterizer}"; log_dir.mkdir(parents=True, exist_ok=True)
    if args.clean and reconstruction_root.exists(): print(f"Removing previous reconstruction: {reconstruction_root}"); shutil.rmtree(reconstruction_root)

    resolved_config = copy.deepcopy(config)
    resolved_config["run"].update({"dataset_root": str(dataset), "caseid": case, "rasterizer": args.rasterizer, "npz_compression": args.npz_compression, "pcl_threads": args.pcl_threads, "mls_workers": args.mls_workers, "attribute_workers": args.attribute_workers, "shm_min_free_gb": args.shm_min_free_gb, "raycast_sensors": list(args.raycast_sensors), "sensor_workers": args.sensor_workers, "raycast_start_frame": args.raycast_start_frame, "raycast_end_frame": args.raycast_end_frame})
    resolved_config["raycast"]["shared"].update({"intersection_mode": args.intersection_mode, "patch_radius_m": args.patch_radius, "hit_radius_m": args.hit_radius, "minimum_range_m": args.minimum_range, "maximum_range_m": args.maximum_range, "first_mirror_side": args.first_mirror_side})
    resolved_config["raycast"]["noise"].update({"output": args.noise_output, "range_sigma_m": args.noise_range_sigma_m, "azimuth_sigma_deg": args.noise_azimuth_sigma_deg, "polar_sigma_deg": args.noise_polar_sigma_deg, "seed": args.noise_seed})
    resolved_config["raycast"][args.rasterizer]["point_batch_size"] = int(raycast_point_batch_size)
    resolved_config["raycast"]["cuda"].update({"precision": args.cuda_precision, "gpu_cache_gb_per_gpu": args.cuda_gpu_cache_gb, "devices": [str(v) for v in args.cuda_devices], "detailed_stats": bool(args.cuda_detailed_stats)})
    resolved_config_path = log_dir / "resolved_pipeline_config.json"; resolved_config_path.write_text(json.dumps(resolved_config, indent=2))

    summary = {"schema_version": 3, "run_id": run_id, "case": case, "dataset_root": str(dataset), "project_root": str(project), "config_file": str(args.config), "resolved_config": str(resolved_config_path), "settings": vars(args) | {"dataset_root": str(dataset), "config": str(args.config), "static_input": str(static_npz), "reconstruction_root": str(reconstruction_root), "pcl_work_root": None if selected_pcl_work_root is None else str(selected_pcl_work_root), "pcl_work_root_auto_shm": bool(auto_shm), "selected_raycast_point_batch_size": int(raycast_point_batch_size)}, "paths": {"static_input": str(static_npz), "dynamic_root": str(dynamic_root), "reconstruction_root": str(reconstruction_root), "raycast_root": str(raycast_root), "log_dir": str(log_dir)}, "stages": {}}

    for command, filename in [(["lscpu"], "lscpu.txt"), (["free", "-h"], "memory.txt"), (["nvidia-smi"], "nvidia_smi.txt")]:
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False); (log_dir / filename).write_text(result.stdout)
        except FileNotFoundError: pass

    common_env = os.environ.copy(); common_env["PYTHONUNBUFFERED"] = "1"
    if args.rasterizer == "cuda" and not args.skip_raycast: subprocess.run([args.python, str(project / "check_cuda_environment.py")], env=common_env, check=True)
    pipeline_started = time.perf_counter()

    if not args.skip_reconstruct:
        command = [args.python, str(project / "reconstruct_semantic_static_mls.py"), "--dataset-root", str(dataset), "--caseid", case, "--static-input", str(static_npz), "--dynamic-input-root", str(dynamic_root), "--pcl-executable", str(project_path(project, mls_cfg["pcl_executable"])), "--config", str(project_path(project, mls_cfg["config_file"])), "--output-root", str(reconstruction_root), "--stages", *[str(v) for v in mls_cfg["stages"]], "--tile-size", str(mls_cfg["tile_size_m"]), "--tile-halo", str(mls_cfg["tile_halo_m"]), "--pcl-threads", str(args.pcl_threads), "--mls-workers", str(args.mls_workers), "--attribute-workers", str(args.attribute_workers), "--minimum-label-confidence", str(mls_cfg["minimum_label_confidence"])]
        if not mls_cfg["exclude_points_in_tracked_boxes"]: command.append("--no-exclude-points-in-tracked-boxes")
        if mls_cfg["require_no_voxel_downsampling"]: command.append("--require-no-voxel-downsampling")
        command.extend(["--npz-compression", args.npz_compression])
        if mls_cfg["overwrite"]: command.append("--overwrite")
        if selected_pcl_work_root is not None: selected_pcl_work_root.mkdir(parents=True, exist_ok=True); command.extend(["--work-root", str(selected_pcl_work_root)])
        env = common_env.copy(); env.update({"OMP_NUM_THREADS": str(args.pcl_threads), "OPENBLAS_NUM_THREADS": str(env_cfg["reconstruct_openblas_threads"]), "MKL_NUM_THREADS": str(env_cfg["reconstruct_mkl_threads"]), "NUMEXPR_NUM_THREADS": str(env_cfg["reconstruct_numexpr_threads"])})
        summary["stages"]["reconstruct"] = run_stage("reconstruct", command, env, log_dir)
        report = load_json_if_exists(reconstruction_root / "reconstruction_report.json")
        if report is not None: summary["stages"]["reconstruct"]["detail"] = {"elapsed_seconds": report.get("elapsed_seconds"), "stage_timings_seconds": report.get("stage_timings_seconds"), "pcl_call_timing_sums": report.get("pcl_call_timing_sums")}

    if not args.skip_raycast:
        common_raycast = ["--dataset-root", str(dataset), "--caseid", case, "--reconstruction-root", str(reconstruction_root), "--output-root", str(raycast_root), "--sensors", *args.raycast_sensors, "--start-frame", str(args.raycast_start_frame), "--first-mirror-side", str(args.first_mirror_side), "--minimum-range", str(args.minimum_range), "--max-range", str(args.maximum_range), "--intersection-mode", args.intersection_mode, "--patch-radius", str(args.patch_radius), "--hit-radius", str(args.hit_radius), "--point-batch-size", str(raycast_point_batch_size), "--static-tile-cache", str(shared_ray_cfg["static_tile_cache"]), "--property-cache", str(shared_ray_cfg["property_cache"]), "--npz-compression", args.npz_compression, "--noise-output", args.noise_output, "--noise-range-sigma-m", str(args.noise_range_sigma_m), "--noise-azimuth-sigma-deg", str(args.noise_azimuth_sigma_deg), "--noise-polar-sigma-deg", str(args.noise_polar_sigma_deg), "--noise-seed", str(args.noise_seed)]
        if args.rasterizer == "cpu": command = [args.python, str(project / "raycaster" / "raycast_mls_scala2.py"), *common_raycast, "--ray-neighbor-count", str(cpu_cfg["ray_neighbor_count"]), "--candidate-lookup", str(cpu_cfg["candidate_lookup"]), "--sensor-workers", str(args.sensor_workers), "--ckdtree-workers", str(cpu_cfg["ckdtree_workers"])]
        else:
            command = [args.python, str(project / "raycaster" / "raycast_mls_scala2_cuda.py"), *common_raycast, "--gpu-cache-gb", str(args.cuda_gpu_cache_gb), "--devices", *args.cuda_devices, "--precision", args.cuda_precision]
            if args.cuda_detailed_stats: command.append("--detailed-stats")
        if shared_ray_cfg["overwrite"]: command.append("--overwrite")
        if args.raycast_end_frame is not None: command.extend(["--end-frame", str(args.raycast_end_frame)])
        env = common_env.copy(); env.update({"OMP_NUM_THREADS": str(env_cfg["raycast_omp_threads"]), "OPENBLAS_NUM_THREADS": str(env_cfg["raycast_openblas_threads"]), "MKL_NUM_THREADS": str(env_cfg["raycast_mkl_threads"]), "NUMEXPR_NUM_THREADS": str(env_cfg["raycast_numexpr_threads"])})
        summary["stages"]["raycast"] = run_stage("raycast", command, env, log_dir); summary["stages"]["raycast"]["detail"] = summarize_raycast(raycast_root / "raycast_summary.json")
        raycast_root.mkdir(parents=True, exist_ok=True); shutil.copy2(resolved_config_path, raycast_root / "resolved_pipeline_config.json")

    if auto_shm and selected_pcl_work_root is not None: shutil.rmtree(selected_pcl_work_root, ignore_errors=True)
    summary["pipeline_wall_seconds"] = time.perf_counter() - pipeline_started; summary_path = log_dir / "unified_pipeline_benchmark.json"; summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 80); print(f"SEMANTIC MLS PIPELINE COMPLETE — {args.rasterizer.upper()} RAYCASTER"); print("=" * 80)
    for stage, record in summary["stages"].items(): print(f"{stage:14s}: {record['wall_seconds']:10.2f} s")
    print(f"{'TOTAL':14s}: {summary['pipeline_wall_seconds']:10.2f} s")
    reconstruction_detail = summary["stages"].get("reconstruct", {}).get("detail")
    if isinstance(reconstruction_detail, dict):
        print("\nMLS stage timings:")
        for name, seconds in sorted((reconstruction_detail.get("stage_timings_seconds") or {}).items(), key=lambda item: float(item[1]), reverse=True): print(f"  {name:34s} {float(seconds):10.2f} s")
    raycast_detail = summary["stages"].get("raycast", {}).get("detail")
    if isinstance(raycast_detail, dict) and raycast_detail.get("completed_sensor_frames", 0):
        print("\nRaycast mean per sensor-frame:"); print(f"  wall recorded inside renderer       {raycast_detail['mean_sensor_frame_seconds']:10.2f} s")
        for name, seconds in sorted(raycast_detail.get("timing_breakdown_mean_s", {}).items(), key=lambda item: float(item[1]), reverse=True): print(f"  {name:34s} {float(seconds):10.2f} s")
    print(f"Static input  : {static_npz}"); print(f"Rasterizer    : {args.rasterizer}"); print(f"Raycast output: {raycast_root}"); print(f"Summary       : {summary_path}"); print(f"Logs          : {log_dir}")


if __name__ == "__main__": main()
