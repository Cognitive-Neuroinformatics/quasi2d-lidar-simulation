#!/usr/bin/env python3
"""CUDA SCALA2 raycaster for the semantic MLS reconstruction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

try:
    import torch
except Exception as exc:
    raise RuntimeError(
        "PyTorch is required for the CUDA raycaster. Install a CUDA-enabled "
        "PyTorch build in this environment."
    ) from exc

# Import the validated CPU renderer's I/O, tile culling and output helpers.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from raycast_mls_scala2 import (  # noqa: E402
    DynamicObjectStore,
    LazyPropertyStore,
    PointSurfaceRaycaster,
    StaticTileStore,
    frame_summary,
    load_active_dynamic_tracks,
    load_frame_mapping,
    save_npz,
)
from scala2_noise import apply_scala2_measurement_noise  # noqa: E402
from scala2_geometry import (  # noqa: E402
    SCALA2_HEIGHT,
    SCALA2_SENSOR_EXTRINSICS,
    SCALA2_WIDTH,
    generate_scala2_ray_geometry,
    get_scala2_to_vehicle,
    mirror_side_for_frame,
    scala2_world_pose,
)
from cuda_scala2_backend import (  # noqa: E402
    CudaGeometryCache,
    TorchStructuredScala2,
    per_ray_nearest_candidate,
)


class CudaPointSurfaceRaycaster(PointSurfaceRaycaster):
    """GPU implementation with CPU-compatible final state/output methods."""

    def __init__(
        self,
        geometry: dict[str, np.ndarray],
        sensor_to_world: np.ndarray,
        minimum_range: float,
        maximum_range: float,
        hit_radius: float,
        intersection_mode: str,
        patch_radius: float,
        point_batch_size: int,
        gpu_cache: CudaGeometryCache,
        precision: str = "float32",
        detailed_stats: bool = False,
    ):
        # The parent gives us validated tile culling, property resolution and
        # exact output schema. CPU candidate search is never called here.
        super().__init__(
            geometry,
            sensor_to_world,
            minimum_range,
            maximum_range,
            hit_radius,
            intersection_mode,
            patch_radius,
            ray_neighbor_count=0,
            point_batch_size=point_batch_size,
            use_fov_prefilter=False,
            ckdtree_workers=1,
            candidate_lookup="structured",
        )

        self.gpu_cache = gpu_cache
        self.device = gpu_cache.device
        self.dtype = torch.float32 if precision == "float32" else torch.float64
        if self.dtype != gpu_cache.dtype:
            raise ValueError("Raycaster precision must match GPU cache dtype")
        self.detailed_stats = bool(detailed_stats)
        self.ray_count = SCALA2_HEIGHT * SCALA2_WIDTH

        self.structured_gpu = TorchStructuredScala2(
            geometry["directions"],
            SCALA2_HEIGHT,
            SCALA2_WIDTH,
            self.device,
            self.dtype,
        )
        self.directions_gpu = self.structured_gpu.flat_directions

        self.origin_gpu = torch.as_tensor(
            self.origin_world, device=self.device, dtype=self.dtype
        )
        self.rotation_sensor_to_world_gpu = torch.as_tensor(
            self.rotation_sensor_to_world, device=self.device, dtype=self.dtype
        )

        inf = float("inf")
        self.g_best_range = torch.full((self.ray_count,), inf, dtype=self.dtype, device=self.device)
        self.g_best_distance = torch.full((self.ray_count,), inf, dtype=self.dtype, device=self.device)
        self.g_surface_xyz_world = torch.full((self.ray_count, 3), float("nan"), dtype=self.dtype, device=self.device)
        self.g_normal_world = torch.full((self.ray_count, 3), float("nan"), dtype=self.dtype, device=self.device)
        self.g_surface_xyz_source = torch.full((self.ray_count, 3), float("nan"), dtype=self.dtype, device=self.device)
        self.g_surface_normal_source = torch.full((self.ray_count, 3), float("nan"), dtype=self.dtype, device=self.device)
        self.g_source_coordinate_frame = torch.full((self.ray_count,), -1, dtype=torch.int8, device=self.device)
        self.g_source_type = torch.full((self.ray_count,), -1, dtype=torch.int8, device=self.device)
        self.g_source_object_id = torch.full((self.ray_count,), -1, dtype=torch.int32, device=self.device)
        self.g_source_key = torch.full((self.ray_count,), -1, dtype=torch.int32, device=self.device)
        self.g_source_row = torch.full((self.ray_count,), -1, dtype=torch.int64, device=self.device)

        self.path_to_key: dict[str, int] = {}
        self.key_to_path: dict[int, str] = {}
        self.next_source_key = 0

        self.g_points_in_range = torch.zeros((), dtype=torch.int64, device=self.device)
        self.g_points_in_fov = torch.zeros((), dtype=torch.int64, device=self.device)
        self.g_candidate_pairs = torch.zeros((), dtype=torch.int64, device=self.device)

        # Parent timing keys are kept for output compatibility. CUDA work is
        # reported separately because per-stage synchronization would distort
        # the accelerated benchmark.
        self.timing = {
            "gpu_surface_compute": 0.0,
            "gpu_download": 0.0,
            "property_resolve": 0.0,
        }
        self._downloaded = False

    def _source_key(self, path: Path) -> int:
        text = str(path)
        if text not in self.path_to_key:
            key = self.next_source_key
            self.next_source_key += 1
            self.path_to_key[text] = key
            self.key_to_path[key] = text
        return self.path_to_key[text]

    def _transform_world(
        self,
        xyz: torch.Tensor,
        normal: torch.Tensor,
        local_to_world: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if local_to_world is None:
            return xyz, normal, 0
        transform = torch.as_tensor(local_to_world, device=self.device, dtype=self.dtype).reshape(4, 4)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        xyz_world = xyz @ rotation.T + translation
        normal_world = normal @ rotation.T
        return xyz_world, normal_world, 1

    def process_surface(
        self,
        surface: dict,
        source_type: int,
        source_path: Path,
        source_object_id: int = -1,
        local_to_world: np.ndarray | None = None,
        pretransformed_world=None,
    ) -> None:
        if len(surface["xyz"]) == 0:
            return

        # CUDA version transforms dynamic geometry itself; the CPU-side
        # pretransformed tuple is intentionally ignored to avoid huge duplicate
        # world arrays and CPU work.
        entry = self.gpu_cache.get(source_path, surface)
        source_xyz = entry.xyz
        source_normal = entry.normal
        source_row_map = entry.source_row
        source_count = int(entry.source_count)
        path_text = str(source_path)
        previous = self.source_count_by_path.get(path_text)
        if previous is not None and previous != source_count:
            raise ValueError(f"Inconsistent source count for {source_path}: {previous} vs {source_count}")
        self.source_count_by_path[path_text] = source_count
        source_key = self._source_key(source_path)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        surface_started = time.perf_counter()

        minimum_sq = self.minimum_range ** 2
        maximum_sq = self.maximum_range ** 2
        support_radius = self.hit_radius if self.intersection_mode == "point_tube" else self.patch_radius

        dynamic_rotation = None
        dynamic_translation = None
        if local_to_world is not None:
            dynamic_transform = torch.as_tensor(
                local_to_world, device=self.device, dtype=self.dtype
            ).reshape(4, 4)
            dynamic_rotation = dynamic_transform[:3, :3]
            dynamic_translation = dynamic_transform[:3, 3]

        for start in range(0, len(source_xyz), self.point_batch_size):
            stop = min(start + self.point_batch_size, len(source_xyz))
            self.points_examined += stop - start

            xyz_src = source_xyz[start:stop]
            normal_src = source_normal[start:stop]
            if dynamic_rotation is None:
                xyz_world = xyz_src
                normal_world = normal_src
                source_coordinate_frame = 0
            else:
                xyz_world = xyz_src @ dynamic_rotation.T + dynamic_translation
                normal_world = normal_src @ dynamic_rotation.T
                source_coordinate_frame = 1

            # CPU reference row-vector convention:
            # points_sensor = (points_world - origin_world) @ R_sensor_to_world
            points_sensor_all = (xyz_world - self.origin_gpu) @ self.rotation_sensor_to_world_gpu
            range_sq = torch.sum(points_sensor_all * points_sensor_all, dim=1)
            eligible = (
                torch.isfinite(range_sq)
                & (range_sq >= minimum_sq)
                & (range_sq <= maximum_sq)
                & (points_sensor_all[:, 0] > 0.0)
            )
            self.g_points_in_range += eligible.sum()
            eligible_rows = torch.nonzero(eligible, as_tuple=False).flatten()
            if eligible_rows.numel() == 0:
                continue

            points_sensor = points_sensor_all[eligible_rows]
            radial_range = torch.sqrt(range_sq[eligible_rows])
            angular_limit = torch.asin(torch.clamp(support_radius / radial_range, min=0.0, max=1.0))
            unit = points_sensor / radial_range[:, None]

            point_rows, ray_indices = self.structured_gpu.candidates(unit, angular_limit)
            if ray_indices.numel() == 0:
                continue

            # Diagnostic definition matches the fused CPU path: number of
            # surface points that touch at least one SCALA2 angular support cap.
            if self.detailed_stats:
                self.g_points_in_fov += torch.unique(point_rows).numel()
            self.g_candidate_pairs += ray_indices.numel()

            candidate_points = points_sensor[point_rows]
            ray_directions = self.directions_gpu[ray_indices]

            if self.intersection_mode == "point_tube":
                axial_range = torch.sum(candidate_points * ray_directions, dim=1)
                perpendicular_sq = torch.clamp(
                    torch.sum(candidate_points * candidate_points, dim=1) - axial_range * axial_range,
                    min=0.0,
                )
                support_distance = torch.sqrt(perpendicular_sq)
                valid = (
                    (axial_range >= self.minimum_range)
                    & (axial_range <= self.maximum_range)
                    & (support_distance <= self.hit_radius)
                )
            else:
                # Map candidate point rows back through eligible_rows to the
                # original batch geometry, then transform normals only for
                # actual candidate pairs.
                candidate_batch_rows = eligible_rows[point_rows]
                candidate_normal_world = normal_world[candidate_batch_rows]
                normal_sensor = candidate_normal_world @ self.rotation_sensor_to_world_gpu
                normal_length = torch.linalg.vector_norm(normal_sensor, dim=1)
                valid_normal = torch.isfinite(normal_length) & (normal_length > 1e-8)
                safe_length = torch.where(valid_normal, normal_length, torch.ones_like(normal_length))
                normal_sensor = normal_sensor / safe_length[:, None]

                denominator = torch.sum(normal_sensor * ray_directions, dim=1)
                numerator = torch.sum(normal_sensor * candidate_points, dim=1)
                valid_denominator = valid_normal & (torch.abs(denominator) > 1e-6)
                axial_range = torch.full_like(denominator, float("inf"))
                axial_range = torch.where(valid_denominator, numerator / denominator, axial_range)
                hit_points = ray_directions * axial_range[:, None]
                support_distance = torch.linalg.vector_norm(hit_points - candidate_points, dim=1)
                valid = (
                    valid_denominator
                    & torch.isfinite(axial_range)
                    & (axial_range >= self.minimum_range)
                    & (axial_range <= self.maximum_range)
                    & (support_distance <= self.patch_radius)
                )

            valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            vray = ray_indices[valid_positions]
            vrange = axial_range[valid_positions]
            vdistance = support_distance[valid_positions]
            vpoint_rows = point_rows[valid_positions]

            selected_in_valid = per_ray_nearest_candidate(
                vray, vrange, vdistance, self.ray_count
            )
            if selected_in_valid.numel() == 0:
                continue

            selected_rays = vray[selected_in_valid]
            selected_range = vrange[selected_in_valid]
            selected_distance = vdistance[selected_in_valid]
            selected_point_rows = vpoint_rows[selected_in_valid]

            current_range = self.g_best_range[selected_rays]
            current_distance = self.g_best_distance[selected_rays]
            nearer = selected_range < current_range
            same_range = torch.isclose(
                selected_range,
                current_range,
                rtol=1e-7,
                atol=1e-6,
                equal_nan=False,
            )
            improves = nearer | (same_range & (selected_distance < current_distance))
            update_positions = torch.nonzero(improves, as_tuple=False).flatten()
            if update_positions.numel() == 0:
                continue

            rays = selected_rays[update_positions]
            ranges = selected_range[update_positions]
            distances = selected_distance[update_positions]
            point_rows_update = selected_point_rows[update_positions]
            batch_rows = eligible_rows[point_rows_update]
            geometry_rows = batch_rows + start

            if source_row_map is None:
                original_rows = geometry_rows
            else:
                original_rows = source_row_map[geometry_rows]

            selected_xyz_source = source_xyz[geometry_rows]
            selected_normal_source = source_normal[geometry_rows]
            if dynamic_rotation is None:
                selected_xyz_world = selected_xyz_source
                selected_normal_world = selected_normal_source
            else:
                selected_xyz_world = selected_xyz_source @ dynamic_rotation.T + dynamic_translation
                selected_normal_world = selected_normal_source @ dynamic_rotation.T

            self.g_best_range[rays] = ranges
            self.g_best_distance[rays] = distances
            self.g_surface_xyz_world[rays] = selected_xyz_world
            self.g_normal_world[rays] = selected_normal_world
            self.g_surface_xyz_source[rays] = selected_xyz_source
            self.g_surface_normal_source[rays] = selected_normal_source
            self.g_source_coordinate_frame[rays] = source_coordinate_frame
            self.g_source_type[rays] = int(source_type)
            self.g_source_object_id[rays] = int(source_object_id)
            self.g_source_key[rays] = int(source_key)
            self.g_source_row[rays] = original_rows

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timing["gpu_surface_compute"] += time.perf_counter() - surface_started

    def download_gpu_state(self) -> None:
        if self._downloaded:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        self.best_range = self.g_best_range.detach().cpu().numpy().astype(np.float64, copy=False)
        self.best_distance = self.g_best_distance.detach().cpu().numpy().astype(np.float64, copy=False)
        self.surface_xyz_world = self.g_surface_xyz_world.detach().cpu().numpy().astype(np.float64, copy=False)
        self.normal_world = self.g_normal_world.detach().cpu().numpy().astype(np.float64, copy=False)
        self.surface_xyz_source = self.g_surface_xyz_source.detach().cpu().numpy().astype(np.float64, copy=False)
        self.surface_normal_source = self.g_surface_normal_source.detach().cpu().numpy().astype(np.float64, copy=False)
        self.surface_source_coordinate_frame = self.g_source_coordinate_frame.detach().cpu().numpy()
        self.source_type = self.g_source_type.detach().cpu().numpy()
        self.source_object_id = self.g_source_object_id.detach().cpu().numpy()
        source_key = self.g_source_key.detach().cpu().numpy()
        self.source_row = self.g_source_row.detach().cpu().numpy()

        self.source_path = np.empty(self.ray_count, dtype=object)
        self.source_path[:] = None
        hit = source_key >= 0
        for key in np.unique(source_key[hit]):
            self.source_path[source_key == key] = self.key_to_path[int(key)]

        self.points_in_range = int(self.g_points_in_range.detach().cpu().item())
        self.candidate_pairs = int(self.g_candidate_pairs.detach().cpu().item())
        if self.detailed_stats:
            self.points_in_fov = int(self.g_points_in_fov.detach().cpu().item())
        else:
            self.points_in_fov = -1

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timing["gpu_download"] += time.perf_counter() - t0
        self._downloaded = True

    def resolve_properties(self, property_store: LazyPropertyStore) -> None:
        self.download_gpu_state()
        # Parent implementation adds its duration to property_resolve.
        super().resolve_properties(property_store)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--caseid", required=True)
    parser.add_argument("--reconstruction-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sensors", nargs="+", choices=sorted(SCALA2_SENSOR_EXTRINSICS), default=["front_center"])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive")
    parser.add_argument("--first-mirror-side", type=int, choices=[0, 1], default=0)
    parser.add_argument("--minimum-range", type=float, default=0.5)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--hit-radius", type=float, default=0.03)
    parser.add_argument("--intersection-mode", choices=["point_tube", "tangent_patch"], default="tangent_patch")
    parser.add_argument("--patch-radius", type=float, default=0.03)
    parser.add_argument(
        "--point-batch-size",
        type=int,
        default=2_000_000,
        help="Source points processed per CUDA batch. Reduce if VRAM is insufficient.",
    )
    parser.add_argument("--static-tile-cache", type=int, default=64)
    parser.add_argument("--property-cache", type=int, default=64)
    parser.add_argument(
        "--gpu-cache-gb",
        type=float,
        default=5.0,
        help="Per-GPU LRU budget for resident xyz+normal tensors.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["0"],
        help="CUDA device indices, e.g. --devices 0 1 2. Sensors are assigned round-robin.",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float64"],
        default="float32",
        help="float32 is the accelerated mode; float64 is the stricter CPU-reference check mode.",
    )
    parser.add_argument(
        "--npz-compression",
        choices=["stored", "compressed"],
        default="stored",
        help="stored is recommended for accelerator benchmarks to avoid CPU compression dominating wall time.",
    )
    parser.add_argument("--no-tile-fov-cull", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--ground-only", action="store_true")
    parser.add_argument("--include-curb", action="store_true")
    parser.add_argument("--detailed-stats", action="store_true", help="Compute extra GPU point-in-FOV diagnostics; slightly slower.")
    parser.add_argument("--noise-output", choices=["clean", "noisy", "both"], default="clean", help="clean=existing points/, noisy=points_noisy/ only, both=write both without reraycasting twice")
    parser.add_argument("--noise-range-sigma-m", type=float, default=0.05)
    parser.add_argument("--noise-azimuth-sigma-deg", type=float, default=0.1)
    parser.add_argument("--noise-polar-sigma-deg", type=float, default=0.6)
    parser.add_argument("--noise-seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.minimum_range < 0 or args.max_range <= args.minimum_range:
        parser.error("Require 0 <= minimum-range < max-range")
    if args.hit_radius <= 0 or args.patch_radius <= 0:
        parser.error("--hit-radius and --patch-radius must be positive")
    if args.point_batch_size < 1:
        parser.error("--point-batch-size must be positive")
    if args.gpu_cache_gb < 0:
        parser.error("--gpu-cache-gb must be non-negative")
    if args.ground_only and not args.static_only:
        parser.error("--ground-only requires --static-only")
    if args.include_curb and not args.ground_only:
        parser.error("--include-curb is meaningful only with --ground-only")
    if min(args.noise_range_sigma_m, args.noise_azimuth_sigma_deg, args.noise_polar_sigma_deg) < 0:
        parser.error("Noise standard deviations must be non-negative")

    args.dataset_root = args.dataset_root.resolve()
    if args.reconstruction_root is None:
        args.reconstruction_root = args.dataset_root / "semantic_aware_mls" / "semantic_static_mls" / args.caseid
    else:
        args.reconstruction_root = args.reconstruction_root.resolve()
    if args.output_root is None:
        args.output_root = args.reconstruction_root / "scala2_raycast_cuda"
    else:
        args.output_root = args.output_root.resolve()
    return args


def normalize_devices(values: list[str]) -> list[torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. This environment does not have a usable CUDA-enabled PyTorch build."
        )
    devices = []
    count = torch.cuda.device_count()
    for text in values:
        if str(text).startswith("cuda:"):
            index = int(str(text).split(":", 1)[1])
        else:
            index = int(text)
        if not (0 <= index < count):
            raise ValueError(f"CUDA device {index} does not exist; torch sees {count} device(s)")
        devices.append(torch.device(f"cuda:{index}"))
    if not devices:
        raise ValueError("At least one CUDA device is required")
    return devices


def main():
    args = parse_args()
    devices = normalize_devices(args.devices)

    # Numerical stability: do not silently use TF32 for geometry matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    dtype = torch.float32 if args.precision == "float32" else torch.float64

    calibration_path = (
        args.dataset_root / "laser_calibrations" / args.caseid
        / "laser_calibrations" / "laser_calibrations.npz"
    )
    with np.load(calibration_path, allow_pickle=False) as calibration:
        frame_poses = np.asarray(calibration["frame_pose"], dtype=np.float64)

    frame_count = len(frame_poses)
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if not (0 <= args.start_frame < end_frame <= frame_count):
        raise ValueError(f"Invalid frame interval [{args.start_frame}, {end_frame}) for {frame_count} frames")

    frame_mapping = load_frame_mapping(args.dataset_root, args.caseid, frame_count)
    dynamic_tracks = [] if args.static_only else load_active_dynamic_tracks(args.dataset_root, args.caseid)
    dynamic_by_frame: dict[int, list[tuple[dict, dict]]] = {}
    for track in dynamic_tracks:
        for frame_text, frame_record in track["frames"].items():
            dynamic_by_frame.setdefault(int(frame_text), []).append((track, frame_record))

    static_semantic_ids = None
    if args.ground_only:
        static_semantic_ids = set(range(18, 23))
        if args.include_curb:
            static_semantic_ids.add(17)

    static_store = StaticTileStore(
        args.reconstruction_root / "static_manifest.json",
        cache_size=args.static_tile_cache,
        semantic_ids=static_semantic_ids,
    )
    dynamic_store = DynamicObjectStore(args.reconstruction_root, cache_size=args.static_tile_cache)
    property_store = LazyPropertyStore(cache_size=args.property_cache)

    gpu_caches = {
        device.index: CudaGeometryCache(device, dtype, max_gb=args.gpu_cache_gb)
        for device in devices
    }
    device_locks = {device.index: threading.Lock() for device in devices}
    missing_models: set[int] = set()
    missing_lock = threading.Lock()

    # Stable round-robin assignment: with 3 GPUs and 6 sensors each GPU gets 2.
    sensor_device = {
        sensor: devices[i % len(devices)]
        for i, sensor in enumerate(args.sensors)
    }

    point_dirs = {}
    noisy_point_dirs = {}
    for sensor in args.sensors:
        point_dirs[sensor] = args.output_root / sensor / "points"
        noisy_point_dirs[sensor] = args.output_root / sensor / "points_noisy"
        if args.noise_output in ("clean", "both"): point_dirs[sensor].mkdir(parents=True, exist_ok=True)
        if args.noise_output in ("noisy", "both"): noisy_point_dirs[sensor].mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCALA2 MLS TANGENT-PATCH RAYCAST - PYTORCH CUDA")
    print("=" * 80)
    print(f"Case             : {args.caseid}")
    print(f"Frames           : {args.start_frame}..{end_frame - 1}")
    print(f"Sensors          : {args.sensors}")
    print(f"Output layout    : clean=points/, noisy=points_noisy/, selected={args.noise_output}")
    print(f"Devices          : {[str(d) for d in devices]}")
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        print(f"  {device}: {props.name}, VRAM={props.total_memory / 1024**3:.1f} GiB")
    print(f"Precision        : {args.precision} (TF32 disabled)")
    print(f"Point batch size : {args.point_batch_size:,}")
    print(f"GPU cache/device : {args.gpu_cache_gb:.2f} GiB")
    print(f"Maximum range    : {args.max_range:.3f} m")
    print(f"Intersection mode: {args.intersection_mode}")
    print(f"Patch radius     : {args.patch_radius:.3f} m")
    print(f"Tile FOV cull    : {not args.no_tile_fov_cull}")
    print(f"NPZ output       : {args.npz_compression}")
    print(f"Static only      : {args.static_only}")
    print(f"Noise output     : {args.noise_output}")
    if args.noise_output != "clean": print(f"Noise sigma      : range={args.noise_range_sigma_m:.3f} m, azimuth={args.noise_azimuth_sigma_deg:.3f} deg, polar={args.noise_polar_sigma_deg:.3f} deg, seed={args.noise_seed}")
    print("Sensor -> device : " + ", ".join(f"{s}={sensor_device[s]}" for s in args.sensors))

    total_rays = SCALA2_HEIGHT * SCALA2_WIDTH
    run_started = time.time()
    run_summary = {
        "backend": "pytorch_cuda_structured_exact",
        "case": args.caseid,
        "parameters": {
            "devices": [str(d) for d in devices],
            "precision": args.precision,
            "point_batch_size": args.point_batch_size,
            "gpu_cache_gb": args.gpu_cache_gb,
            "minimum_range_m": args.minimum_range,
            "maximum_range_m": args.max_range,
            "hit_radius_m": args.hit_radius,
            "patch_radius_m": args.patch_radius,
            "intersection_mode": args.intersection_mode,
            "npz_compression": args.npz_compression,
            "tile_fov_cull": not args.no_tile_fov_cull,
            "noise_output": args.noise_output,
            "noise_model": "independent_gaussian_spherical_post_hit_v1" if args.noise_output != "clean" else None,
            "noise_range_sigma_m": args.noise_range_sigma_m,
            "noise_azimuth_sigma_deg": args.noise_azimuth_sigma_deg,
            "noise_polar_sigma_deg": args.noise_polar_sigma_deg,
            "noise_seed": args.noise_seed,
        },
        "sensor_device": {s: str(sensor_device[s]) for s in args.sensors},
        "sensors": {s: [] for s in args.sensors},
    }

    def render_sensor_frame(sensor_name: str, frame_index: int, geometry: dict, mirror_side: int):
        device = sensor_device[sensor_name]
        lock = device_locks[device.index]
        # Never run two independent renderers concurrently on one GPU. Different
        # GPUs still render in parallel through the frame-level executor.
        with lock, torch.cuda.device(device):
            clean_output_path = point_dirs[sensor_name] / f"{frame_index:03d}.npz"
            noisy_output_path = noisy_point_dirs[sensor_name] / f"{frame_index:03d}.npz"
            required_paths = ([clean_output_path] if args.noise_output == "clean" else [noisy_output_path] if args.noise_output == "noisy" else [clean_output_path, noisy_output_path])
            reuse_path = clean_output_path if args.noise_output in ("clean", "both") else noisy_output_path
            if all(path.exists() for path in required_paths) and not args.overwrite:
                output_path = reuse_path
                with np.load(output_path, allow_pickle=False) as existing:
                    summary = frame_summary(
                        {name: np.asarray(existing[name]) for name in ("xyz", "semantic_id", "source_type")},
                        total_rays,
                    )
                summary.update({"output_frame_index": frame_index, "status": "reused", "device": str(device)})
                return sensor_name, summary, f"{sensor_name} frame {frame_index:03d}: reuse {summary['hits']:,} hits"

            started = time.time()
            cache = gpu_caches[device.index]
            h2d_before_s = cache.h2d_seconds
            h2d_before_b = cache.h2d_bytes
            sensor_to_world, _ = scala2_world_pose(frame_poses[frame_index], sensor_name)
            raycaster = CudaPointSurfaceRaycaster(
                geometry,
                sensor_to_world,
                args.minimum_range,
                args.max_range,
                args.hit_radius,
                args.intersection_mode,
                args.patch_radius,
                args.point_batch_size,
                gpu_cache=cache,
                precision=args.precision,
                detailed_stats=args.detailed_stats,
            )

            support_radius = args.hit_radius if args.intersection_mode == "point_tube" else args.patch_radius
            broad_tiles = static_store.relevant_tiles(raycaster.origin_world, args.max_range + support_radius)
            visible_tiles = []
            tile_cull_s = 0.0
            geometry_io_s = 0.0

            for tile in broad_tiles:
                keep = True
                if not args.no_tile_fov_cull:
                    t0 = time.perf_counter()
                    bounds = static_store.ensure_bounds(tile)
                    if bounds is not None and bounds.get("min_xyz") is not None and bounds.get("max_xyz") is not None:
                        keep = raycaster.bounds_may_hit(
                            np.asarray(bounds["min_xyz"], dtype=np.float64),
                            np.asarray(bounds["max_xyz"], dtype=np.float64),
                            support_radius,
                        )
                    tile_cull_s += time.perf_counter() - t0
                if not keep:
                    continue
                visible_tiles.append(tile)
                t0 = time.perf_counter()
                surface = static_store.load_geometry(tile)
                geometry_io_s += time.perf_counter() - t0
                raycaster.process_surface(
                    surface,
                    source_type=0,
                    source_path=static_store.source_path(tile),
                )

            active_objects = 0
            if not args.static_only:
                for track, frame_record in dynamic_by_frame.get(frame_index, ()):
                    object_id = int(track["lidargs_object_id"])
                    surface = dynamic_store.load_geometry(object_id)
                    if surface is None:
                        with missing_lock:
                            if object_id not in missing_models:
                                print(f"WARNING: dynamic object {object_id} has no MLS model; skip")
                                missing_models.add(object_id)
                        continue
                    local_to_world = np.asarray(frame_record["box_pose_world"], dtype=np.float64).reshape(4, 4)
                    raycaster.process_surface(
                        surface,
                        source_type=1,
                        source_path=dynamic_store.path(object_id),
                        source_object_id=object_id,
                        local_to_world=local_to_world,
                    )
                    active_objects += 1

            raycaster.resolve_properties(property_store)
            arrays = raycaster.output(
                frame_index,
                frame_mapping.get(frame_index, frame_index),
                sensor_name,
                vehicle_to_world=frame_poses[frame_index],
                sensor_to_vehicle=get_scala2_to_vehicle(sensor_name),
            )

            write_started = time.perf_counter()
            noisy_arrays = None
            if args.noise_output in ("noisy", "both"):
                noisy_arrays = apply_scala2_measurement_noise(arrays, args.noise_range_sigma_m, args.noise_azimuth_sigma_deg, args.noise_polar_sigma_deg, args.noise_seed)
            if args.noise_output in ("clean", "both") and (args.overwrite or not clean_output_path.exists()): save_npz(clean_output_path, arrays, args.npz_compression)
            if args.noise_output in ("noisy", "both") and (args.overwrite or not noisy_output_path.exists()): save_npz(noisy_output_path, noisy_arrays, args.npz_compression)
            write_s = time.perf_counter() - write_started
            static_store.flush_bounds()

            summary = frame_summary(noisy_arrays if args.noise_output == "noisy" else arrays, total_rays)
            elapsed = time.time() - started
            h2d_s = cache.h2d_seconds - h2d_before_s
            h2d_b = cache.h2d_bytes - h2d_before_b
            summary.update({
                "output_frame_index": frame_index,
                "source_frame_index": frame_mapping.get(frame_index, frame_index),
                "mirror_side": mirror_side,
                "device": str(device),
                "seconds": elapsed,
                "static_tiles_considered": len(broad_tiles),
                "static_tiles_visible": len(visible_tiles),
                "static_tiles_culled": len(broad_tiles) - len(visible_tiles),
                "static_tile_cull_seconds": tile_cull_s,
                "static_geometry_io_seconds": geometry_io_s,
                "gpu_h2d_seconds": h2d_s,
                "gpu_h2d_bytes": h2d_b,
                "output_write_seconds": write_s,
                "active_dynamic_models": active_objects,
                "points_examined": raycaster.points_examined,
                "points_in_range": raycaster.points_in_range,
                "points_in_fov": raycaster.points_in_fov,
                "candidate_pairs": raycaster.candidate_pairs,
                "timing_breakdown_s": {k: float(v) for k, v in raycaster.timing.items()},
                "gpu_cache_resident_gb": cache.bytes / 1024**3,
                "status": "completed",
            })

            diagnostics = (
                f"{sensor_name} frame {frame_index:03d} MS{mirror_side} {device}: "
                f"{summary['hits']:,}/{total_rays:,} hits ({summary['coverage_percent']:.2f}%), "
                f"dynamic={summary['dynamic_hits']:,}, {elapsed:.2f}s\n"
                f"  points: examined={raycaster.points_examined:,} in_range={raycaster.points_in_range:,} "
                f"candidate_pairs={raycaster.candidate_pairs:,}\n"
                f"  tiles: broad={len(broad_tiles)} visible={len(visible_tiles)} culled={len(broad_tiles)-len(visible_tiles)} "
                f"geometry_io={geometry_io_s:.2f}s h2d={h2d_s:.2f}s ({h2d_b/1024**3:.2f} GiB) "
                f"gpu_cache={cache.bytes/1024**3:.2f} GiB output_write={write_s:.2f}s\n"
                "  timing: " + ", ".join(f"{k}={v:.2f}s" for k, v in raycaster.timing.items())
            )
            return sensor_name, summary, diagnostics

    # At most one concurrent task per GPU; six sensors on three GPUs run in two waves.
    executor = ThreadPoolExecutor(max_workers=len(devices)) if len(devices) > 1 else None
    try:
        for frame_index in range(args.start_frame, end_frame):
            mirror_side = mirror_side_for_frame(frame_index, args.first_mirror_side)
            geometry = generate_scala2_ray_geometry(mirror_side)

            if executor is None:
                for sensor in args.sensors:
                    name, summary, diagnostics = render_sensor_frame(sensor, frame_index, geometry, mirror_side)
                    run_summary["sensors"][name].append(summary)
                    print(diagnostics)
            else:
                # Submit all sensors; per-device locks serialize sensors mapped to
                # the same GPU while independent GPUs execute concurrently.
                futures = {
                    executor.submit(render_sensor_frame, sensor, frame_index, geometry, mirror_side): sensor
                    for sensor in args.sensors
                }
                results = [future.result() for future in as_completed(futures)]
                by_sensor = {name: (summary, diagnostics) for name, summary, diagnostics in results}
                for sensor in args.sensors:
                    summary, diagnostics = by_sensor[sensor]
                    run_summary["sensors"][sensor].append(summary)
                    print(diagnostics)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    run_summary["total_wall_seconds"] = time.time() - run_started
    run_summary["gpu_cache"] = {
        str(torch.device(f"cuda:{idx}")): {
            "resident_gb": cache.bytes / 1024**3,
            "h2d_total_seconds": cache.h2d_seconds,
            "h2d_total_gb": cache.h2d_bytes / 1024**3,
        }
        for idx, cache in gpu_caches.items()
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "raycast_summary.json").open("w") as stream:
        json.dump(run_summary, stream, indent=2)

    print("\n" + "=" * 80)
    print(f"CUDA raycast complete: {args.output_root}")
    print(f"TOTAL WALL TIME: {run_summary['total_wall_seconds']:.2f} s ({run_summary['total_wall_seconds']/60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
