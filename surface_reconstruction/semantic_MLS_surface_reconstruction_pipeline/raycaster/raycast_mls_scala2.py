#!/usr/bin/env python3
"""Raycast alternating SCALA2 scans against PCL-MLS point surfaces.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial import cKDTree

from scala2_noise import apply_scala2_measurement_noise
from scala2_geometry import (
    SCALA2_HEIGHT,
    SCALA2_SENSOR_EXTRINSICS,
    SCALA2_WIDTH,
    generate_scala2_ray_geometry,
    get_scala2_to_vehicle,
    mirror_side_for_frame,
    scala2_world_pose,
)


REQUIRED_SURFACE_ARRAYS = (
    "xyz",
    "normal",
    "intensity",
    "semantic_id",
    "ground_id",
    "instance_id",
)

CORE_SURFACE_ARRAYS = set(REQUIRED_SURFACE_ARRAYS)


def save_npz(path: Path, arrays: dict[str, np.ndarray], compression: str) -> None:
    if compression == "stored":
        np.savez(path, **arrays)
    elif compression == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError(f"Unknown NPZ compression mode: {compression}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.asarray(normals, dtype=np.float64) @ transform[:3, :3].T


def load_surface_geometry_npz(path: Path, semantic_ids: set[int] | None = None) -> dict:
    """Load only raycasting geometry, not labels/provenance payloads."""
    with np.load(path, allow_pickle=False) as data:
        required = {"xyz", "normal"}
        if semantic_ids is not None:
            required.add("semantic_id")
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing geometry arrays: {missing}")

        xyz_full = np.asarray(data["xyz"])
        normal_full = np.asarray(data["normal"])
        source_count = int(len(xyz_full))
        if len(normal_full) != source_count:
            raise ValueError(f"Geometry length mismatch in {path}: xyz={source_count}, normal={len(normal_full)}")

        if semantic_ids is None:
            xyz = xyz_full
            normal = normal_full
            source_row = None
        else:
            semantic = np.asarray(data["semantic_id"])
            if len(semantic) != source_count:
                raise ValueError(f"semantic_id length mismatch in {path}")
            keep = np.isin(semantic, np.asarray(sorted(semantic_ids), dtype=np.int16))
            source_row = np.flatnonzero(keep).astype(np.int64, copy=False)
            xyz = xyz_full[keep]
            normal = normal_full[keep]

    return {"xyz": xyz, "normal": normal, "_source_row": source_row, "_source_count": source_count}


def load_surface_xyz_npz(path: Path) -> tuple[np.ndarray, int]:
    """Load only xyz, used to learn exact per-tile bounds before normals."""
    with np.load(path, allow_pickle=False) as data:
        if "xyz" not in data.files:
            raise KeyError(f"{path} is missing xyz")
        xyz = np.asarray(data["xyz"])
    return xyz, int(len(xyz))


class LazyPropertyStore:
    """Load non-geometry point properties only for final winning source files."""
    def __init__(self, cache_size: int = 64):
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.lock = threading.RLock()

    def load(self, path: Path, source_count: int) -> dict[str, np.ndarray]:
        key = str(path)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            with np.load(path, allow_pickle=False) as data:
                required = {"intensity", "semantic_id", "ground_id", "instance_id"}
                missing = sorted(required - set(data.files))
                if missing:
                    raise KeyError(f"{path} is missing surface properties: {missing}")
                properties = {}
                for name in data.files:
                    if name in {"xyz", "normal"}:
                        continue
                    values = np.asarray(data[name])
                    if values.ndim >= 1 and values.shape[0] == source_count:
                        properties[name] = values
            for name in ("intensity", "semantic_id", "ground_id", "instance_id"):
                if name not in properties:
                    raise KeyError(f"{path}:{name} is not point-aligned with xyz")
            if self.cache_size > 0:
                self.cache[key] = properties
                self.cache.move_to_end(key)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            return properties

def property_fill_value(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.signedinteger):
        return -1
    if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(dtype, np.bool_):
        return 0
    if np.issubdtype(dtype, np.str_):
        return ""
    if np.issubdtype(dtype, np.bytes_):
        return b""
    raise TypeError(f"Unsupported point-property dtype for raycast preservation: {dtype}")


def make_filled_property(ray_count: int, source: np.ndarray) -> np.ndarray:
    shape = (ray_count,) + tuple(source.shape[1:])
    return np.full(shape, property_fill_value(source.dtype), dtype=source.dtype)


def bbox_distance_squared(
    point_xy: np.ndarray,
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
) -> float:
    delta = np.maximum(np.maximum(minimum_xy - point_xy, 0.0), point_xy - maximum_xy)
    return float(delta @ delta)


class StaticTileStore:
    def __init__(self, manifest_path: Path, cache_size: int, semantic_ids: set[int] | None = None):
        self.manifest_path = manifest_path
        with manifest_path.open() as stream:
            self.manifest = json.load(stream)
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"Static manifest is incomplete: {manifest_path}")
        self.tiles = self.manifest["tiles"]
        self.cache_size = max(0, int(cache_size))
        self.semantic_ids = None if semantic_ids is None else set(semantic_ids)
        self.geometry_cache: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.RLock()

        # Persistent exact 3D bounds. First encounter learns them from xyz only;
        # later frames/runs can cull invisible tiles before opening their NPZ.
        self.bounds_path = manifest_path.parent / "static_tile_bounds_raycast.json"
        stat = manifest_path.stat()
        self.manifest_signature = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "tile_count": int(len(self.tiles)),
            "output_points": int(self.manifest.get("output_points", -1)),
        }
        self.bounds = {}
        self.bounds_dirty = False
        if self.bounds_path.is_file():
            try:
                with self.bounds_path.open() as stream:
                    payload = json.load(stream)
                if payload.get("manifest_signature") == self.manifest_signature:
                    self.bounds = dict(payload.get("bounds", {}))
            except Exception:
                self.bounds = {}

    def relevant_tiles(self, origin_world: np.ndarray, maximum_range: float):
        maximum_squared = maximum_range**2
        point_xy = np.asarray(origin_world[:2], dtype=np.float64)
        return [tile for tile in self.tiles if bbox_distance_squared(
            point_xy,
            np.asarray(tile["core_min_xy"], dtype=np.float64),
            np.asarray(tile["core_max_xy"], dtype=np.float64),
        ) <= maximum_squared]

    def _remember(self, relative: str, surface: dict) -> None:
        if self.cache_size <= 0:
            return
        self.geometry_cache[relative] = surface
        self.geometry_cache.move_to_end(relative)
        while len(self.geometry_cache) > self.cache_size:
            self.geometry_cache.popitem(last=False)

    def ensure_bounds(self, tile: dict) -> dict | None:
        relative = tile["file"]
        with self.lock:
            # CPU-optimized reconstruction writes exact 3D bounds directly into
            # the manifest, so new reconstructions require zero first-run tile
            # decompression just to learn visibility bounds.
            if tile.get("min_xyz") is not None and tile.get("max_xyz") is not None:
                return {
                    "count": int(tile.get("point_count", -1)),
                    "min_xyz": tile["min_xyz"],
                    "max_xyz": tile["max_xyz"],
                }
            if relative in self.bounds:
                return self.bounds[relative]

            cached = self.geometry_cache.get(relative)
            if cached is not None and self.semantic_ids is None:
                xyz_full = np.asarray(cached["xyz"])
                source_count = int(cached["_source_count"])
            else:
                xyz_full, source_count = load_surface_xyz_npz(self.manifest_path.parent / relative)
                if self.semantic_ids is None:
                    self._remember(relative, {
                        "xyz": xyz_full,
                        "normal": None,
                        "_source_row": None,
                        "_source_count": source_count,
                    })

            finite = np.isfinite(xyz_full).all(axis=1)
            if source_count == 0 or not np.any(finite):
                record = {"count": source_count, "min_xyz": None, "max_xyz": None}
            else:
                valid = np.asarray(xyz_full[finite], dtype=np.float64)
                record = {
                    "count": source_count,
                    "min_xyz": valid.min(axis=0).tolist(),
                    "max_xyz": valid.max(axis=0).tolist(),
                }
            self.bounds[relative] = record
            self.bounds_dirty = True
            return record

    def flush_bounds(self) -> None:
        with self.lock:
            if not self.bounds_dirty:
                return
            payload = {
                "format_version": 1,
                "manifest_signature": self.manifest_signature,
                "bounds": self.bounds,
            }
            tmp = self.bounds_path.with_suffix(self.bounds_path.suffix + ".tmp")
            with tmp.open("w") as stream:
                json.dump(payload, stream)
            tmp.replace(self.bounds_path)
            self.bounds_dirty = False

    def load_geometry(self, tile: dict) -> dict:
        relative = tile["file"]
        path = self.manifest_path.parent / relative
        with self.lock:
            cached = self.geometry_cache.get(relative)
            if cached is not None and cached.get("normal") is not None:
                self.geometry_cache.move_to_end(relative)
                return cached

            if cached is not None and self.semantic_ids is None:
                xyz = np.asarray(cached["xyz"])
                source_count = int(cached["_source_count"])
                with np.load(path, allow_pickle=False) as data:
                    normal = np.asarray(data["normal"])
                if len(normal) != source_count:
                    raise ValueError(f"Geometry length mismatch in {path}")
                surface = {
                    "xyz": xyz,
                    "normal": normal,
                    "_source_row": None,
                    "_source_count": source_count,
                }
            else:
                surface = load_surface_geometry_npz(path, semantic_ids=self.semantic_ids)

            self._remember(relative, surface)
            return surface

    def source_path(self, tile: dict) -> Path:
        return self.manifest_path.parent / tile["file"]


class DynamicObjectStore:
    def __init__(self, reconstruction_root: Path, cache_size: int = 64):
        self.root = reconstruction_root / "dynamic_objects"
        self.cache_size = max(0, int(cache_size))
        self.cache: OrderedDict[int, dict | None] = OrderedDict()
        self.lock = threading.RLock()

    def path(self, object_id: int) -> Path:
        return self.root / str(object_id) / "mls_surface.npz"

    def load_geometry(self, object_id: int) -> dict | None:
        with self.lock:
            if object_id in self.cache:
                self.cache.move_to_end(object_id)
                return self.cache[object_id]
            path = self.path(object_id)
            surface = load_surface_geometry_npz(path) if path.is_file() else None
            if self.cache_size > 0:
                self.cache[object_id] = surface
                self.cache.move_to_end(object_id)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            return surface


class PointSurfaceRaycaster:
    def __init__(
        self,
        geometry: dict[str, np.ndarray],
        sensor_to_world: np.ndarray,
        minimum_range: float,
        maximum_range: float,
        hit_radius: float,
        intersection_mode: str,
        patch_radius: float,
        ray_neighbor_count: int,
        point_batch_size: int,
        use_fov_prefilter: bool = True,
        ckdtree_workers: int = -1,
        candidate_lookup: str = "structured",
    ):
        self.geometry = geometry
        self.directions_sensor = geometry["directions"].reshape(-1, 3).astype(np.float64)
        norms = np.linalg.norm(self.directions_sensor, axis=1)
        self.directions_sensor /= norms[:, None]
        self.ray_tree = cKDTree(self.directions_sensor)

        # Conservative angular bounds of the actual SCALA2 rays.  These are
        # used only to reject surface samples that cannot possibly be within
        # the finite support radius of any SCALA2 ray.
        ray_azimuth = np.arctan2(self.directions_sensor[:, 1], self.directions_sensor[:, 0])
        ray_elevation = np.arctan2(
            self.directions_sensor[:, 2],
            np.hypot(self.directions_sensor[:, 0], self.directions_sensor[:, 1]),
        )
        self.ray_azimuth_min = float(ray_azimuth.min())
        self.ray_azimuth_max = float(ray_azimuth.max())
        self.ray_elevation_min = float(ray_elevation.min())
        self.ray_elevation_max = float(ray_elevation.max())

        self.sensor_to_world = np.asarray(sensor_to_world, dtype=np.float64).reshape(4, 4)
        self.rotation_sensor_to_world = self.sensor_to_world[:3, :3]
        self.origin_world = self.sensor_to_world[:3, 3]
        self.minimum_range = float(minimum_range)
        self.maximum_range = float(maximum_range)
        self.hit_radius = float(hit_radius)
        self.intersection_mode = intersection_mode
        self.patch_radius = float(patch_radius)
        self.ray_neighbor_count = min(int(ray_neighbor_count), len(self.directions_sensor))
        self.point_batch_size = int(point_batch_size)
        self.use_fov_prefilter = bool(use_fov_prefilter)
        self.ckdtree_workers = int(ckdtree_workers)
        self.candidate_lookup = str(candidate_lookup)

        # SCALA2 is a structured 16 x 653 ray lattice.  For exact angular
        # candidate lookup we can exploit this structure instead of asking a
        # generic cKDTree about tens of millions of surface points.
        structured = np.asarray(geometry["directions"], dtype=np.float64).copy()
        structured /= np.linalg.norm(structured, axis=2, keepdims=True)
        self.structured_directions = structured
        self.row_azimuth = np.arctan2(structured[:, :, 1], structured[:, :, 0])
        self.row_elevation = np.arctan2(
            structured[:, :, 2],
            np.hypot(structured[:, :, 0], structured[:, :, 1]),
        )
        self.row_elevation_min = self.row_elevation.min(axis=1)
        self.row_elevation_max = self.row_elevation.max(axis=1)

        # Horizontal azimuth must be monotonic within every SCALA2 row.
        if not np.all(np.diff(self.row_azimuth, axis=1) > 0):
            raise RuntimeError("SCALA2 row azimuths are not strictly increasing")

        ray_count = len(self.directions_sensor)
        self.best_range = np.full(ray_count, np.inf, dtype=np.float64)
        self.best_distance = np.full(ray_count, np.inf, dtype=np.float64)
        self.surface_xyz_world = np.full((ray_count, 3), np.nan, dtype=np.float64)
        self.normal_world = np.full((ray_count, 3), np.nan, dtype=np.float64)
        self.intensity = np.full(ray_count, np.nan, dtype=np.float32)
        self.semantic_id = np.full(ray_count, -1, dtype=np.int16)
        self.ground_id = np.full(ray_count, -1, dtype=np.int8)
        self.instance_id = np.zeros(ray_count, dtype=np.int32)
        self.source_type = np.full(ray_count, -1, dtype=np.int8)
        self.source_object_id = np.full(ray_count, -1, dtype=np.int32)
        self.source_path = np.empty(ray_count, dtype=object)
        self.source_path[:] = None
        self.source_row = np.full(ray_count, -1, dtype=np.int64)
        self.source_count_by_path: dict[str, int] = {}

        # Exact MLS source sample selected to support each ray hit.
        # For static surfaces source_xyz/source_normal are in world coordinates.
        # For dynamic surfaces they are in the object's local coordinates.
        self.surface_xyz_source = np.full((ray_count, 3), np.nan, dtype=np.float64)
        self.surface_normal_source = np.full((ray_count, 3), np.nan, dtype=np.float64)
        self.surface_source_coordinate_frame = np.full(ray_count, -1, dtype=np.int8)

        # Arbitrary additional point-aligned properties copied from the winning
        # MLS source sample.  Standard properties remain available at top level.
        self.extra_properties: dict[str, np.ndarray] = {}

        self.points_examined = 0
        self.points_in_range = 0
        self.points_in_fov = 0
        self.candidate_pairs = 0
        self.timing = {
            "transform_range": 0.0,
            "fov_prefilter": 0.0,
            "candidate_lookup": 0.0,
            "intersection": 0.0,
            "nearest_reduction": 0.0,
            "property_resolve": 0.0,
        }

    def bounds_may_hit(self, minimum_xyz_world: np.ndarray, maximum_xyz_world: np.ndarray, support_radius: float) -> bool:
        """Conservative exact tile cull using an enclosing 3D bounding sphere."""
        minimum_xyz_world = np.asarray(minimum_xyz_world, dtype=np.float64)
        maximum_xyz_world = np.asarray(maximum_xyz_world, dtype=np.float64)
        center_world = 0.5 * (minimum_xyz_world + maximum_xyz_world)
        radius = 0.5 * float(np.linalg.norm(maximum_xyz_world - minimum_xyz_world))
        center_sensor = (center_world - self.origin_world) @ self.rotation_sensor_to_world
        distance = float(np.linalg.norm(center_sensor))
        if not np.isfinite(distance) or not np.isfinite(radius):
            return True
        if distance - radius > self.maximum_range:
            return False
        if distance + radius < self.minimum_range:
            return False
        if distance <= radius + 1e-12:
            return True

        center_unit = center_sensor / distance
        sphere_angle = float(np.arcsin(np.clip(radius / distance, 0.0, 1.0)))
        nearest_possible_range = max(distance - radius, self.minimum_range, 1e-12)
        patch_angle = float(np.arcsin(np.clip(support_radius / nearest_possible_range, 0.0, 1.0)))
        chord, _ = self.ray_tree.query(center_unit, k=1, workers=1)
        nearest_ray_angle = float(2.0 * np.arcsin(np.clip(0.5 * float(chord), 0.0, 1.0)))
        return nearest_ray_angle <= sphere_angle + patch_angle + 1e-12

    def _structured_exact_candidates(
        self,
        unit: np.ndarray,
        angular_limit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exactly the same angular candidate pairs as radius lookup.

        For a surface-point direction u and angular support alpha, the cKDTree
        baseline accepts every SCALA2 ray d for which angle(u,d) <= alpha.

        SCALA2 is not an arbitrary set of directions: it is 16 monotonic rows
        with 653 columns each.  We therefore:
          1. reject rows whose elevation range cannot overlap the spherical cap;
          2. use the exact longitude span of that spherical cap to binary-search
             only possible columns in that row;
          3. apply the exact dot-product angular test to the small candidate set.

        This changes only candidate-search implementation, not the hit criterion.
        """
        count = len(unit)
        if count == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        point_azimuth = np.arctan2(unit[:, 1], unit[:, 0])
        point_elevation = np.arctan2(
            unit[:, 2],
            np.hypot(unit[:, 0], unit[:, 1]),
        )

        # Exact spherical-cap longitude span around each point direction.
        # SCALA2 elevations and the maximum patch-induced angular radius are far
        # from the poles, but clipping keeps the expression numerically safe.
        ratio = np.sin(angular_limit) / np.maximum(
            np.cos(point_elevation), 1e-12
        )
        longitude_margin = np.arcsin(np.clip(ratio, 0.0, 1.0))
        cos_limit = np.cos(angular_limit)

        point_chunks = []
        ray_chunks = []
        width = SCALA2_WIDTH

        for row in range(SCALA2_HEIGHT):
            # Necessary condition: spherical angular distance is never smaller
            # than the absolute elevation difference.
            possible = (
                (point_elevation >= self.row_elevation_min[row] - angular_limit)
                & (point_elevation <= self.row_elevation_max[row] + angular_limit)
            )
            point_ids = np.flatnonzero(possible)
            if len(point_ids) == 0:
                continue

            az = point_azimuth[point_ids]
            margin = longitude_margin[point_ids]
            row_az = self.row_azimuth[row]

            lo = np.searchsorted(row_az, az - margin, side="left")
            hi = np.searchsorted(row_az, az + margin, side="right")
            counts = hi - lo

            have_columns = counts > 0
            if not np.any(have_columns):
                continue

            point_ids = point_ids[have_columns]
            lo = lo[have_columns].astype(np.int64, copy=False)
            counts = counts[have_columns].astype(np.int64, copy=False)

            total = int(counts.sum())
            if total == 0:
                continue

            # Expand the short contiguous column intervals without Python loops.
            starts = np.cumsum(counts, dtype=np.int64) - counts
            pair_points = np.repeat(point_ids, counts)
            pair_lo = np.repeat(lo, counts)
            pair_starts = np.repeat(starts, counts)
            columns = pair_lo + (
                np.arange(total, dtype=np.int64) - pair_starts
            )
            pair_rays = row * width + columns

            # Exact angular test.  This is equivalent to the cKDTree chord-radius
            # criterion used by the original code.
            dot = np.einsum(
                "ij,ij->i",
                unit[pair_points],
                self.directions_sensor[pair_rays],
            )
            exact = dot >= (cos_limit[pair_points] - 1e-12)

            if np.any(exact):
                point_chunks.append(pair_points[exact])
                ray_chunks.append(pair_rays[exact])

        if not point_chunks:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        return (
            np.concatenate(point_chunks),
            np.concatenate(ray_chunks),
        )

    def process_surface(
        self,
        surface: dict,
        source_type: int,
        source_path: Path,
        source_object_id: int = -1,
        local_to_world: np.ndarray | None = None,
        pretransformed_world: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if len(surface["xyz"]) == 0:
            return

        source_xyz = np.asarray(surface["xyz"])
        source_normal = np.asarray(surface["normal"])
        source_row_map = surface.get("_source_row")
        source_count = int(surface["_source_count"])
        source_path_string = str(source_path)
        previous_count = self.source_count_by_path.get(source_path_string)
        if previous_count is not None and previous_count != source_count:
            raise ValueError(f"Inconsistent source count for {source_path}: {previous_count} vs {source_count}")
        self.source_count_by_path[source_path_string] = source_count

        if pretransformed_world is not None:
            # Already transformed once per frame and shared by all sensors.
            xyz_world = np.asarray(pretransformed_world[0])
            normal_world = np.asarray(pretransformed_world[1])
            source_coordinate_frame = 1 if source_type == 1 else 0
        elif local_to_world is None:
            # Static MLS is already in world coordinates. Keep the original
            # float32 arrays here instead of duplicating an entire tile as
            # float64. Batch arithmetic below promotes to float64 automatically
            # because sensor pose/origin are float64, so numerical results are
            # unchanged.
            xyz_world = source_xyz
            normal_world = source_normal
            source_coordinate_frame = 0  # world
        else:
            xyz_world = transform_points(source_xyz, local_to_world)
            normal_world = transform_normals(source_normal, local_to_world)
            source_coordinate_frame = 1  # dynamic object-local

        for start in range(0, len(xyz_world), self.point_batch_size):
            stop = min(start + self.point_batch_size, len(xyz_world))
            self.points_examined += stop - start

            t_stage = time.perf_counter()
            batch_points_world = xyz_world[start:stop]
            batch_normals_world = normal_world[start:stop]
            points_sensor = (
                batch_points_world - self.origin_world
            ) @ self.rotation_sensor_to_world

            # Squared-range rejection avoids sqrt for points that can never hit.
            range_squared = np.einsum("ij,ij->i", points_sensor, points_sensor)
            eligible = (
                np.isfinite(range_squared)
                & (range_squared >= self.minimum_range ** 2)
                & (range_squared <= self.maximum_range ** 2)
                & (points_sensor[:, 0] > 0.0)
            )
            if not np.any(eligible):
                self.timing["transform_range"] += time.perf_counter() - t_stage
                continue

            self.points_in_range += int(np.count_nonzero(eligible))
            # local_indices always indexes the original batch. We deliberately
            # do not copy world XYZ/normals for all surviving points; only final
            # candidates/winners gather from the original batch arrays.
            local_indices = np.flatnonzero(eligible)
            points_sensor = points_sensor[eligible]
            radial_range = np.sqrt(range_squared[eligible])
            self.timing["transform_range"] += time.perf_counter() - t_stage

            support_radius = (
                self.hit_radius
                if self.intersection_mode == "point_tube"
                else self.patch_radius
            )

            # Exact conservative SCALA2 FOV prefilter.
            # Any valid finite-patch hit requires the MLS sample centre P to
            # lie within support_radius of the ray.  Therefore the angular
            # separation cannot exceed asin(support_radius / ||P||).
            t_stage = time.perf_counter()
            angular_limit = np.arcsin(
                np.minimum(support_radius / radial_range, 1.0)
            )

            if self.use_fov_prefilter:
                elevation = np.arctan2(
                    points_sensor[:, 2],
                    np.hypot(points_sensor[:, 0], points_sensor[:, 1]),
                )
                keep = (
                    (elevation >= self.ray_elevation_min - angular_limit)
                    & (elevation <= self.ray_elevation_max + angular_limit)
                )
                if not np.any(keep):
                    self.timing["fov_prefilter"] += time.perf_counter() - t_stage
                    continue

                local_indices = local_indices[keep]
                points_sensor = points_sensor[keep]
                radial_range = radial_range[keep]
                angular_limit = angular_limit[keep]
                elevation = elevation[keep]

                azimuth = np.arctan2(points_sensor[:, 1], points_sensor[:, 0])
                endpoint_elevation_bound = np.maximum(
                    np.maximum(
                        np.abs(self.ray_elevation_min),
                        np.abs(self.ray_elevation_max),
                    ),
                    np.abs(elevation),
                )
                # A geodesic of length angular_limit cannot move in elevation
                # by more than angular_limit.  Expanding the latitude bound by
                # that amount keeps the azimuth rejection strictly conservative.
                elevation_bound = np.minimum(
                    endpoint_elevation_bound + angular_limit,
                    np.deg2rad(89.0),
                )
                azimuth_margin = angular_limit / np.maximum(
                    np.cos(elevation_bound), 1e-3
                )
                keep = (
                    (azimuth >= self.ray_azimuth_min - azimuth_margin)
                    & (azimuth <= self.ray_azimuth_max + azimuth_margin)
                )
                if not np.any(keep):
                    self.timing["fov_prefilter"] += time.perf_counter() - t_stage
                    continue

                local_indices = local_indices[keep]
                points_sensor = points_sensor[keep]
                radial_range = radial_range[keep]
                angular_limit = angular_limit[keep]

            # When the separate FOV pass is active, this is the number of
            # samples surviving it. In the default structured-fused path we
            # update the same diagnostic after exact candidate generation so
            # it still means "points that can touch at least one SCALA2 ray".
            fused_structured_fov = (
                not self.use_fov_prefilter
                and self.ray_neighbor_count == 0
                and self.candidate_lookup == "structured"
            )
            if not fused_structured_fov:
                self.points_in_fov += len(points_sensor)
            self.timing["fov_prefilter"] += time.perf_counter() - t_stage

            unit = points_sensor / radial_range[:, None]

            t_stage = time.perf_counter()
            if self.ray_neighbor_count == 0:
                if self.candidate_lookup == "structured":
                    point_rows, ray_indices = self._structured_exact_candidates(
                        unit, angular_limit
                    )
                    if len(ray_indices) == 0:
                        self.timing["candidate_lookup"] += time.perf_counter() - t_stage
                        continue
                else:
                    chord_radius = 2.0 * np.sin(angular_limit / 2.0)
                    neighbor_lists = self.ray_tree.query_ball_point(
                        unit, r=chord_radius, workers=self.ckdtree_workers
                    )
                    neighbor_counts = np.fromiter(
                        (len(values) for values in neighbor_lists),
                        dtype=np.int64,
                        count=len(neighbor_lists),
                    )
                    total_neighbors = int(neighbor_counts.sum())
                    if total_neighbors == 0:
                        self.timing["candidate_lookup"] += time.perf_counter() - t_stage
                        continue
                    point_rows = np.repeat(
                        np.arange(len(points_sensor), dtype=np.int64),
                        neighbor_counts,
                    )
                    ray_indices = np.concatenate(neighbor_lists).astype(
                        np.int64, copy=False
                    )
            else:
                # Positive ray-neighbor-count is intentionally the old k-nearest
                # approximation and therefore still uses cKDTree.
                _, neighbors = self.ray_tree.query(
                    unit,
                    k=self.ray_neighbor_count,
                    workers=self.ckdtree_workers,
                )
                neighbors = np.asarray(neighbors, dtype=np.int64)
                if self.ray_neighbor_count == 1:
                    neighbors = neighbors[:, None]
                point_rows = np.repeat(
                    np.arange(len(points_sensor), dtype=np.int64),
                    self.ray_neighbor_count,
                )
                ray_indices = neighbors.reshape(-1)

            self.candidate_pairs += len(ray_indices)
            if fused_structured_fov:
                has_candidate = np.zeros(len(points_sensor), dtype=bool)
                has_candidate[point_rows] = True
                self.points_in_fov += int(np.count_nonzero(has_candidate))
            self.timing["candidate_lookup"] += time.perf_counter() - t_stage

            candidate_points = points_sensor[point_rows]
            ray_directions = self.directions_sensor[ray_indices]
            if self.intersection_mode == "point_tube":
                t_stage = time.perf_counter()
                axial_range = np.einsum(
                    "ij,ij->i", candidate_points, ray_directions
                )
                perpendicular_squared = np.maximum(
                    np.einsum("ij,ij->i", candidate_points, candidate_points)
                    - axial_range**2,
                    0.0,
                )
                support_distance = np.sqrt(perpendicular_squared)
                valid = (
                    (axial_range >= self.minimum_range)
                    & (axial_range <= self.maximum_range)
                    & (support_distance <= self.hit_radius)
                )
                self.timing["intersection"] += time.perf_counter() - t_stage
            else:
                t_stage = time.perf_counter()

                # Candidate pairs are now far fewer than all in-range/FOV
                # surface samples. Transform normals only for actual point-ray
                # candidates instead of every surviving MLS point. This is exact
                # and is especially important when the separate FOV pass is fused
                # into the structured candidate lookup.
                candidate_batch_rows = local_indices[point_rows]
                candidate_normals = batch_normals_world[candidate_batch_rows]
                normal_sensor = candidate_normals @ self.rotation_sensor_to_world
                normal_length = np.linalg.norm(normal_sensor, axis=1)
                valid_normal = np.isfinite(normal_length) & (normal_length > 1e-8)
                normal_sensor[valid_normal] /= normal_length[valid_normal, None]
                denominator = np.einsum(
                    "ij,ij->i", normal_sensor, ray_directions
                )
                numerator = np.einsum(
                    "ij,ij->i", normal_sensor, candidate_points
                )
                valid_denominator = valid_normal & (np.abs(denominator) > 1e-6)
                axial_range = np.full(len(ray_indices), np.inf, dtype=np.float64)
                axial_range[valid_denominator] = (
                    numerator[valid_denominator] / denominator[valid_denominator]
                )
                hit_points = ray_directions * axial_range[:, None]
                support_distance = np.linalg.norm(
                    hit_points - candidate_points,
                    axis=1,
                )
                valid = (
                    valid_denominator
                    & np.isfinite(axial_range)
                    & (axial_range >= self.minimum_range)
                    & (axial_range <= self.maximum_range)
                    & (support_distance <= self.patch_radius)
                )
                self.timing["intersection"] += time.perf_counter() - t_stage
            if not np.any(valid):
                continue

            ray_indices = ray_indices[valid]
            axial_range = axial_range[valid]
            distance = support_distance[valid]
            point_rows = point_rows[valid]

            # O(N) reduction into the fixed set of 10,448 rays.  This is
            # mathematically equivalent to sorting by (ray, range, distance)
            # but avoids O(N log N) lexsort on every batch.
            t_stage = time.perf_counter()
            ray_count = len(self.directions_sensor)

            batch_min_range = np.full(ray_count, np.inf, dtype=np.float64)
            np.minimum.at(batch_min_range, ray_indices, axial_range)

            min_positions = np.flatnonzero(
                axial_range == batch_min_range[ray_indices]
            )
            min_rays = ray_indices[min_positions]
            min_distances = distance[min_positions]

            batch_min_distance = np.full(ray_count, np.inf, dtype=np.float64)
            np.minimum.at(batch_min_distance, min_rays, min_distances)

            best_positions = min_positions[
                min_distances == batch_min_distance[min_rays]
            ]
            best_rays = ray_indices[best_positions]
            _, first = np.unique(best_rays, return_index=True)
            selected = best_positions[first]

            selected_rays = ray_indices[selected]
            selected_range = axial_range[selected]
            selected_distance = distance[selected]

            current_range = self.best_range[selected_rays]
            current_distance = self.best_distance[selected_rays]

            nearer = selected_range < current_range
            same_range = np.isclose(
                selected_range, current_range,
                rtol=1e-7, atol=1e-6,
                equal_nan=False,
            )
            better_tie = same_range & (selected_distance < current_distance)
            improves = nearer | better_tie

            if not np.any(improves):
                continue

            selected = selected[improves]
            selected_rays = ray_indices[selected]
            rows = point_rows[selected]
            batch_rows = local_indices[rows]
            geometry_rows = batch_rows + start
            if source_row_map is None:
                original_rows = geometry_rows
            else:
                original_rows = np.asarray(source_row_map)[geometry_rows]

            self.best_range[selected_rays] = axial_range[selected]
            self.best_distance[selected_rays] = distance[selected]
            self.surface_xyz_world[selected_rays] = batch_points_world[batch_rows]
            self.normal_world[selected_rays] = batch_normals_world[batch_rows]

            # Preserve the exact source MLS sample as well as its transformed
            # world-space geometry.
            self.surface_xyz_source[selected_rays] = source_xyz[geometry_rows]
            self.surface_normal_source[selected_rays] = source_normal[geometry_rows]
            self.surface_source_coordinate_frame[selected_rays] = source_coordinate_frame

            self.source_type[selected_rays] = source_type
            self.source_object_id[selected_rays] = source_object_id
            self.source_path[selected_rays] = source_path_string
            self.source_row[selected_rays] = original_rows
            self.timing["nearest_reduction"] += time.perf_counter() - t_stage

    def resolve_properties(self, property_store: LazyPropertyStore) -> None:
        """Resolve labels/intensity/provenance only for final winning samples."""
        t0 = time.perf_counter()
        hit_rays = np.flatnonzero(np.isfinite(self.best_range))
        if len(hit_rays) == 0:
            self.timing["property_resolve"] += time.perf_counter() - t0
            return

        paths = self.source_path[hit_rays]
        unique_paths = sorted({str(value) for value in paths if value is not None})
        for path_string in unique_paths:
            mask = np.fromiter((value == path_string for value in paths), dtype=bool, count=len(paths))
            rays = hit_rays[mask]
            if len(rays) == 0:
                continue
            source_count = self.source_count_by_path[path_string]
            properties = property_store.load(Path(path_string), source_count)
            rows = self.source_row[rays]

            self.intensity[rays] = properties["intensity"][rows]
            self.semantic_id[rays] = properties["semantic_id"][rows]
            self.ground_id[rays] = properties["ground_id"][rows]
            self.instance_id[rays] = properties["instance_id"][rows]

            for name, values in properties.items():
                if name in CORE_SURFACE_ARRAYS:
                    continue
                values = np.asarray(values)
                if values.ndim < 1 or len(values) != source_count:
                    continue
                if name not in self.extra_properties:
                    self.extra_properties[name] = make_filled_property(len(self.best_range), values)
                else:
                    destination = self.extra_properties[name]
                    if destination.shape[1:] != values.shape[1:] or destination.dtype != values.dtype:
                        raise ValueError(
                            f"Incompatible point property {name!r}: existing shape/dtype "
                            f"{destination.shape[1:]}/{destination.dtype}, new {values.shape[1:]}/{values.dtype}"
                        )
                self.extra_properties[name][rays] = values[rows]

        self.timing["property_resolve"] += time.perf_counter() - t0

    def output(
        self,
        frame_index: int,
        source_frame_index: int,
        sensor_name: str,
        vehicle_to_world: np.ndarray,
        sensor_to_vehicle: np.ndarray,
    ):
        hit = np.isfinite(self.best_range)
        ray_index = np.flatnonzero(hit)
        ranges = self.best_range[hit]
        directions = self.directions_sensor[hit]
        xyz_sensor = directions * ranges[:, None]
        xyz_world = xyz_sensor @ self.rotation_sensor_to_world.T + self.origin_world
        normal_world = self.normal_world[hit]
        normal_sensor = normal_world @ self.rotation_sensor_to_world

        geometry_flat = {
            name: values.reshape(-1)[hit]
            for name, values in self.geometry.items()
            if name != "directions"
        }

        # Full 16x653 ray geometry is also stored, not only hit-selected values.
        full_ray_geometry = {
            "all_ray_directions_sensor": np.asarray(self.geometry["directions"], dtype=np.float32),
            "all_horizontal_angle_deg": np.asarray(self.geometry["horizontal_angle_deg"], dtype=np.float32),
            "all_vertical_angle_deg": np.asarray(self.geometry["vertical_angle_deg"], dtype=np.float32),
            "all_ray_row": np.asarray(self.geometry["ray_row"]),
            "all_ray_column": np.asarray(self.geometry["ray_column"]),
            "all_apd_group": np.asarray(self.geometry["apd_group"]),
            "all_layer": np.asarray(self.geometry["layer"]),
            "all_mirror_side": np.asarray(self.geometry["mirror_side"]),
        }

        vehicle_to_world = np.asarray(vehicle_to_world, dtype=np.float64).reshape(4, 4)
        sensor_to_vehicle = np.asarray(sensor_to_vehicle, dtype=np.float64).reshape(4, 4)
        world_to_vehicle = np.linalg.inv(vehicle_to_world)
        world_to_sensor = np.linalg.inv(self.sensor_to_world)
        vehicle_to_sensor = np.linalg.inv(sensor_to_vehicle)

        extra_hit_properties = {
            f"surface_property_{name}": values[hit]
            for name, values in sorted(self.extra_properties.items())
        }
        hit_mask = hit.reshape(SCALA2_HEIGHT, SCALA2_WIDTH)
        range_image = self.best_range.copy()
        range_image[~hit] = np.nan
        semantic_image = self.semantic_id.copy()
        ground_image = self.ground_id.copy()
        instance_image = self.instance_id.copy()
        source_image = self.source_type.copy()

        arrays = {
            "xyz": xyz_sensor.astype(np.float32),
            "xyz_sensor": xyz_sensor.astype(np.float32),
            "xyz_world": xyz_world.astype(np.float32),
            "surface_xyz_world": self.surface_xyz_world[hit].astype(np.float32),
            "surface_xyz_source": self.surface_xyz_source[hit].astype(np.float32),
            "surface_normal_source": self.surface_normal_source[hit].astype(np.float32),
            "surface_source_coordinate_frame": self.surface_source_coordinate_frame[hit],
            "range": ranges.astype(np.float32),
            "range_m": ranges.astype(np.float32),
            "intensity": self.intensity[hit],
            "semantic_id": self.semantic_id[hit],
            "ground_id": self.ground_id[hit],
            "instance_id": self.instance_id[hit],
            "normal_world": normal_world.astype(np.float32),
            "normal_sensor": normal_sensor.astype(np.float32),
            "surface_distance_to_ray_m": self.best_distance[hit].astype(np.float32),
            "source_type": self.source_type[hit],
            "source_object_id": self.source_object_id[hit],
            "ray_index": ray_index.astype(np.int32),
            "ray_direction_sensor": directions.astype(np.float32),
            **geometry_flat,
            **extra_hit_properties,
            **full_ray_geometry,
            "surface_property_names": np.asarray(
                sorted(self.extra_properties.keys()), dtype="U128"
            ),
            "surface_source_coordinate_frame_legend": np.asarray(
                ["0=static/world", "1=dynamic/object_local"]
            ),
            "hit_mask": hit_mask,
            "range_image": range_image.reshape(SCALA2_HEIGHT, SCALA2_WIDTH).astype(np.float32),
            "semantic_image": semantic_image.reshape(SCALA2_HEIGHT, SCALA2_WIDTH),
            "ground_image": ground_image.reshape(SCALA2_HEIGHT, SCALA2_WIDTH),
            "instance_image": instance_image.reshape(SCALA2_HEIGHT, SCALA2_WIDTH),
            "source_type_image": source_image.reshape(SCALA2_HEIGHT, SCALA2_WIDTH),
            "vehicle_to_world": vehicle_to_world,
            "world_to_vehicle": world_to_vehicle,
            "sensor_to_vehicle": sensor_to_vehicle,
            "vehicle_to_sensor": vehicle_to_sensor,
            "sensor_to_world": self.sensor_to_world,
            "world_to_sensor": world_to_sensor,
            "vehicle_translation_world": vehicle_to_world[:3, 3],
            "sensor_translation_vehicle": sensor_to_vehicle[:3, 3],
            "sensor_translation_world": self.origin_world,
            "ray_origin_world": self.origin_world,
            "output_frame_index": np.array([frame_index], dtype=np.int32),
            "source_frame_index": np.array([source_frame_index], dtype=np.int32),
            "sensor_name": np.array([sensor_name]),
            "frame_mirror_side": np.array(
                [int(self.geometry["mirror_side"][0, 0])], dtype=np.int8
            ),
            "scala2_height": np.array([SCALA2_HEIGHT], dtype=np.int16),
            "scala2_width": np.array([SCALA2_WIDTH], dtype=np.int16),
            "minimum_range_m": np.array([self.minimum_range], dtype=np.float32),
            "maximum_range_m": np.array([self.maximum_range], dtype=np.float32),
            "hit_radius_m": np.array([self.hit_radius], dtype=np.float32),
            "intersection_mode": np.array([self.intersection_mode]),
            "patch_radius_m": np.array([self.patch_radius], dtype=np.float32),
            "ray_neighbor_count": np.array([self.ray_neighbor_count], dtype=np.int32),
        }
        return arrays


def load_frame_mapping(dataset_root: Path, caseid: str, frame_count: int):
    path = dataset_root / "temp" / caseid / "label_propagation_config.json"
    if not path.is_file():
        return {index: index for index in range(frame_count)}
    with path.open() as stream:
        config = json.load(stream)
    mapping = {
        int(output_index): int(source_index)
        for output_index, source_index in config["output_to_source_frame"].items()
    }
    return mapping


def load_active_dynamic_tracks(dataset_root: Path, caseid: str):
    path = dataset_root / "temp" / caseid / "stage_a_tracks.json"
    with path.open() as stream:
        data = json.load(stream)
    return [
        track
        for track in data["tracks"].values()
        if bool(track["is_dynamic"]) and int(track["lidargs_object_id"]) > 0
    ]


def frame_summary(arrays: dict[str, np.ndarray], total_rays: int):
    semantics = Counter(int(value) for value in arrays["semantic_id"])
    return {
        "hits": int(len(arrays["xyz"])),
        "total_rays": int(total_rays),
        "coverage_percent": float(100.0 * len(arrays["xyz"]) / total_rays),
        "static_hits": int(np.count_nonzero(arrays["source_type"] == 0)),
        "dynamic_hits": int(np.count_nonzero(arrays["source_type"] == 1)),
        "semantic_counts": {str(key): count for key, count in sorted(semantics.items())},
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--caseid", required=True)
    parser.add_argument("--reconstruction-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=sorted(SCALA2_SENSOR_EXTRINSICS),
        default=["front_center"],
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive")
    parser.add_argument("--first-mirror-side", type=int, choices=[0, 1], default=0)
    parser.add_argument("--minimum-range", type=float, default=0.5)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--hit-radius", type=float, default=0.10)
    parser.add_argument(
        "--intersection-mode",
        choices=["point_tube", "tangent_patch"],
        default="point_tube",
        help=(
            "point_tube accepts a sample near a ray; tangent_patch requires "
            "a finite local MLS tangent-plane intersection."
        ),
    )
    parser.add_argument(
        "--patch-radius",
        type=float,
        default=0.05,
        help="Finite local MLS tangent-patch radius in metres.",
    )
    parser.add_argument(
        "--ray-neighbor-count",
        type=int,
        default=0,
        help=(
            "0 performs exact angular-radius candidate lookup; a positive "
            "value tests only that many nearest rays as a faster approximation"
        ),
    )
    parser.add_argument("--point-batch-size", type=int, default=500000)
    parser.add_argument("--static-tile-cache", type=int, default=64)
    parser.add_argument(
        "--npz-compression",
        choices=["stored", "compressed"],
        default="compressed",
        help=(
            "Raycast output NPZ mode. stored preserves identical arrays and is "
            "much faster/larger; compressed preserves the original disk format."
        ),
    )
    parser.add_argument(
        "--sensor-workers",
        type=int,
        default=1,
        help=(
            "Number of sensors rendered concurrently within the same frame. "
            "Use 1 for a single-sensor benchmark. For six sensors, 2-3 is a "
            "good starting point on a many-core CPU because the workload is "
            "memory-bandwidth heavy."
        ),
    )
    parser.add_argument(
        "--ckdtree-workers",
        type=int,
        default=0,
        help=(
            "CPU workers used by each SciPy cKDTree query. 0=auto. In auto "
            "mode, a single sensor uses all cores; concurrent sensors split "
            "the available logical CPUs between sensor workers."
        ),
    )
    parser.add_argument("--no-tile-fov-cull", action="store_true", help="Disable conservative exact tile-level visibility culling for validation.")
    parser.add_argument("--property-cache", type=int, default=64, help="Number of lazily loaded property payloads retained in RAM.")
    parser.add_argument(
        "--candidate-lookup",
        choices=["structured", "ckdtree"],
        default="structured",
        help=(
            "Exact angular candidate lookup for --ray-neighbor-count 0. "
            "'structured' exploits the 16x653 SCALA2 lattice; 'ckdtree' "
            "uses the original scipy radius search for validation."
        ),
    )
    parser.add_argument(
        "--no-fov-prefilter",
        action="store_true",
        help=(
            "Disable the conservative SCALA2 angular FOV prefilter. "
            "Useful for validating that optimized and baseline outputs match."
        ),
    )
    parser.add_argument(
        "--force-fov-prefilter",
        action="store_true",
        help=(
            "Force the older separate point-level FOV prefilter even when the "
            "exact structured lookup is active. By default structured lookup "
            "fuses the same rejection into candidate generation and avoids a "
            "second azimuth/elevation pass."
        ),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Raycast the MLS static world only; do not add dynamic-object models.",
    )
    parser.add_argument(
        "--ground-only",
        action="store_true",
        help=(
            "Raycast only static ground surface classes. This requires "
            "--static-only and excludes CURB by default."
        ),
    )
    parser.add_argument(
        "--include-curb",
        action="store_true",
        help=(
            "Include semantic class 17 (CURB) with --ground-only. Keep it "
            "disabled for the initial hybrid experiment to avoid smoothing "
            "across the road/sidewalk discontinuity."
        ),
    )
    parser.add_argument("--noise-output", choices=["clean", "noisy", "both"], default="clean", help="clean=existing points/, noisy=points_noisy/ only, both=write both without reraycasting twice")
    parser.add_argument("--noise-range-sigma-m", type=float, default=0.05, help="Gaussian range standard deviation in metres")
    parser.add_argument("--noise-azimuth-sigma-deg", type=float, default=0.1, help="Gaussian azimuth standard deviation in degrees")
    parser.add_argument("--noise-polar-sigma-deg", type=float, default=0.6, help="Gaussian polar/elevation standard deviation in degrees")
    parser.add_argument("--noise-seed", type=int, default=12345, help="Base seed; realization is deterministic per frame and sensor")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.minimum_range < 0 or args.max_range <= args.minimum_range:
        parser.error("Require 0 <= minimum-range < max-range")
    if args.hit_radius <= 0 or args.patch_radius <= 0:
        parser.error("--hit-radius and --patch-radius must be positive")
    if args.ray_neighbor_count < 0 or args.point_batch_size < 1:
        parser.error("Neighbor count must be non-negative and batch size positive")
    if args.sensor_workers < 1:
        parser.error("--sensor-workers must be >= 1")
    if args.property_cache < 0:
        parser.error("--property-cache must be >= 0")
    if args.ckdtree_workers < -1:
        parser.error("--ckdtree-workers must be -1, 0, or a positive integer")
    if args.ground_only and not args.static_only:
        parser.error("--ground-only requires --static-only")
    if args.include_curb and not args.ground_only:
        parser.error("--include-curb is meaningful only with --ground-only")
    if min(args.noise_range_sigma_m, args.noise_azimuth_sigma_deg, args.noise_polar_sigma_deg) < 0:
        parser.error("Noise standard deviations must be non-negative")
    args.dataset_root = args.dataset_root.resolve()
    if args.reconstruction_root is None:
        args.reconstruction_root = args.dataset_root / "mls_baseline" / args.caseid
    else:
        args.reconstruction_root = args.reconstruction_root.resolve()
    if args.output_root is None:
        args.output_root = args.reconstruction_root / "scala2_raycast"
    else:
        args.output_root = args.output_root.resolve()
    return args


def main():
    args = parse_args()

    calibration_path = (
        args.dataset_root / "laser_calibrations" / args.caseid
        / "laser_calibrations" / "laser_calibrations.npz"
    )
    with np.load(calibration_path, allow_pickle=False) as calibration:
        frame_poses = np.asarray(calibration["frame_pose"], dtype=np.float64)

    frame_count = len(frame_poses)
    end_frame = frame_count if args.end_frame is None else args.end_frame
    if not (0 <= args.start_frame < end_frame <= frame_count):
        raise ValueError(
            f"Invalid frame interval [{args.start_frame}, {end_frame}) for {frame_count} frames"
        )

    frame_mapping = load_frame_mapping(args.dataset_root, args.caseid, frame_count)
    dynamic_tracks = (
        [] if args.static_only
        else load_active_dynamic_tracks(args.dataset_root, args.caseid)
    )
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
    missing_models: set[int] = set()
    missing_models_lock = threading.Lock()

    sensor_worker_count = min(args.sensor_workers, len(args.sensors))
    effective_fov_prefilter = (
        not args.no_fov_prefilter
        and (args.candidate_lookup != "structured" or args.force_fov_prefilter)
    )
    cpu_count = os.cpu_count() or 1
    if args.ckdtree_workers == 0:
        if sensor_worker_count == 1:
            ckdtree_workers = -1
        else:
            ckdtree_workers = max(1, cpu_count // sensor_worker_count)
    else:
        ckdtree_workers = args.ckdtree_workers

    # IMPORTANT: output remains SENSOR-FIRST exactly as requested:
    # output_root/front_left/points/017.npz
    # output_root/front_center/points/017.npz
    # ...
    point_dirs = {}
    noisy_point_dirs = {}
    for sensor_name in args.sensors:
        point_dirs[sensor_name] = args.output_root / sensor_name / "points"
        noisy_point_dirs[sensor_name] = args.output_root / sensor_name / "points_noisy"
        if args.noise_output in ("clean", "both"): point_dirs[sensor_name].mkdir(parents=True, exist_ok=True)
        if args.noise_output in ("noisy", "both"): noisy_point_dirs[sensor_name].mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SCALA2 MLS POINT-SURFACE RAYCAST - CPU OPTIMIZED EXACT + TILE CULL + LAZY PROPERTIES")
    print("=" * 72)
    print(f"Case             : {args.caseid}")
    print(f"Frames           : {args.start_frame}..{end_frame - 1}")
    print(f"Sensors          : {args.sensors}")
    print(f"Output layout    : clean=points/, noisy=points_noisy/, selected={args.noise_output}")
    print(f"Maximum range    : {args.max_range:.3f} m")
    print(f"Ray hit radius   : {args.hit_radius:.3f} m")
    print(f"Intersection mode: {args.intersection_mode}")
    if args.intersection_mode == "tangent_patch":
        print(f"Patch radius     : {args.patch_radius:.3f} m")
    print(f"First mirror side: MS{args.first_mirror_side}")
    if effective_fov_prefilter:
        print("FOV prefilter    : separate conservative pass")
    elif args.candidate_lookup == "structured" and not args.no_fov_prefilter:
        print("FOV prefilter    : fused into structured exact lookup")
    else:
        print("FOV prefilter    : disabled")
    print(f"Point batch size : {args.point_batch_size:,}")
    print(f"Static tile cache: {args.static_tile_cache}")
    print(f"Output NPZ mode  : {args.npz_compression}")
    print(f"Property cache   : {args.property_cache}")
    print(f"Tile FOV cull    : {not args.no_tile_fov_cull}")
    print(f"Lazy properties  : True")
    print(f"Candidate lookup : {args.candidate_lookup}")
    print(f"Sensor workers   : {sensor_worker_count}")
    print(f"cKDTree workers  : {ckdtree_workers} ({'all cores' if ckdtree_workers == -1 else 'per sensor'})")
    print(f"Logical CPUs     : {cpu_count}")
    print(f"Static only      : {args.static_only}")
    print(f"Ground only      : {args.ground_only}")
    print(f"Noise output     : {args.noise_output}")
    if args.noise_output != "clean": print(f"Noise sigma      : range={args.noise_range_sigma_m:.3f} m, azimuth={args.noise_azimuth_sigma_deg:.3f} deg, polar={args.noise_polar_sigma_deg:.3f} deg, seed={args.noise_seed}")
    if static_semantic_ids is not None:
        print(f"Static semantics : {sorted(static_semantic_ids)}")

    total_rays = SCALA2_HEIGHT * SCALA2_WIDTH
    run_summary = {
        "case": args.caseid,
        "coordinate_convention": {
            "xyz": "SCALA2 sensor frame, exact point on selected ray",
            "xyz_world": "Waymo world frame, exact point on selected ray",
            "surface_xyz_world": "selected MLS sample before projection onto ray",
            "source_type": {"0": "static MLS", "1": "dynamic-object MLS"},
        },
        "parameters": {
            "minimum_range_m": args.minimum_range,
            "maximum_range_m": args.max_range,
            "hit_radius_m": args.hit_radius,
            "intersection_mode": args.intersection_mode,
            "patch_radius_m": args.patch_radius,
            "ray_neighbor_count": args.ray_neighbor_count,
            "fov_prefilter": bool(effective_fov_prefilter),
            "fov_prefilter_fused": bool(
                args.candidate_lookup == "structured"
                and not args.no_fov_prefilter
                and not args.force_fov_prefilter
            ),
            "point_batch_size": args.point_batch_size,
            "static_tile_cache": args.static_tile_cache,
            "npz_compression": args.npz_compression,
            "property_cache": args.property_cache,
            "tile_fov_cull": not args.no_tile_fov_cull,
            "lazy_properties": True,
            "candidate_lookup": args.candidate_lookup,
            "sensor_workers": sensor_worker_count,
            "ckdtree_workers": ckdtree_workers,
            "first_mirror_side": args.first_mirror_side,
            "ground_only": args.ground_only,
            "noise_output": args.noise_output,
            "noise_model": "independent_gaussian_spherical_post_hit_v1" if args.noise_output != "clean" else None,
            "noise_range_sigma_m": args.noise_range_sigma_m,
            "noise_azimuth_sigma_deg": args.noise_azimuth_sigma_deg,
            "noise_polar_sigma_deg": args.noise_polar_sigma_deg,
            "noise_seed": args.noise_seed,
            "static_semantic_ids": (
                None if static_semantic_ids is None
                else sorted(static_semantic_ids)
            ),
        },
        "sensors": {sensor_name: [] for sensor_name in args.sensors},
    }

    def render_sensor_frame(
        sensor_name: str,
        frame_index: int,
        geometry: dict[str, np.ndarray],
        mirror_side: int,
        prepared_dynamic: list[tuple[int, dict, Path, tuple[np.ndarray, np.ndarray]]],
    ):
        clean_output_path = point_dirs[sensor_name] / f"{frame_index:03d}.npz"
        noisy_output_path = noisy_point_dirs[sensor_name] / f"{frame_index:03d}.npz"
        required_paths = ([clean_output_path] if args.noise_output == "clean" else [noisy_output_path] if args.noise_output == "noisy" else [clean_output_path, noisy_output_path])
        reuse_path = clean_output_path if args.noise_output in ("clean", "both") else noisy_output_path

        if all(path.exists() for path in required_paths) and not args.overwrite:
            output_path = reuse_path
            with np.load(output_path, allow_pickle=False) as existing:
                expected_values = {
                    "minimum_range_m": args.minimum_range,
                    "maximum_range_m": args.max_range,
                    "hit_radius_m": args.hit_radius,
                    "patch_radius_m": args.patch_radius,
                    "ray_neighbor_count": args.ray_neighbor_count,
                    "frame_mirror_side": mirror_side,
                }
                for name, expected in expected_values.items():
                    actual = float(np.asarray(existing[name]).reshape(-1)[0])
                    if not np.isclose(actual, expected):
                        raise RuntimeError(
                            f"Existing {output_path} has {name}={actual}, "
                            f"expected {expected}. Use --overwrite or a new output root."
                        )

                actual_mode = str(
                    np.asarray(existing["intersection_mode"]).reshape(-1)[0]
                )
                if actual_mode != args.intersection_mode:
                    raise RuntimeError(
                        f"Existing {output_path} has intersection_mode={actual_mode!r}, "
                        f"expected {args.intersection_mode!r}. "
                        "Use --overwrite or a new output root."
                    )

                summary = frame_summary(
                    {
                        name: np.asarray(existing[name])
                        for name in ("xyz", "semantic_id", "source_type")
                    },
                    total_rays,
                )
            summary.update({
                "output_frame_index": frame_index,
                "status": "reused",
            })
            return sensor_name, summary, None

        started = time.time()
        sensor_to_world, _ = scala2_world_pose(
            frame_poses[frame_index], sensor_name
        )
        raycaster = PointSurfaceRaycaster(
            geometry,
            sensor_to_world,
            args.minimum_range,
            args.max_range,
            args.hit_radius,
            args.intersection_mode,
            args.patch_radius,
            args.ray_neighbor_count,
            args.point_batch_size,
            use_fov_prefilter=effective_fov_prefilter,
            ckdtree_workers=ckdtree_workers,
            candidate_lookup=args.candidate_lookup,
        )

        tile_support_radius = (
            args.hit_radius
            if args.intersection_mode == "point_tube"
            else args.patch_radius
        )
        broad_tiles = static_store.relevant_tiles(
            raycaster.origin_world,
            args.max_range + tile_support_radius,
        )
        visible_tiles = []
        static_tile_cull_s = 0.0
        static_geometry_io_s = 0.0

        for tile in broad_tiles:
            keep_tile = True
            if not args.no_tile_fov_cull:
                t0 = time.perf_counter()
                bounds = static_store.ensure_bounds(tile)
                if bounds is not None and bounds.get("min_xyz") is not None and bounds.get("max_xyz") is not None:
                    keep_tile = raycaster.bounds_may_hit(
                        np.asarray(bounds["min_xyz"], dtype=np.float64),
                        np.asarray(bounds["max_xyz"], dtype=np.float64),
                        tile_support_radius,
                    )
                static_tile_cull_s += time.perf_counter() - t0
            if not keep_tile:
                continue

            visible_tiles.append(tile)
            t0 = time.perf_counter()
            surface = static_store.load_geometry(tile)
            static_geometry_io_s += time.perf_counter() - t0
            raycaster.process_surface(
                surface,
                source_type=0,
                source_path=static_store.source_path(tile),
            )

        active_objects = 0
        for object_id, surface, source_path, world_geometry in prepared_dynamic:
            raycaster.process_surface(
                surface,
                source_type=1,
                source_path=source_path,
                source_object_id=object_id,
                pretransformed_world=world_geometry,
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
        output_write_started = time.perf_counter()
        noisy_arrays = None
        if args.noise_output in ("noisy", "both"):
            noisy_arrays = apply_scala2_measurement_noise(arrays, args.noise_range_sigma_m, args.noise_azimuth_sigma_deg, args.noise_polar_sigma_deg, args.noise_seed)
        if args.noise_output in ("clean", "both") and (args.overwrite or not clean_output_path.exists()): save_npz(clean_output_path, arrays, args.npz_compression)
        if args.noise_output in ("noisy", "both") and (args.overwrite or not noisy_output_path.exists()): save_npz(noisy_output_path, noisy_arrays, args.npz_compression)
        output_write_s = time.perf_counter() - output_write_started
        static_store.flush_bounds()

        summary = frame_summary(noisy_arrays if args.noise_output == "noisy" else arrays, total_rays)
        summary.update({
            "output_frame_index": frame_index,
            "source_frame_index": frame_mapping.get(frame_index, frame_index),
            "mirror_side": mirror_side,
            "static_tiles_considered": len(broad_tiles),
            "static_tiles_visible": len(visible_tiles),
            "static_tiles_culled": len(broad_tiles) - len(visible_tiles),
            "static_tile_cull_seconds": float(static_tile_cull_s),
            "static_geometry_io_seconds": float(static_geometry_io_s),
            "output_write_seconds": float(output_write_s),
            "active_dynamic_models": active_objects,
            "points_examined": raycaster.points_examined,
            "points_in_range": raycaster.points_in_range,
            "points_in_fov": raycaster.points_in_fov,
            "candidate_pairs": raycaster.candidate_pairs,
            "timing_breakdown_s": {
                key: float(value)
                for key, value in raycaster.timing.items()
            },
            "seconds": time.time() - started,
            "status": "completed",
        })

        diagnostics = (
            f"{sensor_name} frame {frame_index:03d} MS{mirror_side}: "
            f"{summary['hits']:,}/{total_rays:,} hits "
            f"({summary['coverage_percent']:.2f}%), "
            f"dynamic={summary['dynamic_hits']:,}, "
            f"{summary['seconds']:.1f}s\n"
            f"  points: examined={raycaster.points_examined:,} "
            f"in_range={raycaster.points_in_range:,} "
            f"in_scala_fov={raycaster.points_in_fov:,} "
            f"candidate_pairs={raycaster.candidate_pairs:,}\n"
            f"  tiles: broad={len(broad_tiles):,} visible={len(visible_tiles):,} "
            f"culled={len(broad_tiles) - len(visible_tiles):,} "
            f"tile_cull={static_tile_cull_s:.2f}s geometry_io={static_geometry_io_s:.2f}s "
            f"output_write={output_write_s:.2f}s\n"
            "  timing: "
            + ", ".join(
                f"{key}={value:.2f}s"
                for key, value in raycaster.timing.items()
            )
        )
        return sensor_name, summary, diagnostics

    # Frame-first execution allows the sensors of one physical timestamp to be
    # rendered concurrently while all of them share the same static/dynamic
    # caches. Output remains sensor-first on disk.
    executor = (
        ThreadPoolExecutor(max_workers=sensor_worker_count)
        if sensor_worker_count > 1
        else None
    )

    try:
        for frame_index in range(args.start_frame, end_frame):
            mirror_side = mirror_side_for_frame(
                frame_index, args.first_mirror_side
            )
            # Same physical SCALA2 beam pattern for all six sensors at a given
            # mirror state. Each sensor has its own extrinsic pose.
            geometry = generate_scala2_ray_geometry(mirror_side)

            # Dynamic object local->world transforms are identical for all six
            # sensors at this frame. Compute them once and share read-only arrays
            # across sensor workers instead of repeating the transform six times.
            prepared_dynamic = []
            if not args.static_only:
                for track, frame_record in dynamic_by_frame.get(frame_index, ()):
                    object_id = int(track["lidargs_object_id"])
                    surface = dynamic_store.load_geometry(object_id)
                    if surface is None:
                        with missing_models_lock:
                            if object_id not in missing_models:
                                print(f"WARNING: dynamic object {object_id} has no MLS model; skip")
                                missing_models.add(object_id)
                        continue
                    local_to_world = np.asarray(
                        frame_record["box_pose_world"], dtype=np.float64
                    ).reshape(4, 4)
                    world_geometry = (
                        transform_points(surface["xyz"], local_to_world),
                        transform_normals(surface["normal"], local_to_world),
                    )
                    prepared_dynamic.append((
                        object_id, surface, dynamic_store.path(object_id), world_geometry
                    ))

            if executor is None:
                for sensor_name in args.sensors:
                    sensor_name, summary, diagnostics = render_sensor_frame(
                        sensor_name,
                        frame_index,
                        geometry,
                        mirror_side,
                        prepared_dynamic,
                    )
                    run_summary["sensors"][sensor_name].append(summary)
                    if diagnostics is None:
                        print(
                            f"{sensor_name} frame {frame_index:03d}: "
                            f"reuse {summary['hits']:,} hits"
                        )
                    else:
                        print(diagnostics)
            else:
                futures = {
                    executor.submit(
                        render_sensor_frame,
                        sensor_name,
                        frame_index,
                        geometry,
                        mirror_side,
                        prepared_dynamic,
                    ): sensor_name
                    for sensor_name in args.sensors
                }
                results = []
                for future in as_completed(futures):
                    results.append(future.result())

                # Print/save summaries in the user-requested sensor order even
                # though workers complete in arbitrary order.
                result_by_sensor = {
                    sensor_name: (summary, diagnostics)
                    for sensor_name, summary, diagnostics in results
                }
                for sensor_name in args.sensors:
                    summary, diagnostics = result_by_sensor[sensor_name]
                    run_summary["sensors"][sensor_name].append(summary)
                    if diagnostics is None:
                        print(
                            f"{sensor_name} frame {frame_index:03d}: "
                            f"reuse {summary['hits']:,} hits"
                        )
                    else:
                        print(diagnostics)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "raycast_summary.json").open("w") as stream:
        json.dump(run_summary, stream, indent=2)

    print(f"\nRaycast complete: {args.output_root}")


if __name__ == "__main__":
    main()