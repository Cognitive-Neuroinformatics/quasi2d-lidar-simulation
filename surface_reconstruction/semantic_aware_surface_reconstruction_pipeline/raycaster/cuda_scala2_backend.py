#!/usr/bin/env python3
"""CUDA backend for exact structured SCALA2 tangent-patch raycasting.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
import torch


@dataclass
class CacheEntry:
    xyz: torch.Tensor
    normal: torch.Tensor
    source_row: torch.Tensor | None
    source_count: int
    nbytes: int


class CudaGeometryCache:
    """LRU cache of source geometry resident on one device.

    The cache stores only xyz, normal and optional source-row mapping. Labels,
    intensity and provenance stay on CPU and are resolved lazily after the final
    winning source samples are known.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, max_gb: float = 6.0):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_bytes = int(max(0.0, float(max_gb)) * (1024 ** 3))
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()
        self.h2d_seconds = 0.0
        self.h2d_bytes = 0

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor | None) -> int:
        return 0 if tensor is None else tensor.numel() * tensor.element_size()

    def _evict_until(self, need_bytes: int) -> None:
        if self.max_bytes <= 0:
            return
        while self.cache and self.bytes + need_bytes > self.max_bytes:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.nbytes
            del old

    def get(self, path: Path, surface: dict) -> CacheEntry:
        key = str(path)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

            xyz_np = np.ascontiguousarray(surface["xyz"])
            normal_np = np.ascontiguousarray(surface["normal"])
            source_count = int(surface["_source_count"])
            source_row_np = surface.get("_source_row")

            # Estimate before allocating so LRU eviction happens first.
            element_size = torch.tensor([], dtype=self.dtype).element_size()
            estimate = xyz_np.size * element_size + normal_np.size * element_size
            if source_row_np is not None:
                estimate += np.asarray(source_row_np).size * 8
            self._evict_until(estimate)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            xyz = torch.as_tensor(xyz_np, device=self.device, dtype=self.dtype)
            normal = torch.as_tensor(normal_np, device=self.device, dtype=self.dtype)
            source_row = None
            if source_row_np is not None:
                source_row = torch.as_tensor(
                    np.ascontiguousarray(source_row_np),
                    device=self.device,
                    dtype=torch.int64,
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - t0

            nbytes = self._tensor_bytes(xyz) + self._tensor_bytes(normal) + self._tensor_bytes(source_row)
            self.h2d_seconds += elapsed
            self.h2d_bytes += nbytes

            entry = CacheEntry(
                xyz=xyz,
                normal=normal,
                source_row=source_row,
                source_count=source_count,
                nbytes=nbytes,
            )

            # Very large single entries are still usable but not retained if
            # they exceed the configured cache budget.
            if self.max_bytes <= 0 or nbytes > self.max_bytes:
                return entry

            self.cache[key] = entry
            self.bytes += nbytes
            self.cache.move_to_end(key)
            return entry

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.bytes = 0
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class TorchStructuredScala2:
    """Exact structured candidate lookup for the 16x653 SCALA2 lattice."""

    def __init__(
        self,
        directions_sensor: np.ndarray,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)

        directions = np.asarray(directions_sensor, dtype=np.float64).reshape(height, width, 3)
        directions = directions / np.linalg.norm(directions, axis=2, keepdims=True)
        self.directions = torch.as_tensor(directions, device=device, dtype=dtype)
        self.flat_directions = self.directions.reshape(-1, 3)

        row_azimuth = np.arctan2(directions[:, :, 1], directions[:, :, 0])
        if not np.all(np.diff(row_azimuth, axis=1) > 0):
            raise RuntimeError("SCALA2 row azimuths are not strictly increasing")
        row_elevation = np.arctan2(
            directions[:, :, 2],
            np.hypot(directions[:, :, 0], directions[:, :, 1]),
        )

        self.row_azimuth = torch.as_tensor(row_azimuth, device=device, dtype=dtype)
        self.row_elevation_min = torch.as_tensor(row_elevation.min(axis=1), device=device, dtype=dtype)
        self.row_elevation_max = torch.as_tensor(row_elevation.max(axis=1), device=device, dtype=dtype)

    def candidates(
        self,
        unit: torch.Tensor,
        angular_limit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact (point_row, ray_index) pairs.

        The method mirrors the validated NumPy implementation. It first derives
        a conservative spherical-cap longitude interval within each of the 16
        rows, binary-searches possible columns, then applies the exact dot-product
        angular criterion.
        """
        if unit.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty

        point_azimuth = torch.atan2(unit[:, 1], unit[:, 0])
        point_elevation = torch.atan2(
            unit[:, 2],
            torch.hypot(unit[:, 0], unit[:, 1]),
        )

        tiny = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        ratio = torch.sin(angular_limit) / torch.clamp(torch.cos(point_elevation), min=tiny)
        longitude_margin = torch.asin(torch.clamp(ratio, min=0.0, max=1.0))
        cos_limit = torch.cos(angular_limit)

        point_chunks: list[torch.Tensor] = []
        ray_chunks: list[torch.Tensor] = []

        for row in range(self.height):
            possible = (
                (point_elevation >= self.row_elevation_min[row] - angular_limit)
                & (point_elevation <= self.row_elevation_max[row] + angular_limit)
            )
            point_ids = torch.nonzero(possible, as_tuple=False).flatten()
            if point_ids.numel() == 0:
                continue

            az = point_azimuth[point_ids]
            margin = longitude_margin[point_ids]
            row_az = self.row_azimuth[row]
            lo = torch.searchsorted(row_az, az - margin, right=False)
            hi = torch.searchsorted(row_az, az + margin, right=True)
            counts = hi - lo
            keep = counts > 0
            if not torch.any(keep):
                continue

            point_ids = point_ids[keep]
            lo = lo[keep].to(torch.int64)
            counts = counts[keep].to(torch.int64)
            total = int(counts.sum().item())
            if total == 0:
                continue

            starts = torch.cumsum(counts, dim=0) - counts
            pair_points = torch.repeat_interleave(point_ids, counts)
            pair_lo = torch.repeat_interleave(lo, counts)
            pair_starts = torch.repeat_interleave(starts, counts)
            columns = pair_lo + (
                torch.arange(total, device=self.device, dtype=torch.int64) - pair_starts
            )
            pair_rays = row * self.width + columns

            dot = torch.sum(unit[pair_points] * self.flat_directions[pair_rays], dim=1)
            # Match the CPU boundary allowance. For float32 the 1e-12 term is
            # below machine epsilon and therefore has no practical effect.
            exact = dot >= (cos_limit[pair_points] - 1e-12)
            if torch.any(exact):
                point_chunks.append(pair_points[exact])
                ray_chunks.append(pair_rays[exact])

        if not point_chunks:
            empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return empty, empty
        return torch.cat(point_chunks), torch.cat(ray_chunks)


def per_ray_nearest_candidate(
    ray_indices: torch.Tensor,
    ranges: torch.Tensor,
    distances: torch.Tensor,
    ray_count: int,
) -> torch.Tensor:
    """Return candidate positions for range-minimum, distance-tiebroken hits."""
    device = ray_indices.device
    dtype = ranges.dtype
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)

    min_range = torch.full((ray_count,), inf, dtype=dtype, device=device)
    min_range.scatter_reduce_(0, ray_indices, ranges, reduce="amin", include_self=True)
    range_min_mask = ranges == min_range[ray_indices]
    range_positions = torch.nonzero(range_min_mask, as_tuple=False).flatten()
    if range_positions.numel() == 0:
        return range_positions

    rays2 = ray_indices[range_positions]
    dist2 = distances[range_positions]
    min_dist = torch.full((ray_count,), inf, dtype=dtype, device=device)
    min_dist.scatter_reduce_(0, rays2, dist2, reduce="amin", include_self=True)
    dist_min_mask = dist2 == min_dist[rays2]
    final_positions = range_positions[dist_min_mask]
    if final_positions.numel() == 0:
        return final_positions

    # Exact duplicate candidates can survive both reductions. Keep the first,
    # mirroring the deterministic CPU behavior without sorting all candidates.
    final_rays = ray_indices[final_positions]
    sentinel = torch.iinfo(torch.int64).max
    first = torch.full((ray_count,), sentinel, dtype=torch.int64, device=device)
    first.scatter_reduce_(0, final_rays, final_positions, reduce="amin", include_self=True)
    active = torch.nonzero(first != sentinel, as_tuple=False).flatten()
    return first[active]
