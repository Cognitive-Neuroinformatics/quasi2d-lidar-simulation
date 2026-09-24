#!/usr/bin/env python3
"""Build a semantic, instance-aware PCL-MLS scene for Scala2 raycasting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from scipy.spatial import cKDTree


ATTRIBUTES = ("intensity", "semantic_id", "ground_id", "instance_id")
SURFACE_ARRAYS = ("xyz", "normal", *ATTRIBUTES)
FOREGROUND_IDS = frozenset((1, 2, 3, 4, 5, 6, 7, 12, 13))
GROUND_IDS = frozenset((18, 19, 20, 21, 22))


def save_npz(path: Path, arrays: dict[str, np.ndarray], compression: str) -> None:
    if compression == "stored":
        np.savez(path, **arrays)
    elif compression == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError(f"Unknown NPZ compression mode: {compression}")


def reconstruct_from_indices(runner, data, indices, config, tag):
    """Thread-pool helper; only XYZ is materialized before PCL.

    Attribute arrays stay in the original scene arrays until the final nearest
    source point for each MLS output sample is known. This avoids repeatedly
    copying intensity/semantic/ground/instance arrays for every tile/group.
    """
    return runner.reconstruct_indices(data, indices, config, tag)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted({"xyz", *ATTRIBUTES} - set(data.files))
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        result = {
            "xyz": np.asarray(data["xyz"], dtype=np.float32),
            "intensity": np.asarray(data["intensity"], dtype=np.float32),
            "semantic_id": np.asarray(data["semantic_id"], dtype=np.int16),
            "ground_id": np.asarray(data["ground_id"], dtype=np.int8),
            "instance_id": np.asarray(data["instance_id"], dtype=np.int32),
        }
        for optional, dtype in (
            ("label_confidence", np.float32),
            ("observation_frame_index", np.int32),
        ):
            if optional in data.files:
                result[optional] = np.asarray(data[optional], dtype=dtype)
    count = len(result["xyz"])
    if result["xyz"].shape != (count, 3):
        raise ValueError(f"{path}: xyz must have shape (N, 3)")
    if any(len(values) != count for values in result.values()):
        raise ValueError(f"{path}: aligned-array length mismatch")
    finite = np.isfinite(result["xyz"]).all(axis=1)
    if not np.all(finite):
        print(f"WARNING: dropping {np.count_nonzero(~finite):,} non-finite points")
        result = {name: values[finite] for name, values in result.items()}
    return result


def empty_surface() -> dict[str, np.ndarray]:
    return {
        "xyz": np.empty((0, 3), dtype=np.float32),
        "normal": np.empty((0, 3), dtype=np.float32),
        "intensity": np.empty(0, dtype=np.float32),
        "semantic_id": np.empty(0, dtype=np.int16),
        "ground_id": np.empty(0, dtype=np.int8),
        "instance_id": np.empty(0, dtype=np.int32),
    }


def concatenate_surfaces(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    nonempty = [part for part in parts if len(part["xyz"])]
    if not nonempty:
        return empty_surface()
    return {
        name: np.concatenate([part[name] for part in nonempty], axis=0)
        for name in SURFACE_ARRAYS
    }


def write_xyz_pcd(path: Path, xyz: np.ndarray) -> None:
    points = np.ascontiguousarray(xyz, dtype="<f4")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(points.tobytes(order="C"))


def _pcd_numpy_dtype(fields, sizes, types, counts):
    names = []
    formats = []
    for field, size, kind, count in zip(fields, sizes, types, counts):
        code = {
            ("F", 4): "<f4",
            ("F", 8): "<f8",
            ("I", 1): "i1",
            ("I", 2): "<i2",
            ("I", 4): "<i4",
            ("U", 1): "u1",
            ("U", 2): "<u2",
            ("U", 4): "<u4",
        }.get((kind, size))
        if code is None:
            raise ValueError(f"Unsupported PCD field type: TYPE={kind}, SIZE={size}")
        if count == 1:
            names.append(field)
            formats.append(code)
        else:
            names.append(field)
            formats.append((code, (count,)))
    return np.dtype({"names": names, "formats": formats})


def read_point_normal_pcd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: truncated PCD header")
            decoded = line.decode("ascii").strip()
            if not decoded or decoded.startswith("#"):
                continue
            key, *values = decoded.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        mode = header["DATA"][0].lower()
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        count = int(header["POINTS"][0])
        if mode == "binary":
            dtype = _pcd_numpy_dtype(fields, sizes, types, counts)
            records = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype, count=count)
            xyz = np.column_stack([records[name] for name in ("x", "y", "z")])
            normal = np.column_stack(
                [records[name] for name in ("normal_x", "normal_y", "normal_z")]
            )
        elif mode == "ascii":
            values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
            field_index = {name: index for index, name in enumerate(fields)}
            xyz = values[:, [field_index[name] for name in ("x", "y", "z")]]
            normal = values[
                :, [field_index[name] for name in ("normal_x", "normal_y", "normal_z")]
            ]
        else:
            raise ValueError(f"{path}: only PCD binary/ascii are supported, got {mode}")
    return np.asarray(xyz, np.float32), np.asarray(normal, np.float32)


class PCLMLSRunner:
    def __init__(self, executable: Path, work_root: Path, threads: int, minimum_points: int, attribute_workers: int = 1):
        self.executable = executable
        self.work_root = work_root
        self.threads = threads
        self.minimum_points = minimum_points
        self.attribute_workers = int(attribute_workers)
        self.skipped_surfaces: list[dict] = []
        self.call_timings: list[dict] = []
        self.lock = threading.RLock()
        self.work_root.mkdir(parents=True, exist_ok=True)

    def reconstruct(
        self,
        source: dict[str, np.ndarray],
        config: dict,
        tag: str,
    ) -> dict[str, np.ndarray]:
        """Compatibility path for already-materialized source dictionaries."""
        return self._reconstruct_xyz(
            np.asarray(source["xyz"], dtype=np.float32),
            source,
            None,
            config,
            tag,
        )

    def reconstruct_indices(
        self,
        data: dict[str, np.ndarray],
        indices: np.ndarray,
        config: dict,
        tag: str,
    ) -> dict[str, np.ndarray]:
        """Reconstruct from scene indices without copying all source metadata.

        Only XYZ must be materialized for the PCL subprocess and source KD-tree.
        Once the nearest original source row is known for every MLS output point,
        the four output attributes are gathered directly from the original scene
        arrays. The numerical result is identical to slicing all attributes first.
        """
        indices = np.asarray(indices, dtype=np.int64)
        source_xyz = np.ascontiguousarray(data["xyz"][indices], dtype=np.float32)
        return self._reconstruct_xyz(source_xyz, data, indices, config, tag)

    def _reconstruct_xyz(
        self,
        source_xyz: np.ndarray,
        attribute_data: dict[str, np.ndarray],
        source_indices: np.ndarray | None,
        config: dict,
        tag: str,
    ) -> dict[str, np.ndarray]:
        input_count = int(len(source_xyz))
        if input_count < self.minimum_points:
            print(f"  {tag}: skip; only {input_count:,} source points")
            return empty_surface()

        call_started = time.perf_counter()
        write_s = pcl_s = read_s = transfer_s = 0.0
        with tempfile.TemporaryDirectory(prefix="mls_", dir=self.work_root) as temp_name:
            temp = Path(temp_name)
            input_path = temp / "input.pcd"
            output_path = temp / "output.pcd"

            stage_started = time.perf_counter()
            write_xyz_pcd(input_path, source_xyz)
            write_s = time.perf_counter() - stage_started

            command = [
                str(self.executable),
                "--input", str(input_path),
                "--output", str(output_path),
                "--search-radius", str(config["search_radius"]),
                "--voxel-size", str(config.get("voxel_size", 0.0)),
                "--polynomial-order", str(config.get("polynomial_order", 2)),
                "--threads", str(self.threads),
                "--upsampling", config.get("upsampling", "none"),
            ]
            if config.get("upsampling", "none") == "sample_local_plane":
                command.extend([
                    "--upsampling-radius", str(config["upsampling_radius"]),
                    "--upsampling-step", str(config["upsampling_step"]),
                ])

            try:
                stage_started = time.perf_counter()
                pcl_env = os.environ.copy()
                # The PCL binary is linked against the system PCL stack. Conda's
                # LD_LIBRARY_PATH can shadow ABI-incompatible system libraries.
                pcl_env.pop("LD_LIBRARY_PATH", None)
                completed = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=pcl_env,
                )
                pcl_s = time.perf_counter() - stage_started
            except subprocess.CalledProcessError as error:
                pcl_output = (error.stdout or "").strip()
                if "MLS produced no finite point normals" in pcl_output:
                    print(
                        f"  {tag}: skip; PCL MLS produced no finite point normals "
                        f"from {input_count:,} source points"
                    )
                    with self.lock:
                        self.skipped_surfaces.append({
                            "tag": tag,
                            "input_points": input_count,
                            "reason": "no_finite_point_normals",
                            "pcl_output": pcl_output,
                        })
                        self.call_timings.append({
                            "tag": tag,
                            "input_points": input_count,
                            "output_points": 0,
                            "write_pcd_s": float(write_s),
                            "pcl_s": float(time.perf_counter() - stage_started),
                            "read_pcd_s": 0.0,
                            "attribute_transfer_s": 0.0,
                            "wall_s": float(time.perf_counter() - call_started),
                            "status": "no_finite_point_normals",
                        })
                    return empty_surface()
                details = pcl_output or "PCL produced no diagnostic output"
                raise RuntimeError(
                    f"{tag}: PCL MLS failed with exit code {error.returncode}:\n{details}"
                ) from error

            print(f"  {tag}: {completed.stdout.strip()}")
            stage_started = time.perf_counter()
            xyz, normal = read_point_normal_pcd(output_path)
            read_s = time.perf_counter() - stage_started

        norm = np.linalg.norm(normal, axis=1)
        keep = (
            np.isfinite(xyz).all(axis=1)
            & np.isfinite(normal).all(axis=1)
            & (norm > 1e-8)
        )
        xyz = xyz[keep]
        normal = normal[keep] / norm[keep, None]
        if bool(config.get("orient_normal_up", False)):
            flip = normal[:, 2] < 0
            normal[flip] *= -1
        if not len(xyz):
            return empty_surface()

        stage_started = time.perf_counter()
        _, nearest = cKDTree(np.asarray(source_xyz, np.float64)).query(
            np.asarray(xyz, np.float64),
            k=1,
            workers=self.attribute_workers,
        )
        transfer_s = time.perf_counter() - stage_started
        nearest = np.asarray(nearest, dtype=np.int64)

        if source_indices is None:
            attribute_rows = nearest
        else:
            attribute_rows = np.asarray(source_indices, dtype=np.int64)[nearest]

        result = {
            "xyz": xyz.astype(np.float32, copy=False),
            "normal": normal.astype(np.float32, copy=False),
        }
        for name in ATTRIBUTES:
            result[name] = attribute_data[name][attribute_rows]

        with self.lock:
            self.call_timings.append({
                "tag": tag,
                "input_points": input_count,
                "output_points": int(len(result["xyz"])),
                "write_pcd_s": float(write_s),
                "pcl_s": float(pcl_s),
                "read_pcd_s": float(read_s),
                "attribute_transfer_s": float(transfer_s),
                "wall_s": float(time.perf_counter() - call_started),
                "status": "ok",
            })
        return result


class StaticTileIndex:
    def __init__(self, xyz: np.ndarray, tile_size: float):
        self.tile_size = float(tile_size)
        self.ix = np.floor(xyz[:, 0] / tile_size).astype(np.int32)
        self.iy = np.floor(xyz[:, 1] / tile_size).astype(np.int32)
        keys = (self.ix.astype(np.int64) << 32) | self.iy.astype(np.uint32).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
        self.order = order
        self.bins = {
            self.decode_key(int(key)): order[start:start + count]
            for key, start, count in zip(unique, starts, counts)
        }

    @staticmethod
    def decode_key(key: int) -> tuple[int, int]:
        raw_ix = (key >> 32) & 0xFFFFFFFF
        raw_iy = key & 0xFFFFFFFF
        ix = raw_ix if raw_ix < 2**31 else raw_ix - 2**32
        iy = raw_iy if raw_iy < 2**31 else raw_iy - 2**32
        return ix, iy

    @property
    def occupied(self) -> list[tuple[int, int]]:
        return sorted(self.bins)

    def halo_indices(self, ix: int, iy: int, halo: float) -> np.ndarray:
        radius = max(1, int(math.ceil(halo / self.tile_size)))
        chunks = []
        for x_index in range(ix - radius, ix + radius + 1):
            for y_index in range(iy - radius, iy + radius + 1):
                values = self.bins.get((x_index, y_index))
                if values is not None:
                    chunks.append(values)
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    def bounds(self, ix: int, iy: int) -> tuple[np.ndarray, np.ndarray]:
        minimum = np.asarray([ix * self.tile_size, iy * self.tile_size], np.float64)
        return minimum, minimum + self.tile_size


def take_source(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: data[name][indices] for name in ("xyz", *ATTRIBUTES)}


def crop_surface_xy(
    surface: dict[str, np.ndarray], minimum: np.ndarray, maximum: np.ndarray
) -> dict[str, np.ndarray]:
    if not len(surface["xyz"]):
        return surface
    xy = surface["xyz"][:, :2]
    keep = np.all(xy >= minimum[None, :], axis=1) & np.all(xy < maximum[None, :], axis=1)
    return {name: values[keep] for name, values in surface.items()}


def object_group_for_semantic(config: dict, semantic_id: int) -> dict | None:
    for group in config["object_groups"]:
        if int(semantic_id) in set(group["semantic_ids"]):
            return group
    return None


def dominant_positive(values: np.ndarray) -> int:
    positive = values[values > 0]
    if not len(positive):
        return 0
    unique, counts = np.unique(positive, return_counts=True)
    return int(unique[np.argmax(counts)])


def load_box_cache(path: Path) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if not path.is_file():
        return {}
    with path.open() as stream:
        tracks = json.load(stream)["tracks"].values()
    cache: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for track in tracks:
        if int(track.get("semantic_id", 0)) not in FOREGROUND_IDS:
            continue
        for frame_text, record in track["frames"].items():
            pose = np.asarray(record["box_pose_world"], np.float64).reshape(4, 4)
            dimensions = np.asarray(record["box_vehicle"][3:6], np.float64)
            cache.setdefault(int(frame_text), []).append(
                (pose[:3, :3], pose[:3, 3], dimensions / 2.0)
            )
    return cache


def reject_tracked_box_points(
    xyz: np.ndarray,
    observation_frames: np.ndarray,
    box_cache: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    xy_margin: float,
    z_margin: float,
) -> np.ndarray:
    keep = np.ones(len(xyz), dtype=bool)
    for frame_index in np.unique(observation_frames):
        boxes = box_cache.get(int(frame_index), ())
        if not boxes:
            continue
        local_indices = np.flatnonzero(observation_frames == frame_index)
        frame_xyz = np.asarray(xyz[local_indices], np.float64)
        rejected = np.zeros(len(local_indices), dtype=bool)
        for rotation, translation, half_size in boxes:
            local = (frame_xyz - translation[None, :]) @ rotation
            margins = np.asarray([xy_margin, xy_margin, z_margin])
            rejected |= np.all(np.abs(local) <= (half_size + margins)[None, :], axis=1)
        keep[local_indices[rejected]] = False
    return keep


def semantic_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts)}


def reconstruct_instance_objects(
    data: dict[str, np.ndarray],
    config: dict,
    runner: PCLMLSRunner,
    minimum_confidence: float,
    executor: ThreadPoolExecutor,
) -> tuple[list[dict[str, np.ndarray]], list[dict]]:
    foreground = np.isin(data["semantic_id"], np.asarray(sorted(FOREGROUND_IDS), np.int16))
    foreground &= data["instance_id"] > 0
    if "label_confidence" in data:
        foreground &= data["label_confidence"] >= minimum_confidence
    foreground_indices = np.flatnonzero(foreground)
    foreground_instances = data["instance_id"][foreground_indices]
    order = np.argsort(foreground_instances, kind="stable")
    sorted_instances = foreground_instances[order]
    instance_ids, starts, counts = np.unique(
        sorted_instances, return_index=True, return_counts=True
    )

    jobs = []
    for instance_id, begin, count in zip(instance_ids, starts, counts):
        instance_indices = foreground_indices[order[begin:begin + count]]
        semantic_id = dominant_positive(data["semantic_id"][instance_indices])
        group = object_group_for_semantic(config, semantic_id)
        if group is None:
            continue
        compatible = np.isin(
            data["semantic_id"][instance_indices],
            np.asarray(group["semantic_ids"], np.int16),
        )
        indices = instance_indices[compatible]
        tag = f"static {group['name']} instance {instance_id}"
        future = executor.submit(
            reconstruct_from_indices, runner, data, indices, group, tag
        )
        jobs.append((instance_id, semantic_id, group, indices, future))

    surfaces = []
    records = []
    # Deterministic output ordering is preserved even though the PCL jobs run in parallel.
    for instance_id, semantic_id, group, indices, future in jobs:
        output = future.result()
        if len(output["xyz"]):
            surfaces.append(output)
        records.append({
            "instance_id": int(instance_id),
            "group": group["name"],
            "semantic_id": semantic_id,
            "input_points": int(len(indices)),
            "output_points": int(len(output["xyz"])),
        })
    return surfaces, records


def reconstruct_dynamic_objects(
    input_root: Path,
    output_root: Path,
    config: dict,
    runner: PCLMLSRunner,
    minimum_confidence: float,
    executor: ThreadPoolExecutor,
    npz_compression: str,
) -> list[dict]:
    del minimum_confidence  # Dynamic stitch files do not currently carry confidence.
    if not input_root.is_dir():
        print(f"WARNING: dynamic-object input directory not found: {input_root}")
        return []

    object_dirs = [
        item for item in sorted(
            input_root.iterdir(),
            key=lambda item: int(item.name) if item.name.isdigit() else 10**12,
        )
        if item.name.isdigit() and (item / "stitch_labeled.npz").is_file()
    ]

    def one_object(object_dir: Path):
        input_path = object_dir / "stitch_labeled.npz"
        data = load_npz(input_path)
        semantic_id = dominant_positive(data["semantic_id"])
        info_path = object_dir / "info.json"
        if info_path.is_file():
            with info_path.open() as stream:
                semantic_id = int(json.load(stream).get("semantic_id", semantic_id))
        group = object_group_for_semantic(config, semantic_id)
        if group is None:
            print(f"  dynamic object {object_dir.name}: skip unknown semantic {semantic_id}")
            return None
        compatible = np.isin(data["semantic_id"], np.asarray(group["semantic_ids"], np.int16))
        compatible &= data["ground_id"] != 1
        compatible &= data["instance_id"] > 0
        indices = np.flatnonzero(compatible)
        output = runner.reconstruct_indices(
            data, indices, group,
            f"dynamic {group['name']} object {object_dir.name}"
        )
        if len(output["xyz"]):
            destination = output_root / "dynamic_objects" / object_dir.name / "mls_surface.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_npz(destination, output, npz_compression)
        return {
            "object_id": int(object_dir.name),
            "group": group["name"],
            "semantic_id": semantic_id,
            "input_points_total": int(len(data["xyz"])),
            "input_points_compatible": int(len(indices)),
            "output_points": int(len(output["xyz"])),
        }

    futures = [executor.submit(one_object, object_dir) for object_dir in object_dirs]
    records = []
    for future in futures:
        record = future.result()
        if record is not None:
            records.append(record)
    return records


def bin_surfaces_by_tile(surfaces: list[dict[str, np.ndarray]], tile_size: float):
    """Assign static-object MLS points to core tiles once, preserving order."""
    result: dict[tuple[int, int], list[dict[str, np.ndarray]]] = {}
    for surface in surfaces:
        if not len(surface["xyz"]):
            continue
        ix = np.floor(surface["xyz"][:, 0] / tile_size).astype(np.int32)
        iy = np.floor(surface["xyz"][:, 1] / tile_size).astype(np.int32)
        keys = (ix.astype(np.int64) << 32) | iy.astype(np.uint32).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
        for key, begin, count in zip(unique, starts, counts):
            tile = StaticTileIndex.decode_key(int(key))
            idx = order[begin:begin + count]
            result.setdefault(tile, []).append({name: values[idx] for name, values in surface.items()})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--caseid", required=True)
    parser.add_argument("--pcl-executable", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--static-input", type=Path, default=None)
    parser.add_argument("--dynamic-input-root", type=Path, default=None)
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tile-size", type=float, default=25.0)
    parser.add_argument("--tile-halo", type=float, default=0.5)
    parser.add_argument("--pcl-threads", type=int, default=0)
    parser.add_argument("--mls-workers", type=int, default=1, help="Concurrent independent PCL MLS jobs")
    parser.add_argument("--attribute-workers", type=int, default=1, help="cKDTree workers per concurrent job for output->source attribute transfer")
    parser.add_argument("--work-root", type=Path, default=None, help="Temporary PCL PCD directory; /dev/shm can reduce I/O when large enough")
    parser.add_argument("--npz-compression", choices=["compressed", "stored"], default="compressed", help="stored is exact and much faster/larger")
    parser.add_argument("--minimum-points", type=int, default=30)
    parser.add_argument(
        "--require-no-voxel-downsampling",
        action="store_true",
        help="Refuse to run if any selected semantic or object group has voxel_size != 0",
    )
    parser.add_argument("--minimum-label-confidence", type=float, default=0.66)
    parser.add_argument("--box-xy-margin", type=float, default=0.15)
    parser.add_argument("--box-z-margin", type=float, default=0.10)
    parser.add_argument(
        "--exclude-points-in-tracked-boxes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional second box-removal pass. Disabled for the cleaned static_recon_labels.npz input.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("background", "static_objects", "dynamic_objects"),
        default=("background", "static_objects"),
    )
    parser.add_argument("--only-background-group", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.pcl_executable = args.pcl_executable.resolve()
    args.config = args.config.resolve()
    args.static_input = (
        args.static_input.resolve() if args.static_input else
        args.dataset_root / "recon_related" / args.caseid / "static_recon_labels.npz"
    )
    args.dynamic_input_root = (
        args.dynamic_input_root.resolve() if args.dynamic_input_root else
        args.dataset_root / "temp" / args.caseid / "occ" / "preproc" / "dynamic" / "objects"
    )
    args.tracks = (
        args.tracks.resolve() if args.tracks else
        args.dataset_root / "temp" / args.caseid / "stage_a_tracks.json"
    )
    args.output_root = (
        args.output_root.resolve() if args.output_root else
        args.dataset_root / "pc_transfer" / "pcl_semantic_mls" / args.caseid
    )
    if args.tile_size <= 0 or args.tile_halo <= 0:
        parser.error("tile size and halo must be positive")
    if not (0.0 <= args.minimum_label_confidence <= 1.0):
        parser.error("minimum label confidence must be in [0, 1]")
    if args.minimum_points < 3 or args.pcl_threads < 0:
        parser.error("minimum points must be >=3 and threads non-negative")
    if args.mls_workers < 1 or args.attribute_workers == 0 or args.attribute_workers < -1:
        parser.error("--mls-workers >=1 and --attribute-workers must be -1 or >=1")
    return args


def main() -> None:
    args = parse_args()
    if not args.pcl_executable.is_file() or not args.pcl_executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"PCL executable is missing or not executable: {args.pcl_executable}")
    with args.config.open() as stream:
        config = json.load(stream)
    selected_groups = set(args.only_background_group)
    known_groups = {group["name"] for group in config["background_groups"]}
    if selected_groups - known_groups:
        raise ValueError(f"Unknown background groups: {sorted(selected_groups - known_groups)}")
    background_groups = [
        group for group in config["background_groups"]
        if not selected_groups or group["name"] in selected_groups
    ]

    # Fast exact semantic->background-group dispatch. The uploaded config uses
    # disjoint semantic sets, so every candidate point can be classified once
    # per tile instead of rescanning candidate_semantic with np.isin for every
    # background family. Fall back to the original masks if a future config
    # intentionally overlaps semantic ids across groups.
    semantic_to_groups: dict[int, list[int]] = {}
    for group_index, group in enumerate(background_groups):
        for semantic_id in group["semantic_ids"]:
            semantic_to_groups.setdefault(int(semantic_id), []).append(group_index)
    background_groups_disjoint = all(len(v) == 1 for v in semantic_to_groups.values())
    max_background_semantic = max(semantic_to_groups, default=0)
    background_group_lut = np.full(max_background_semantic + 1, -1, dtype=np.int16)
    if background_groups_disjoint:
        for semantic_id, group_indices in semantic_to_groups.items():
            background_group_lut[semantic_id] = group_indices[0]

    configured_groups = [*background_groups, *config["object_groups"]]
    nonzero_voxel_groups = [
        (group["name"], float(group.get("voxel_size", 0.0)))
        for group in configured_groups
        if float(group.get("voxel_size", 0.0)) != 0.0
    ]
    if args.require_no_voxel_downsampling and nonzero_voxel_groups:
        details = ", ".join(
            f"{name}={voxel_size:g} m" for name, voxel_size in nonzero_voxel_groups
        )
        raise ValueError(
            "--require-no-voxel-downsampling was requested, but nonzero "
            f"voxel sizes were configured: {details}"
        )
    maximum_search_radius = max(
        group["search_radius"] + group.get("upsampling_radius", 0.0)
        for group in (*background_groups, *config["object_groups"])
    )
    if args.tile_halo < maximum_search_radius:
        raise ValueError(
            f"--tile-halo {args.tile_halo} is smaller than the largest MLS support "
            f"{maximum_search_radius:.3f}; increase the halo to prevent seams"
        )

    fingerprint_payload = {
        "config": config,
        "tile_size": args.tile_size,
        "tile_halo": args.tile_halo,
        "minimum_points": args.minimum_points,
        "minimum_label_confidence": args.minimum_label_confidence,
        "box_xy_margin": args.box_xy_margin,
        "box_z_margin": args.box_z_margin,
        "exclude_points_in_tracked_boxes": args.exclude_points_in_tracked_boxes,
        "require_no_voxel_downsampling": args.require_no_voxel_downsampling,
        "stages": list(args.stages),
        "only_background_group": args.only_background_group,
        "pcl_threads": args.pcl_threads,
        "mls_workers": args.mls_workers,
        "attribute_workers": args.attribute_workers,
        "npz_compression": args.npz_compression,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = args.output_root / "static_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open() as stream:
            old_manifest = json.load(stream)
        if old_manifest.get("complete") and old_manifest.get("config_fingerprint") == fingerprint:
            raise FileExistsError(
                f"Matching reconstruction already exists at {args.output_root}; "
                "use it, choose another --output-root, or pass --overwrite"
            )
        raise FileExistsError(
            f"Output manifest already exists at {manifest_path}; use a new root or --overwrite"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "static_tiles").mkdir(parents=True, exist_ok=True)
    work_root = args.work_root.resolve() if args.work_root is not None else args.output_root / ".pcl_work"
    runner = PCLMLSRunner(
        args.pcl_executable, work_root, args.pcl_threads, args.minimum_points,
        attribute_workers=args.attribute_workers,
    )
    started = time.time()
    print("=" * 76)
    print("SEMANTIC + INSTANCE-AWARE PCL MLS")
    print("=" * 76)
    print(f"Case          : {args.caseid}")
    print(f"Static input  : {args.static_input}")
    print(f"Dynamic input : {args.dynamic_input_root}")
    print(f"Output        : {args.output_root}")
    print(f"Stages        : {list(args.stages)}")
    print(f"PCL threads   : {args.pcl_threads}")
    print(f"MLS workers   : {args.mls_workers}")
    print(f"Attr workers  : {args.attribute_workers}")
    print(f"PCL work root : {work_root}")
    print(f"NPZ mode      : {args.npz_compression}")
    if args.require_no_voxel_downsampling:
        print("Voxel sampling : DISABLED and enforced for every configured group")
    elif nonzero_voxel_groups:
        print(
            "Voxel sampling : enabled for "
            + ", ".join(name for name, _ in nonzero_voxel_groups)
        )
    else:
        print("Voxel sampling : disabled for every configured group")

    stage_timings = {}
    stage_started = time.perf_counter()
    data = load_npz(args.static_input)
    stage_timings["static_input_load_s"] = time.perf_counter() - stage_started
    print(f"Static source : {len(data['xyz']):,} points")
    print(f"Fast semantic dispatch: {background_groups_disjoint}")
    if "label_confidence" not in data:
        print("WARNING: static input has no label_confidence; confidence filtering is disabled")
    if "observation_frame_index" not in data:
        print(
            "WARNING: static input has no observation_frame_index; tracked-box contamination "
            "rejection is disabled. Semantic and instance separation still apply."
        )

    box_cache = {}
    if args.exclude_points_in_tracked_boxes and "observation_frame_index" in data:
        box_cache = load_box_cache(args.tracks)
        print(f"Tracked-box cache: {sum(map(len, box_cache.values())):,} frame boxes")

    executor = ThreadPoolExecutor(max_workers=args.mls_workers, thread_name_prefix="mls")

    static_object_surfaces: list[dict[str, np.ndarray]] = []
    static_object_records: list[dict] = []
    stage_started = time.perf_counter()
    if "static_objects" in args.stages:
        static_object_surfaces, static_object_records = reconstruct_instance_objects(
            data, config, runner, args.minimum_label_confidence, executor
        )
    stage_timings["static_objects_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    static_objects_by_tile = bin_surfaces_by_tile(static_object_surfaces, args.tile_size)
    stage_timings["static_object_tile_bin_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    tile_index = StaticTileIndex(data["xyz"], args.tile_size)
    stage_timings["static_tile_index_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    tiles = []
    background_records = []
    for tile_number, (tile_ix, tile_iy) in enumerate(tile_index.occupied, start=1):
        core_minimum, core_maximum = tile_index.bounds(tile_ix, tile_iy)
        tile_parts = []
        candidate_indices = tile_index.halo_indices(tile_ix, tile_iy, args.tile_halo)
        candidate_xy = data["xyz"][candidate_indices, :2]
        within_halo = np.all(
            candidate_xy >= (core_minimum - args.tile_halo)[None, :], axis=1
        ) & np.all(candidate_xy < (core_maximum + args.tile_halo)[None, :], axis=1)
        candidate_indices = candidate_indices[within_halo]

        if "background" in args.stages:
            jobs = []
            candidate_instance = data["instance_id"][candidate_indices]
            candidate_semantic = data["semantic_id"][candidate_indices]
            candidate_confidence = data.get("label_confidence")
            base_background = candidate_instance == 0
            if candidate_confidence is not None:
                base_background &= (
                    candidate_confidence[candidate_indices] >= args.minimum_label_confidence
                )

            candidate_group = None
            if background_groups_disjoint:
                candidate_group = np.full(len(candidate_semantic), -1, dtype=np.int16)
                valid_semantic = (
                    (candidate_semantic >= 0)
                    & (candidate_semantic < len(background_group_lut))
                )
                candidate_group[valid_semantic] = background_group_lut[
                    candidate_semantic[valid_semantic].astype(np.int64, copy=False)
                ]

            for group_index, group in enumerate(background_groups):
                if candidate_group is not None:
                    mask = base_background & (candidate_group == group_index)
                else:
                    mask = base_background & np.isin(
                        candidate_semantic,
                        np.asarray(group["semantic_ids"], np.int16),
                    )
                group_indices = candidate_indices[mask]
                rejected_boxes = 0
                if len(group_indices) and box_cache:
                    keep = reject_tracked_box_points(
                        data["xyz"][group_indices],
                        data["observation_frame_index"][group_indices],
                        box_cache,
                        args.box_xy_margin,
                        args.box_z_margin,
                    )
                    rejected_boxes = int(np.count_nonzero(~keep))
                    group_indices = group_indices[keep]
                tag = f"tile {tile_ix},{tile_iy} background {group['name']}"
                future = executor.submit(
                    reconstruct_from_indices, runner, data, group_indices, group, tag
                )
                jobs.append((group, group_indices, rejected_boxes, future))

            for group, group_indices, rejected_boxes, future in jobs:
                reconstructed = future.result()
                reconstructed = crop_surface_xy(reconstructed, core_minimum, core_maximum)
                tile_parts.append(reconstructed)
                background_records.append({
                    "tile": [tile_ix, tile_iy],
                    "group": group["name"],
                    "input_points": int(len(group_indices)),
                    "rejected_by_boxes": rejected_boxes,
                    "output_core_points": int(len(reconstructed["xyz"])),
                })

        for object_surface in static_objects_by_tile.get((tile_ix, tile_iy), ()):
            tile_parts.append(object_surface)

        tile_surface = concatenate_surfaces(tile_parts)
        if not len(tile_surface["xyz"]):
            continue
        relative_path = Path("static_tiles") / f"tile_{tile_ix}_{tile_iy}.npz"
        save_npz(args.output_root / relative_path, tile_surface, args.npz_compression)
        tile_min_xyz = np.min(tile_surface["xyz"], axis=0).astype(np.float64)
        tile_max_xyz = np.max(tile_surface["xyz"], axis=0).astype(np.float64)
        tiles.append(
            {
                "tile_index": [tile_ix, tile_iy],
                "file": str(relative_path),
                "core_min_xy": core_minimum.tolist(),
                "core_max_xy": core_maximum.tolist(),
                "min_xyz": tile_min_xyz.tolist(),
                "max_xyz": tile_max_xyz.tolist(),
                "point_count": int(len(tile_surface["xyz"])),
                "semantic_counts": semantic_counts(tile_surface["semantic_id"]),
            }
        )
        print(
            f"Tile {tile_number:03d}/{len(tile_index.occupied):03d} "
            f"({tile_ix},{tile_iy}): {len(tile_surface['xyz']):,} output points"
        )

    stage_timings["background_tiles_s"] = time.perf_counter() - stage_started

    dynamic_records = []
    stage_started = time.perf_counter()
    if "dynamic_objects" in args.stages:
        dynamic_records = reconstruct_dynamic_objects(
            args.dynamic_input_root,
            args.output_root,
            config,
            runner,
            args.minimum_label_confidence,
            executor,
            args.npz_compression,
        )
    stage_timings["dynamic_objects_s"] = time.perf_counter() - stage_started
    executor.shutdown(wait=True)

    untracked_foreground = np.isin(
        data["semantic_id"], np.asarray(sorted(FOREGROUND_IDS), np.int16)
    ) & (data["instance_id"] == 0)
    manifest = {
        "complete": True,
        "format_version": 1,
        "case": args.caseid,
        "method": "semantic_instance_aware_pcl_mls",
        "npz_compression": args.npz_compression,
        "pcl_threads": int(args.pcl_threads),
        "mls_workers": int(args.mls_workers),
        "attribute_workers": int(args.attribute_workers),
        "coordinate_frame": "Waymo world",
        "surface_arrays": list(SURFACE_ARRAYS),
        "config_fingerprint": fingerprint,
        "config": fingerprint_payload,
        "tile_size_m": args.tile_size,
        "tile_halo_m": args.tile_halo,
        "voxel_downsampling": {
            "required_disabled": bool(args.require_no_voxel_downsampling),
            "enabled_groups": [
                {"name": name, "voxel_size_m": voxel_size}
                for name, voxel_size in nonzero_voxel_groups
            ],
        },
        "tiles": tiles,
        "counts": {
            "static_source_points": int(len(data["xyz"])),
            "static_output_points": int(sum(tile["point_count"] for tile in tiles)),
            "static_tracked_object_instances": len(static_object_records),
            "dynamic_object_models": int(sum(record["output_points"] > 0 for record in dynamic_records)),
            "untracked_foreground_points_excluded": int(np.count_nonzero(untracked_foreground)),
            "mls_surfaces_skipped_no_finite_normals": len(runner.skipped_surfaces),
        },
    }
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
    report = {
        "case": args.caseid,
        "config_fingerprint": fingerprint,
        "elapsed_seconds": time.time() - started,
        "stage_timings_seconds": stage_timings,
        "pcl_call_timings": runner.call_timings,
        "pcl_call_timing_sums": {
            key: float(sum(record.get(key, 0.0) for record in runner.call_timings))
            for key in ("wall_s", "write_pcd_s", "pcl_s", "read_pcd_s", "attribute_transfer_s")
        },
        "background": background_records,
        "static_objects": static_object_records,
        "dynamic_objects": dynamic_records,
        "skipped_mls_surfaces": runner.skipped_surfaces,
        "manifest": str(manifest_path),
    }
    with (args.output_root / "reconstruction_report.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    if args.work_root is None:
        shutil.rmtree(work_root, ignore_errors=True)
    print("\nCPU stage timing:")
    for name, seconds in stage_timings.items():
        print(f"  {name:24s}: {seconds:10.2f} s")
    if runner.call_timings:
        print(f"  PCL calls               : {len(runner.call_timings):10d}")
        print(f"  summed PCL subprocess s : {sum(x.get('pcl_s', 0.0) for x in runner.call_timings):10.2f} s (parallel sum)")
        print(f"  summed PCD I/O s        : {sum(x.get('write_pcd_s', 0.0) + x.get('read_pcd_s', 0.0) for x in runner.call_timings):10.2f} s (parallel sum)")
        print(f"  summed attr transfer s  : {sum(x.get('attribute_transfer_s', 0.0) for x in runner.call_timings):10.2f} s (parallel sum)")
    print("\nReconstruction complete")
    print(f"Manifest: {manifest_path}")
    print(f"Static MLS points: {manifest['counts']['static_output_points']:,}")
    print(f"Dynamic MLS models: {manifest['counts']['dynamic_object_models']:,}")
    print(f"Empty MLS surfaces skipped: {len(runner.skipped_surfaces):,}")
    print(
        "Excluded untracked foreground points: "
        f"{manifest['counts']['untracked_foreground_points_excluded']:,}"
    )


if __name__ == "__main__":
    main()