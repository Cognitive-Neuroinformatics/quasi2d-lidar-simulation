#!/usr/bin/env python3
"""Fast SCALA2 measurement-space noise model for raycast NPZ outputs."""

from __future__ import annotations

import zlib
import numpy as np


def deterministic_noise_seed(base_seed: int, frame_index: int, sensor_name: str) -> int:
    sensor_key = zlib.crc32(sensor_name.encode("utf-8")) & 0xFFFFFFFF
    seq = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, int(frame_index) & 0xFFFFFFFF, sensor_key])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def apply_scala2_measurement_noise(arrays: dict[str, np.ndarray], range_sigma_m: float = 0.05, azimuth_sigma_deg: float = 0.1, polar_sigma_deg: float = 0.6, base_seed: int = 12345) -> dict[str, np.ndarray]:
    """Return a shallow-copied NPZ payload with independent Gaussian noise in spherical measurement coordinates.

    This is a post-hit measurement model: raycast visibility, hit identity, semantics, normals and provenance remain unchanged.
    Only measured range/angles and derived XYZ coordinates are perturbed.
    """
    if range_sigma_m < 0 or azimuth_sigma_deg < 0 or polar_sigma_deg < 0:
        raise ValueError("Noise standard deviations must be non-negative")
    required = {"range_m", "horizontal_angle_deg", "vertical_angle_deg", "sensor_to_world", "ray_index"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"Noise model requires arrays: {sorted(missing)}")

    frame_index = int(np.asarray(arrays.get("output_frame_index", [0])).reshape(-1)[0])
    sensor_name = str(np.asarray(arrays.get("sensor_name", ["unknown"])).reshape(-1)[0])
    seed = deterministic_noise_seed(base_seed, frame_index, sensor_name)
    rng = np.random.default_rng(seed)

    nominal_range = np.asarray(arrays["range_m"], dtype=np.float64)
    nominal_az = np.asarray(arrays["horizontal_angle_deg"], dtype=np.float64)
    nominal_pol = np.asarray(arrays["vertical_angle_deg"], dtype=np.float64)
    n = len(nominal_range)
    dr = rng.normal(0.0, range_sigma_m, n) if range_sigma_m > 0 else np.zeros(n, dtype=np.float64)
    daz = rng.normal(0.0, azimuth_sigma_deg, n) if azimuth_sigma_deg > 0 else np.zeros(n, dtype=np.float64)
    dpol = rng.normal(0.0, polar_sigma_deg, n) if polar_sigma_deg > 0 else np.zeros(n, dtype=np.float64)

    measured_range = nominal_range + dr
    measured_range = np.maximum(measured_range, np.finfo(np.float32).eps)
    measured_az = nominal_az + daz
    measured_pol = nominal_pol + dpol
    az = np.deg2rad(measured_az)
    pol = np.deg2rad(measured_pol)
    cp = np.cos(pol)
    measured_dir = np.column_stack((cp * np.cos(az), cp * np.sin(az), np.sin(pol)))
    xyz_sensor = measured_dir * measured_range[:, None]

    sensor_to_world = np.asarray(arrays["sensor_to_world"], dtype=np.float64).reshape(4, 4)
    xyz_world = xyz_sensor @ sensor_to_world[:3, :3].T + sensor_to_world[:3, 3]

    noisy = dict(arrays)
    noisy["nominal_ray_direction_sensor"] = np.asarray(arrays.get("ray_direction_sensor", measured_dir), dtype=np.float32)
    noisy["nominal_horizontal_angle_deg"] = nominal_az.astype(np.float32)
    noisy["nominal_vertical_angle_deg"] = nominal_pol.astype(np.float32)
    noisy["xyz"] = xyz_sensor.astype(np.float32)
    noisy["xyz_sensor"] = xyz_sensor.astype(np.float32)
    noisy["xyz_world"] = xyz_world.astype(np.float32)
    noisy["range"] = measured_range.astype(np.float32)
    noisy["range_m"] = measured_range.astype(np.float32)
    noisy["ray_direction_sensor"] = measured_dir.astype(np.float32)
    noisy["horizontal_angle_deg"] = measured_az.astype(np.float32)
    noisy["vertical_angle_deg"] = measured_pol.astype(np.float32)
    noisy["measured_horizontal_angle_deg"] = measured_az.astype(np.float32)
    noisy["measured_vertical_angle_deg"] = measured_pol.astype(np.float32)
    noisy["noise_delta_range_m"] = dr.astype(np.float32)
    noisy["noise_delta_azimuth_deg"] = daz.astype(np.float32)
    noisy["noise_delta_polar_deg"] = dpol.astype(np.float32)

    if "range_image" in noisy:
        range_image = np.asarray(noisy["range_image"]).copy().reshape(-1)
        range_image[np.asarray(noisy["ray_index"], dtype=np.int64)] = measured_range.astype(range_image.dtype, copy=False)
        noisy["range_image"] = range_image.reshape(np.asarray(arrays["range_image"]).shape)

    noisy["noise_model"] = np.asarray(["independent_gaussian_spherical_post_hit_v1"])
    noisy["noise_range_sigma_m"] = np.asarray([range_sigma_m], dtype=np.float32)
    noisy["noise_azimuth_sigma_deg"] = np.asarray([azimuth_sigma_deg], dtype=np.float32)
    noisy["noise_polar_sigma_deg"] = np.asarray([polar_sigma_deg], dtype=np.float32)
    noisy["noise_base_seed"] = np.asarray([base_seed], dtype=np.int64)
    noisy["noise_realization_seed"] = np.asarray([seed], dtype=np.uint32)
    noisy["noise_scope"] = np.asarray(["post_hit_measurement_only; hit identity/semantics/provenance unchanged"])
    return noisy
