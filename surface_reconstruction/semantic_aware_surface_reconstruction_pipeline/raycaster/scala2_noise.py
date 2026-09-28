#!/usr/bin/env python3
"""Fast SCALA-2 post-hit measurement-noise models for raycast NPZ outputs."""

from __future__ import annotations

import zlib
import numpy as np

NOISE_MODELS = ("datasheet_gaussian", "incidence_secant")


def deterministic_noise_seed(base_seed: int, frame_index: int, sensor_name: str) -> int:
    sensor_key = zlib.crc32(sensor_name.encode("utf-8")) & 0xFFFFFFFF
    seq = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, int(frame_index) & 0xFFFFFFFF, sensor_key])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _incidence_terms(arrays: dict[str, np.ndarray], max_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= max_angle_deg < 90.0): raise ValueError("incidence_max_angle_deg must satisfy 0 <= angle < 90")
    if "normal_sensor" not in arrays or "ray_direction_sensor" not in arrays: raise KeyError("incidence_secant requires normal_sensor and ray_direction_sensor")
    normal = np.asarray(arrays["normal_sensor"], dtype=np.float64)
    direction = np.asarray(arrays["ray_direction_sensor"], dtype=np.float64)
    if normal.shape != direction.shape or normal.ndim != 2 or normal.shape[1] != 3: raise ValueError("normal_sensor and ray_direction_sensor must both have shape (N,3)")
    n_norm = np.linalg.norm(normal, axis=1); d_norm = np.linalg.norm(direction, axis=1)
    valid = (n_norm > 1e-12) & (d_norm > 1e-12)
    cosine = np.ones(len(normal), dtype=np.float64)
    cosine[valid] = np.abs(np.sum(normal[valid] * direction[valid], axis=1) / (n_norm[valid] * d_norm[valid]))
    cosine = np.clip(cosine, 0.0, 1.0)
    angle_deg = np.rad2deg(np.arccos(cosine))
    cosine_floor = float(np.cos(np.deg2rad(max_angle_deg)))
    multiplier = 1.0 / np.maximum(cosine, cosine_floor)
    return cosine, angle_deg, multiplier


def apply_scala2_measurement_noise(arrays: dict[str, np.ndarray], range_sigma_m: float = 0.05, azimuth_sigma_deg: float = 0.1, polar_sigma_deg: float = 0.6, base_seed: int = 12345, model: str = "datasheet_gaussian", incidence_max_angle_deg: float = 75.0) -> dict[str, np.ndarray]:
    """Apply deterministic post-hit SCALA-2 measurement noise in spherical coordinates.

    datasheet_gaussian: constant manufacturer-style sigmas.
    incidence_secant: same angular sigmas, but range sigma is multiplied by sec(beta),
    capped at incidence_max_angle_deg. Hit identity, semantics and provenance are unchanged.
    """
    if model not in NOISE_MODELS: raise ValueError(f"Unknown noise model {model!r}; choose from {NOISE_MODELS}")
    if range_sigma_m < 0 or azimuth_sigma_deg < 0 or polar_sigma_deg < 0: raise ValueError("Noise standard deviations must be non-negative")
    required = {"range_m", "horizontal_angle_deg", "vertical_angle_deg", "sensor_to_world", "ray_index"}
    missing = required.difference(arrays)
    if missing: raise KeyError(f"Noise model requires arrays: {sorted(missing)}")

    frame_index = int(np.asarray(arrays.get("output_frame_index", [0])).reshape(-1)[0])
    sensor_name = str(np.asarray(arrays.get("sensor_name", ["unknown"])).reshape(-1)[0])
    seed = deterministic_noise_seed(base_seed, frame_index, sensor_name)
    rng = np.random.default_rng(seed)

    nominal_range = np.asarray(arrays["range_m"], dtype=np.float64)
    nominal_az = np.asarray(arrays["horizontal_angle_deg"], dtype=np.float64)
    nominal_pol = np.asarray(arrays["vertical_angle_deg"], dtype=np.float64)
    n = len(nominal_range)
    z_range = rng.standard_normal(n) if range_sigma_m > 0 else np.zeros(n, dtype=np.float64)
    z_az = rng.standard_normal(n) if azimuth_sigma_deg > 0 else np.zeros(n, dtype=np.float64)
    z_pol = rng.standard_normal(n) if polar_sigma_deg > 0 else np.zeros(n, dtype=np.float64)

    incidence_cosine = np.ones(n, dtype=np.float64)
    incidence_angle_deg = np.zeros(n, dtype=np.float64)
    range_sigma_multiplier = np.ones(n, dtype=np.float64)
    if model == "incidence_secant": incidence_cosine, incidence_angle_deg, range_sigma_multiplier = _incidence_terms(arrays, incidence_max_angle_deg)

    effective_range_sigma = float(range_sigma_m) * range_sigma_multiplier
    dr = z_range * effective_range_sigma
    daz = z_az * float(azimuth_sigma_deg)
    dpol = z_pol * float(polar_sigma_deg)
    measured_range = np.maximum(nominal_range + dr, np.finfo(np.float32).eps)
    measured_az = nominal_az + daz
    measured_pol = nominal_pol + dpol
    az = np.deg2rad(measured_az); pol = np.deg2rad(measured_pol); cp = np.cos(pol)
    measured_dir = np.column_stack((cp * np.cos(az), cp * np.sin(az), np.sin(pol)))
    xyz_sensor = measured_dir * measured_range[:, None]

    sensor_to_world = np.asarray(arrays["sensor_to_world"], dtype=np.float64).reshape(4, 4)
    xyz_world = xyz_sensor @ sensor_to_world[:3, :3].T + sensor_to_world[:3, 3]

    noisy = dict(arrays)
    noisy["nominal_xyz_sensor"] = np.asarray(arrays.get("xyz_sensor", arrays.get("xyz")), dtype=np.float32)
    if "xyz_world" in arrays: noisy["nominal_xyz_world"] = np.asarray(arrays["xyz_world"], dtype=np.float32)
    noisy["nominal_range_m"] = nominal_range.astype(np.float32)
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
    noisy["noise_unit_gaussian_range"] = z_range.astype(np.float32)
    noisy["noise_unit_gaussian_azimuth"] = z_az.astype(np.float32)
    noisy["noise_unit_gaussian_polar"] = z_pol.astype(np.float32)
    noisy["incidence_cosine"] = incidence_cosine.astype(np.float32)
    noisy["incidence_angle_deg"] = incidence_angle_deg.astype(np.float32)
    noisy["noise_range_sigma_multiplier"] = range_sigma_multiplier.astype(np.float32)
    noisy["noise_effective_range_sigma_m"] = effective_range_sigma.astype(np.float32)

    if "range_image" in noisy:
        range_image = np.asarray(noisy["range_image"]).copy().reshape(-1)
        range_image[np.asarray(noisy["ray_index"], dtype=np.int64)] = measured_range.astype(range_image.dtype, copy=False)
        noisy["range_image"] = range_image.reshape(np.asarray(arrays["range_image"]).shape)

    noisy["noise_model"] = np.asarray([model])
    noisy["noise_range_sigma_m"] = np.asarray([range_sigma_m], dtype=np.float32)
    noisy["noise_azimuth_sigma_deg"] = np.asarray([azimuth_sigma_deg], dtype=np.float32)
    noisy["noise_polar_sigma_deg"] = np.asarray([polar_sigma_deg], dtype=np.float32)
    noisy["noise_incidence_max_angle_deg"] = np.asarray([incidence_max_angle_deg], dtype=np.float32)
    noisy["noise_base_seed"] = np.asarray([base_seed], dtype=np.int64)
    noisy["noise_realization_seed"] = np.asarray([seed], dtype=np.uint32)
    noisy["noise_scope"] = np.asarray(["post_hit_measurement_only; hit identity/semantics/provenance unchanged"])
    return noisy
