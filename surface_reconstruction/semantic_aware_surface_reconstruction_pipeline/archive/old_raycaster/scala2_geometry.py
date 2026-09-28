#!/usr/bin/env python3
"""NumPy implementation of the validated SCALA2 ray geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


AZ_MIN_DEG = -66.5
AZ_MAX_DEG = 66.5
CENTER_MIN_DEG = -15.0
CENTER_MAX_DEG = 15.0
OUTER_STEP_DEG = 0.25
CENTER_STEP_DEG = 0.125
APD_AZIMUTH_OFFSET_DEG = 0.0181
SCALA2_HEIGHT = 16
SCALA2_WIDTH = 653


SCALA2_SENSOR_EXTRINSICS = {
    "front_left": {
        "translation": [3.85, 0.8, 0.743],
        "quaternion": [0.005826, -0.016451, 0.333756, 0.942498],
    },
    "front_center": {
        "translation": [4.105, 0.0, 0.625],
        "quaternion": [0.0, 0.006981, 0.0, 0.999976],
    },
    "front_right": {
        "translation": [3.86, -0.81, 0.74],
        "quaternion": [0.0, 0.0, -0.358368, 0.933580],
    },
    "rear_right": {
        "translation": [-0.68, -0.982, 0.741],
        "quaternion": [0.0, 0.0, -0.7325429, 0.6807209],
    },
    "rear_center": {
        "translation": [-1.065, 0.0, 0.35],
        "quaternion": [-0.01745, 0.00026, 0.9998, 0.00011],
    },
    "rear_left": {
        "translation": [-0.675, 0.985, 0.74],
        "quaternion": [0.0, 0.0, 0.7071068, 0.7071068],
    },
}


def get_scala2_to_vehicle(sensor_name: str) -> np.ndarray:
    """Return the SCALA2-sensor-to-Waymo-vehicle transform."""
    if sensor_name not in SCALA2_SENSOR_EXTRINSICS:
        available = ", ".join(SCALA2_SENSOR_EXTRINSICS)
        raise ValueError(
            f"Unknown SCALA2 sensor {sensor_name!r}; available: {available}"
        )

    calibration = SCALA2_SENSOR_EXTRINSICS[sensor_name]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        np.asarray(calibration["quaternion"], dtype=np.float64)
    ).as_matrix()
    transform[:3, 3] = np.asarray(
        calibration["translation"], dtype=np.float64
    )
    return transform


def scala2_world_pose(
    vehicle_to_world: np.ndarray,
    sensor_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sensor_to_world, world_to_sensor)`` for one frame."""
    vehicle_to_world = np.asarray(vehicle_to_world, dtype=np.float64).reshape(4, 4)
    sensor_to_world = vehicle_to_world @ get_scala2_to_vehicle(sensor_name)
    return sensor_to_world, np.linalg.inv(sensor_to_world)


def scala2_polar_angle_deg(
    phi_deg: float,
    mirror_side: int,
    layer: int,
    apd_group: int,
) -> float:
    """Evaluate the SCALA2 manual polar-angle model."""
    mirror_term = (
        1.512e-8 * phi_deg**3
        - 5.152e-6 * phi_deg**2
        - 1.233e-3 * phi_deg
        + 0.1412
    )
    return float(
        ((-1) ** mirror_side) * mirror_term
        + 0.6025 * layer
        + 2.564 * apd_group
        - 4.749
    )


def generate_scala2_base_azimuths() -> np.ndarray:
    """Return the validated 653 SCALA2 base-column azimuths."""
    left = np.arange(
        AZ_MIN_DEG, CENTER_MIN_DEG, OUTER_STEP_DEG, dtype=np.float64
    )
    center = np.arange(
        CENTER_MIN_DEG, CENTER_MAX_DEG, CENTER_STEP_DEG, dtype=np.float64
    )
    right = np.arange(
        CENTER_MAX_DEG, AZ_MAX_DEG + 1e-9, OUTER_STEP_DEG, dtype=np.float64
    )
    result = np.concatenate([left, center, right])
    if result.shape != (SCALA2_WIDTH,):
        raise RuntimeError(
            f"Generated {len(result)} SCALA2 columns; expected {SCALA2_WIDTH}"
        )
    return result


def generate_scala2_ray_geometry(mirror_side: int) -> dict[str, np.ndarray]:
    """Generate all rays and aligned metadata for one physical scan.

    Arrays use the renderer's ``[row, column] == [16, 653]`` layout.  Rows
    are ordered by APD group first and layer second: ``row = 4*APD + layer``.
    """
    if mirror_side not in (0, 1):
        raise ValueError("mirror_side must be 0 or 1")

    base_azimuths = generate_scala2_base_azimuths()
    directions = np.empty((SCALA2_HEIGHT, SCALA2_WIDTH, 3), dtype=np.float64)
    horizontal = np.empty((SCALA2_HEIGHT, SCALA2_WIDTH), dtype=np.float64)
    vertical = np.empty_like(horizontal)
    apd = np.empty((SCALA2_HEIGHT, SCALA2_WIDTH), dtype=np.int8)
    layer_grid = np.empty_like(apd)

    for column, base_phi in enumerate(base_azimuths):
        for apd_group in range(4):
            phi_deg = float(base_phi + apd_group * APD_AZIMUTH_OFFSET_DEG)
            phi = np.deg2rad(phi_deg)
            for layer in range(4):
                row = 4 * apd_group + layer
                theta_deg = scala2_polar_angle_deg(
                    phi_deg, mirror_side, layer, apd_group
                )
                theta = np.deg2rad(theta_deg)
                directions[row, column] = (
                    np.cos(theta) * np.cos(phi),
                    np.cos(theta) * np.sin(phi),
                    np.sin(theta),
                )
                horizontal[row, column] = phi_deg
                vertical[row, column] = theta_deg
                apd[row, column] = apd_group
                layer_grid[row, column] = layer

    ray_row = np.broadcast_to(
        np.arange(SCALA2_HEIGHT, dtype=np.int16)[:, None],
        (SCALA2_HEIGHT, SCALA2_WIDTH),
    ).copy()
    ray_column = np.broadcast_to(
        np.arange(SCALA2_WIDTH, dtype=np.int16)[None, :],
        (SCALA2_HEIGHT, SCALA2_WIDTH),
    ).copy()
    mirror = np.full(
        (SCALA2_HEIGHT, SCALA2_WIDTH), mirror_side, dtype=np.int8
    )

    return {
        "directions": directions,
        "horizontal_angle_deg": horizontal,
        "vertical_angle_deg": vertical,
        "ray_row": ray_row,
        "ray_column": ray_column,
        "apd_group": apd,
        "layer": layer_grid,
        "mirror_side": mirror,
    }


def mirror_side_for_frame(frame_index: int, first_mirror_side: int = 0) -> int:
    """Return the explicit alternating mirror state for an output frame."""
    if first_mirror_side not in (0, 1):
        raise ValueError("first_mirror_side must be 0 or 1")
    return int((first_mirror_side + int(frame_index)) % 2)
