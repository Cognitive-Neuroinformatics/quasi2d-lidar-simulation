from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rotation_matrix(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)

    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)],
        ],
        dtype=np.float64,
    )


def evaluate_gaussian(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    scale_x: float,
    scale_y: float,
    rotation_deg: float,
) -> np.ndarray:
    scale_x = max(scale_x, 1e-3)
    scale_y = max(scale_y, 1e-3)

    rotation = rotation_matrix(rotation_deg)
    scale = np.diag([scale_x, scale_y])

    covariance = (
        rotation
        @ scale
        @ scale.T
        @ rotation.T
    )

    covariance_inverse = np.linalg.inv(covariance)

    positions = np.stack(
        [x_grid, y_grid],
        axis=-1,
    )

    exponent = np.einsum(
        "...i,ij,...j->...",
        positions,
        covariance_inverse,
        positions,
    )

    return np.exp(-0.5 * exponent)


def save_gaussian_figure(
    output_path: Path,
    scale_x: float,
    scale_y: float,
    rotation_deg: float,
    intensity: float,
    opacity: float,
    ray_drop_probability: float,
) -> None:
    coordinates = np.linspace(-5.0, 5.0, 400)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)

    gaussian = evaluate_gaussian(
        x_grid=x_grid,
        y_grid=y_grid,
        scale_x=scale_x,
        scale_y=scale_y,
        rotation_deg=rotation_deg,
    )

    displayed_values = intensity * gaussian

    figure, axis = plt.subplots(figsize=(8, 8))

    image = axis.imshow(
        displayed_values,
        extent=[-5, 5, -5, 5],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=opacity,
    )

    axis.set_title(
        "Gaussian primitive\n"
        f"Scale=({scale_x:.2f}, {scale_y:.2f}), "
        f"Rotation={rotation_deg:.1f}°\n"
        f"Intensity={intensity:.2f}, "
        f"Opacity={opacity:.2f}, "
        f"Ray-drop={ray_drop_probability:.2f}"
    )

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")

    figure.colorbar(
        image,
        ax=axis,
        label="Intensity-weighted Gaussian value",
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


output_directory = Path("gaussian_figures")
output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

configurations = [
    {
        "filename": "01_spherical.png",
        "scale_x": 1.0,
        "scale_y": 1.0,
        "rotation_deg": 0.0,
        "intensity": 0.8,
        "opacity": 1.0,
        "ray_drop_probability": 0.1,
    },
    {
        "filename": "02_elongated_horizontal.png",
        "scale_x": 2.5,
        "scale_y": 0.4,
        "rotation_deg": 0.0,
        "intensity": 0.8,
        "opacity": 1.0,
        "ray_drop_probability": 0.1,
    },
    {
        "filename": "03_elongated_rotated.png",
        "scale_x": 2.5,
        "scale_y": 0.4,
        "rotation_deg": 45.0,
        "intensity": 0.8,
        "opacity": 1.0,
        "ray_drop_probability": 0.1,
    },
    {
        "filename": "04_low_intensity.png",
        "scale_x": 2.0,
        "scale_y": 0.6,
        "rotation_deg": 30.0,
        "intensity": 0.2,
        "opacity": 1.0,
        "ray_drop_probability": 0.1,
    },
    {
        "filename": "05_low_opacity.png",
        "scale_x": 2.0,
        "scale_y": 0.6,
        "rotation_deg": 30.0,
        "intensity": 0.8,
        "opacity": 0.2,
        "ray_drop_probability": 0.1,
    },
    {
        "filename": "06_high_ray_drop.png",
        "scale_x": 2.0,
        "scale_y": 0.6,
        "rotation_deg": 30.0,
        "intensity": 0.8,
        "opacity": 1.0,
        "ray_drop_probability": 0.9,
    },
]

for configuration in configurations:
    filename = configuration.pop("filename")

    save_gaussian_figure(
        output_path=output_directory / filename,
        **configuration,
    )