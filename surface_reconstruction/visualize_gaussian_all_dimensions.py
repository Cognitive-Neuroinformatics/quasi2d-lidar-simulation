"""
Visualize and save 1D, 2D, and 3D Gaussian functions.

Outputs:
    gaussian_visualizations/
        01_1d_narrow.png
        02_1d_wide.png
        03_2d_isotropic.png
        04_2d_anisotropic.png
        05_2d_anisotropic_rotated.png
        06_3d_isotropic.png
        07_3d_anisotropic.png
        08_3d_anisotropic_rotated.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("gaussian_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gaussian_1d(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_xyz(
    angle_x_deg: float,
    angle_y_deg: float,
    angle_z_deg: float,
) -> np.ndarray:
    ax = np.deg2rad(angle_x_deg)
    ay = np.deg2rad(angle_y_deg)
    az = np.deg2rad(angle_z_deg)

    rx = np.array(
        [[1, 0, 0],
         [0, np.cos(ax), -np.sin(ax)],
         [0, np.sin(ax), np.cos(ax)]],
        dtype=np.float64,
    )

    ry = np.array(
        [[np.cos(ay), 0, np.sin(ay)],
         [0, 1, 0],
         [-np.sin(ay), 0, np.cos(ay)]],
        dtype=np.float64,
    )

    rz = np.array(
        [[np.cos(az), -np.sin(az), 0],
         [np.sin(az), np.cos(az), 0],
         [0, 0, 1]],
        dtype=np.float64,
    )

    return rz @ ry @ rx


def covariance_from_scale_rotation(
    scales: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    if np.any(scales <= 0):
        raise ValueError("All Gaussian scales must be positive.")
    scale_matrix = np.diag(scales)
    return rotation @ scale_matrix @ scale_matrix.T @ rotation.T


def gaussian_nd(
    points: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    covariance_inverse = np.linalg.inv(covariance)
    difference = points - mean
    squared_distance = np.einsum(
        "...i,ij,...j->...",
        difference,
        covariance_inverse,
        difference,
    )
    return np.exp(-0.5 * squared_distance)


def save_1d_gaussian(filename: str, sigma: float, title: str) -> None:
    x = np.linspace(-5.0, 5.0, 1000)
    values = gaussian_1d(x, mean=0.0, sigma=sigma)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, values)
    axis.fill_between(x, values, alpha=0.25)
    axis.axvline(0.0, linestyle="--", linewidth=1.0, label="Mean")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("Gaussian value")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_2d_gaussian(
    filename: str,
    scales: tuple[float, float],
    rotation_deg: float,
    title: str,
) -> None:
    coordinates = np.linspace(-5.0, 5.0, 500)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack([x_grid, y_grid], axis=-1)

    rotation = rotation_matrix_2d(rotation_deg)
    covariance = covariance_from_scale_rotation(
        np.asarray(scales, dtype=np.float64),
        rotation,
    )

    values = gaussian_nd(points, np.zeros(2), covariance)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        values,
        extent=[-5.0, 5.0, -5.0, 5.0],
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axis.contour(
        x_grid,
        y_grid,
        values,
        levels=[0.1, 0.25, 0.5, 0.75],
        linewidths=1.0,
    )
    axis.scatter(0.0, 0.0, s=30, label="Mean")
    axis.set_title(
        f"{title}\nScales={scales}, rotation={rotation_deg:.1f}°"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal")
    axis.legend()
    figure.colorbar(image, ax=axis, label="Gaussian value")

    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_ellipsoid_surface(
    scales: np.ndarray,
    rotation: np.ndarray,
    radius_factor: float = 2.0,
    resolution: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.linspace(0.0, 2.0 * np.pi, resolution)
    polar = np.linspace(0.0, np.pi, resolution)

    unit_x = np.outer(np.cos(azimuth), np.sin(polar))
    unit_y = np.outer(np.sin(azimuth), np.sin(polar))
    unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))

    unit_sphere = np.stack(
        [unit_x.ravel(), unit_y.ravel(), unit_z.ravel()],
        axis=0,
    )

    scaled = radius_factor * np.diag(scales) @ unit_sphere
    transformed = rotation @ scaled

    return (
        transformed[0].reshape(unit_x.shape),
        transformed[1].reshape(unit_y.shape),
        transformed[2].reshape(unit_z.shape),
    )


def set_equal_3d_limits(axis, x, y, z) -> None:
    minimum = np.array([x.min(), y.min(), z.min()])
    maximum = np.array([x.max(), y.max(), z.max()])
    centre = 0.5 * (minimum + maximum)
    radius = 0.55 * np.max(maximum - minimum)

    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)


def save_3d_gaussian(
    filename: str,
    scales: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    title: str,
) -> None:
    rotation = rotation_matrix_xyz(*rotation_xyz_deg)
    scale_array = np.asarray(scales, dtype=np.float64)

    x, y, z = create_ellipsoid_surface(
        scales=scale_array,
        rotation=rotation,
        radius_factor=2.0,
    )

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")

    axis.plot_surface(
        x,
        y,
        z,
        alpha=0.55,
        linewidth=0.15,
        antialiased=True,
        shade=True,
    )

    axis.scatter(0.0, 0.0, 0.0, s=40, label="Mean")

    names = ["Principal axis 1", "Principal axis 2", "Principal axis 3"]
    for index in range(3):
        direction = rotation[:, index]
        endpoint = 2.0 * scale_array[index] * direction
        axis.plot(
            [0.0, endpoint[0]],
            [0.0, endpoint[1]],
            [0.0, endpoint[2]],
            linewidth=2.0,
            label=names[index],
        )

    set_equal_3d_limits(axis, x, y, z)

    axis.set_title(
        f"{title}\nScales={scales}, rotation XYZ={rotation_xyz_deg}°"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.legend(loc="upper left")

    figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    save_1d_gaussian(
        "01_1d_narrow.png",
        sigma=0.6,
        title="1D Gaussian: narrow spread",
    )
    save_1d_gaussian(
        "02_1d_wide.png",
        sigma=1.8,
        title="1D Gaussian: wide spread",
    )

    save_2d_gaussian(
        "03_2d_isotropic.png",
        scales=(1.2, 1.2),
        rotation_deg=0.0,
        title="2D isotropic Gaussian",
    )
    save_2d_gaussian(
        "04_2d_anisotropic.png",
        scales=(2.2, 0.6),
        rotation_deg=0.0,
        title="2D anisotropic Gaussian",
    )
    save_2d_gaussian(
        "05_2d_anisotropic_rotated.png",
        scales=(2.2, 0.6),
        rotation_deg=40.0,
        title="2D rotated anisotropic Gaussian",
    )

    save_3d_gaussian(
        "06_3d_isotropic.png",
        scales=(1.0, 1.0, 1.0),
        rotation_xyz_deg=(0.0, 0.0, 0.0),
        title="3D isotropic Gaussian",
    )
    save_3d_gaussian(
        "07_3d_anisotropic.png",
        scales=(2.0, 1.0, 0.35),
        rotation_xyz_deg=(0.0, 0.0, 0.0),
        title="3D anisotropic Gaussian",
    )
    save_3d_gaussian(
        "08_3d_anisotropic_rotated.png",
        scales=(2.0, 1.0, 0.35),
        rotation_xyz_deg=(20.0, 30.0, 40.0),
        title="3D rotated anisotropic Gaussian",
    )

    print(f"Saved all figures to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()