"""
Covariance and Ellipsoidal Gaussian Visualizer

This script creates a set of figures that explain:

1. Variance along x and y
2. Positive, zero, and negative covariance
3. How eigenvectors determine ellipse orientation
4. How eigenvalues determine ellipse axis lengths
5. How scale and rotation create a covariance matrix
6. The same idea in 3D as an ellipsoidal Gaussian

Run:
    python covariance_gaussian_visualizer.py

Output folder:
    covariance_visualizations/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


OUTPUT_DIR = Path("covariance_visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 7


def gaussian_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate an unnormalized 2D Gaussian."""
    points = np.stack([x_grid, y_grid], axis=-1)
    difference = points - mean
    covariance_inverse = np.linalg.inv(covariance)

    squared_mahalanobis = np.einsum(
        "...i,ij,...j->...",
        difference,
        covariance_inverse,
        difference,
    )

    return np.exp(-0.5 * squared_mahalanobis)


def rotation_matrix_2d(angle_deg: float) -> np.ndarray:
    """Return a 2D counterclockwise rotation matrix."""
    angle_rad = np.deg2rad(angle_deg)

    return np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=float,
    )


def covariance_from_scales_and_rotation(
    scale_x: float,
    scale_y: float,
    angle_deg: float,
) -> np.ndarray:
    """
    Construct covariance using:

        Sigma = R S S^T R^T

    where S contains standard deviations.
    """
    rotation = rotation_matrix_2d(angle_deg)
    scale = np.diag([scale_x, scale_y])

    return rotation @ scale @ scale.T @ rotation.T


def covariance_ellipse_parameters(
    covariance: np.ndarray,
    standard_deviations: float = 2.0,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Convert a covariance matrix into ellipse width, height and angle.

    Eigenvalues determine squared axis lengths.
    Eigenvectors determine axis directions.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    width = 2.0 * standard_deviations * np.sqrt(eigenvalues[0])
    height = 2.0 * standard_deviations * np.sqrt(eigenvalues[1])

    main_axis = eigenvectors[:, 0]
    angle_deg = np.rad2deg(
        np.arctan2(main_axis[1], main_axis[0])
    )

    return width, height, angle_deg, eigenvalues, eigenvectors


def add_covariance_ellipse(
    axis: plt.Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    standard_deviations: float = 2.0,
) -> None:
    """Draw a covariance ellipse and its principal axes."""
    width, height, angle_deg, eigenvalues, eigenvectors = (
        covariance_ellipse_parameters(
            covariance,
            standard_deviations,
        )
    )

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle_deg,
        fill=False,
        linewidth=2.5,
    )
    axis.add_patch(ellipse)

    for index in range(2):
        axis_length = (
            standard_deviations
            * np.sqrt(eigenvalues[index])
        )
        direction = eigenvectors[:, index]

        start = mean - axis_length * direction
        end = mean + axis_length * direction

        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linewidth=2.0,
        )


def save_covariance_case(
    filename: str,
    covariance: np.ndarray,
    title: str,
    sample_count: int = 1000,
) -> None:
    """
    Save a three-part visualization:
    samples, Gaussian heatmap, and covariance matrix.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    mean = np.zeros(2)

    samples = rng.multivariate_normal(
        mean=mean,
        cov=covariance,
        size=sample_count,
    )

    coordinates = np.linspace(-5.0, 5.0, 450)
    x_grid, y_grid = np.meshgrid(
        coordinates,
        coordinates,
    )

    values = gaussian_2d(
        x_grid=x_grid,
        y_grid=y_grid,
        mean=mean,
        covariance=covariance,
    )

    width, height, angle_deg, eigenvalues, eigenvectors = (
        covariance_ellipse_parameters(covariance)
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
    )

    # --------------------------------------------------
    # Panel 1: random samples
    # --------------------------------------------------
    axes[0].scatter(
        samples[:, 0],
        samples[:, 1],
        s=8,
        alpha=0.35,
    )
    add_covariance_ellipse(
        axes[0],
        mean,
        covariance,
    )

    axes[0].set_title("Samples from the Gaussian")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.25)

    # --------------------------------------------------
    # Panel 2: continuous Gaussian influence
    # --------------------------------------------------
    image = axes[1].imshow(
        values,
        extent=[-5, 5, -5, 5],
        origin="lower",
        vmin=0,
        vmax=1,
    )

    axes[1].contour(
        x_grid,
        y_grid,
        values,
        levels=[0.1, 0.25, 0.5, 0.75],
        linewidths=1.0,
    )

    add_covariance_ellipse(
        axes[1],
        mean,
        covariance,
    )

    axes[1].set_title("Continuous Gaussian influence")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    figure.colorbar(
        image,
        ax=axes[1],
        label="Gaussian value",
        fraction=0.046,
    )

    # --------------------------------------------------
    # Panel 3: numerical interpretation
    # --------------------------------------------------
    axes[2].axis("off")

    matrix_text = (
        "Covariance matrix\n\n"
        f"Σ = [[{covariance[0, 0]:.3f}, "
        f"{covariance[0, 1]:.3f}],\n"
        f"     [{covariance[1, 0]:.3f}, "
        f"{covariance[1, 1]:.3f}]]\n\n"
        "Interpretation\n\n"
        f"Variance in x: {covariance[0, 0]:.3f}\n"
        f"Variance in y: {covariance[1, 1]:.3f}\n"
        f"Covariance xy: {covariance[0, 1]:.3f}\n\n"
        f"Eigenvalues: {eigenvalues[0]:.3f}, "
        f"{eigenvalues[1]:.3f}\n"
        f"Principal standard deviations: "
        f"{np.sqrt(eigenvalues[0]):.3f}, "
        f"{np.sqrt(eigenvalues[1]):.3f}\n"
        f"Main-axis angle: {angle_deg:.1f}°\n\n"
        "Eigenvectors:\n"
        f"v₁ = [{eigenvectors[0, 0]:.3f}, "
        f"{eigenvectors[1, 0]:.3f}]\n"
        f"v₂ = [{eigenvectors[0, 1]:.3f}, "
        f"{eigenvectors[1, 1]:.3f}]"
    )

    axes[2].text(
        0.03,
        0.97,
        matrix_text,
        va="top",
        family="monospace",
        fontsize=11,
    )

    figure.suptitle(
        title,
        fontsize=15,
    )

    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_scale_rotation_decomposition() -> None:
    """
    Show how S, R and covariance produce an ellipsoid.
    """
    scale_x = 2.2
    scale_y = 0.6
    rotation_deg = 40.0

    scale_matrix = np.diag(
        [scale_x, scale_y]
    )
    rotation = rotation_matrix_2d(rotation_deg)
    covariance = (
        rotation
        @ scale_matrix
        @ scale_matrix.T
        @ rotation.T
    )

    mean = np.zeros(2)
    coordinates = np.linspace(-5, 5, 450)
    x_grid, y_grid = np.meshgrid(
        coordinates,
        coordinates,
    )

    unrotated_covariance = (
        scale_matrix
        @ scale_matrix.T
    )

    unrotated_values = gaussian_2d(
        x_grid,
        y_grid,
        mean,
        unrotated_covariance,
    )

    rotated_values = gaussian_2d(
        x_grid,
        y_grid,
        mean,
        covariance,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
    )

    axes[0].imshow(
        unrotated_values,
        extent=[-5, 5, -5, 5],
        origin="lower",
        vmin=0,
        vmax=1,
    )
    add_covariance_ellipse(
        axes[0],
        mean,
        unrotated_covariance,
    )
    axes[0].set_title(
        "Step 1: Scale only\n"
        "S = diag(2.2, 0.6)"
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].imshow(
        rotated_values,
        extent=[-5, 5, -5, 5],
        origin="lower",
        vmin=0,
        vmax=1,
    )
    add_covariance_ellipse(
        axes[1],
        mean,
        covariance,
    )
    axes[1].set_title(
        "Step 2: Apply rotation\n"
        "R = rotation(40°)"
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.95,
        "Final covariance\n\n"
        "Σ = R S Sᵀ Rᵀ\n\n"
        f"S =\n{np.array2string(scale_matrix, precision=3)}\n\n"
        f"R =\n{np.array2string(rotation, precision=3)}\n\n"
        f"Σ =\n{np.array2string(covariance, precision=3)}\n\n"
        "The diagonal scale matrix creates the\n"
        "axis lengths. Rotation changes the\n"
        "principal directions. Covariance stores\n"
        "both effects in one matrix.",
        va="top",
        family="monospace",
        fontsize=11,
    )

    figure.suptitle(
        "How scale and rotation create covariance",
        fontsize=15,
    )

    figure.savefig(
        OUTPUT_DIR / "05_scale_rotation_covariance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_3d_ellipsoid(
    scales: np.ndarray,
    rotation: np.ndarray,
    standard_deviations: float = 2.0,
    resolution: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a rotated 3D ellipsoid surface."""
    azimuth = np.linspace(
        0,
        2 * np.pi,
        resolution,
    )
    polar = np.linspace(
        0,
        np.pi,
        resolution,
    )

    x = np.outer(
        np.cos(azimuth),
        np.sin(polar),
    )
    y = np.outer(
        np.sin(azimuth),
        np.sin(polar),
    )
    z = np.outer(
        np.ones_like(azimuth),
        np.cos(polar),
    )

    unit_sphere = np.stack(
        [x.ravel(), y.ravel(), z.ravel()],
        axis=0,
    )

    transformed = (
        rotation
        @ (
            standard_deviations
            * np.diag(scales)
            @ unit_sphere
        )
    )

    return (
        transformed[0].reshape(x.shape),
        transformed[1].reshape(y.shape),
        transformed[2].reshape(z.shape),
    )


def rotation_matrix_3d(
    angle_x_deg: float,
    angle_y_deg: float,
    angle_z_deg: float,
) -> np.ndarray:
    """Construct a 3D rotation matrix."""
    ax, ay, az = np.deg2rad(
        [angle_x_deg, angle_y_deg, angle_z_deg]
    )

    rotation_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(ax), -np.sin(ax)],
            [0, np.sin(ax), np.cos(ax)],
        ],
        dtype=float,
    )

    rotation_y = np.array(
        [
            [np.cos(ay), 0, np.sin(ay)],
            [0, 1, 0],
            [-np.sin(ay), 0, np.cos(ay)],
        ],
        dtype=float,
    )

    rotation_z = np.array(
        [
            [np.cos(az), -np.sin(az), 0],
            [np.sin(az), np.cos(az), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )

    return rotation_z @ rotation_y @ rotation_x


def save_3d_covariance_visualization() -> None:
    """Save isotropic and anisotropic 3D covariance ellipsoids."""
    cases = [
        (
            "3D isotropic covariance",
            np.array([1.0, 1.0, 1.0]),
            rotation_matrix_3d(0, 0, 0),
        ),
        (
            "3D anisotropic covariance",
            np.array([2.0, 1.0, 0.35]),
            rotation_matrix_3d(20, 30, 40),
        ),
    ]

    for index, (
        title,
        scales,
        rotation,
    ) in enumerate(cases, start=1):
        covariance = (
            rotation
            @ np.diag(scales)
            @ np.diag(scales)
            @ rotation.T
        )

        x, y, z = create_3d_ellipsoid(
            scales=scales,
            rotation=rotation,
        )

        figure = plt.figure(figsize=(9, 8))
        axis = figure.add_subplot(
            111,
            projection="3d",
        )

        axis.plot_surface(
            x,
            y,
            z,
            alpha=0.55,
            linewidth=0.15,
            antialiased=True,
        )

        axis.scatter(
            0,
            0,
            0,
            s=40,
            label="Mean",
        )

        for axis_index in range(3):
            direction = rotation[:, axis_index]
            endpoint = (
                2.0
                * scales[axis_index]
                * direction
            )
            axis.plot(
                [0, endpoint[0]],
                [0, endpoint[1]],
                [0, endpoint[2]],
                linewidth=2.0,
                label=f"Principal axis {axis_index + 1}",
            )

        maximum_extent = (
            2.4 * np.max(scales)
        )
        axis.set_xlim(
            -maximum_extent,
            maximum_extent,
        )
        axis.set_ylim(
            -maximum_extent,
            maximum_extent,
        )
        axis.set_zlim(
            -maximum_extent,
            maximum_extent,
        )

        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(
            f"{title}\n"
            f"Scales={tuple(scales)}\n"
            f"Covariance=\n"
            f"{np.array2string(covariance, precision=2)}"
        )
        axis.legend(loc="upper left")

        figure.savefig(
            OUTPUT_DIR
            / f"{index + 5:02d}_{title.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def main() -> None:
    # 1. Isotropic: equal spread in every direction
    save_covariance_case(
        filename="01_isotropic_covariance.png",
        covariance=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        title="Isotropic covariance: equal spread in x and y",
    )

    # 2. Anisotropic but axis-aligned
    save_covariance_case(
        filename="02_axis_aligned_anisotropic.png",
        covariance=np.array(
            [
                [4.0, 0.0],
                [0.0, 0.36],
            ]
        ),
        title=(
            "Anisotropic covariance with zero off-diagonal terms: "
            "elongated but not rotated"
        ),
    )

    # 3. Positive covariance
    save_covariance_case(
        filename="03_positive_covariance.png",
        covariance=np.array(
            [
                [2.0, 1.4],
                [1.4, 2.0],
            ]
        ),
        title=(
            "Positive covariance: x and y tend to increase together"
        ),
    )

    # 4. Negative covariance
    save_covariance_case(
        filename="04_negative_covariance.png",
        covariance=np.array(
            [
                [2.0, -1.4],
                [-1.4, 2.0],
            ]
        ),
        title=(
            "Negative covariance: x increases while y tends to decrease"
        ),
    )

    # 5. Explicit decomposition
    save_scale_rotation_decomposition()

    # 6 and 7. 3D ellipsoids
    save_3d_covariance_visualization()

    print(
        f"Saved covariance visualizations to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()