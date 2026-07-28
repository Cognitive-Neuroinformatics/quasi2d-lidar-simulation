import numpy as np
from plyfile import PlyData, PlyElement


input_path = "/home/samanti/git_repos/object_detection_dl/LiDAR-GS/output/waymo_seq1067/2026-07-28_14:04:47/point_cloud/iteration_4000/point_cloud.ply"
output_path = "gaussian_centres_filtered.ply"

ply = PlyData.read(input_path)
vertex = ply["vertex"].data

names = vertex.dtype.names
print("Available fields:", names)

xyz = np.column_stack([
    vertex["x"],
    vertex["y"],
    vertex["z"],
]).astype(np.float32)

valid = np.isfinite(xyz).all(axis=1)

# Gaussian Splatting often stores opacity before sigmoid.
if "opacity" in names:
    raw_opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-raw_opacity))

    # Start conservatively. You can later test 0.1, 0.3, 0.5, etc.
    valid &= opacity > 0.1

xyz = xyz[valid]

print("Original Gaussians:", len(vertex))
print("Retained Gaussians:", len(xyz))

vertices = np.empty(
    len(xyz),
    dtype=[
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ],
)

vertices["x"] = xyz[:, 0]
vertices["y"] = xyz[:, 1]
vertices["z"] = xyz[:, 2]

PlyData(
    [PlyElement.describe(vertices, "vertex")],
    text=False,
).write(output_path)

print("Saved:", output_path)