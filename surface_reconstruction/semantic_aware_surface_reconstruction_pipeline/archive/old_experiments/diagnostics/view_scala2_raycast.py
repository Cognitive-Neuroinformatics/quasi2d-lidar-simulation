#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d

SEMANTIC_COLORS = {
    0:[0.5,0.5,0.5], 1:[0.0,0.4,1.0], 2:[0.0,0.25,0.7], 3:[0.0,0.2,0.6],
    4:[0.2,0.5,1.0], 5:[1.0,0.0,1.0], 6:[0.8,0.0,1.0], 7:[0.6,0.0,0.8],
    8:[1.0,0.5,0.0], 9:[1.0,0.0,0.0], 10:[0.4,0.4,0.4], 11:[1.0,0.4,0.0],
    12:[0.8,0.0,1.0], 13:[1.0,0.0,1.0], 14:[1.0,0.85,0.0],
    15:[0.1,0.7,0.1], 16:[0.45,0.25,0.1], 17:[0.7,0.2,0.1],
    18:[0.25,0.25,0.25], 19:[1.0,1.0,1.0], 20:[0.55,0.45,0.3],
    21:[0.85,0.35,0.35], 22:[0.65,0.15,0.10]
}

parser = argparse.ArgumentParser()
parser.add_argument("npz")
parser.add_argument("--point-size", type=float, default=3.0)
args = parser.parse_args()

with np.load(args.npz, allow_pickle=False) as d:
    xyz = np.asarray(d["xyz"], dtype=np.float64)
    semantic = np.asarray(d["semantic_id"], dtype=np.int16)
    ranges = np.asarray(d["range_m"], dtype=np.float64)

print("="*60)
print("SCALA2 RAYCAST VIEWER")
print("="*60)
print(f"Points       : {len(xyz):,}")
print(f"Range        : {ranges.min():.2f} - {ranges.max():.2f} m")
print(f"Semantic IDs : {np.unique(semantic).tolist()}")
print()

for sid in np.unique(semantic):
    print(f"semantic {sid:2d}: {np.count_nonzero(semantic == sid):,}")

colors = np.asarray([SEMANTIC_COLORS.get(int(s), [0.7,0.7,0.7]) for s in semantic])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Scala2 raycast - semantic coloring", width=1600, height=900)
vis.add_geometry(pcd)
vis.add_geometry(origin)

opt = vis.get_render_option()
opt.point_size = args.point_size
opt.background_color = np.array([0.05,0.05,0.05])

vis.run()
vis.destroy_window()