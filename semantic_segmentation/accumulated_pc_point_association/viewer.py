import argparse
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument("--pcd", required=True)
args = parser.parse_args()


print(f"Loading: {args.pcd}")

pcd = o3d.io.read_point_cloud(args.pcd)

print(f"Number of points: {len(pcd.points):,}")
print(f"Has colors: {pcd.has_colors()}")
print(f"Bounds min: {pcd.get_min_bound()}")
print(f"Bounds max: {pcd.get_max_bound()}")


o3d.visualization.draw_geometries(
    [pcd],
    window_name="Waymo Accumulated Static Scene",
    width=1600,
    height=900,
)