#!/usr/bin/env python3
"""View the reconstructed static MLS scene with Waymo semantic coloring."""
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d

COLORS = {
  1:[0.90,0.10,0.10],2:[0.75,0.20,0.10],3:[0.70,0.10,0.25],4:[0.65,0.25,0.20],
  5:[1.00,0.45,0.00],6:[1.00,0.70,0.00],7:[0.60,0.10,0.80],8:[0.95,0.85,0.10],
  9:[1.00,0.35,0.35],10:[0.45,0.45,0.45],11:[1.00,0.30,0.00],12:[0.00,0.70,0.90],
  13:[0.20,0.40,1.00],14:[0.95,0.75,0.10],15:[0.10,0.65,0.10],16:[0.35,0.20,0.10],
  17:[0.65,0.35,0.25],18:[0.30,0.30,0.30],19:[1.00,1.00,1.00],20:[0.55,0.50,0.40],
  21:[0.75,0.50,0.40],22:[0.70,0.25,0.20]
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reconstruction-root", required=True, type=Path)
    p.add_argument("--max-points", type=int, default=3000000)
    p.add_argument("--point-size", type=float, default=1.5)
    a=p.parse_args()
    root=a.reconstruction_root.resolve()
    with (root/"static_manifest.json").open() as f: manifest=json.load(f)
    rng=np.random.default_rng(13)
    xyz_parts=[]; sem_parts=[]
    tiles=manifest["tiles"]
    counts=np.array([int(t.get("point_count",t.get("output_points",0))) for t in tiles],dtype=np.int64)
    total=max(int(counts.sum()),1)
    for tile,count in zip(tiles,counts):
        with np.load(root/tile["file"],allow_pickle=False) as d:
            xyz=np.asarray(d["xyz"],np.float32); sem=np.asarray(d["semantic_id"],np.int16)
        if a.max_points>0:
            quota=max(1,int(round(a.max_points*count/total)))
            if len(xyz)>quota:
                idx=rng.choice(len(xyz),quota,replace=False); xyz=xyz[idx]; sem=sem[idx]
        xyz_parts.append(xyz); sem_parts.append(sem)
    xyz=np.concatenate(xyz_parts); sem=np.concatenate(sem_parts)
    colors=np.asarray([COLORS.get(int(s),[0.15,0.15,0.15]) for s in sem],dtype=np.float64)
    cloud=o3d.geometry.PointCloud()
    cloud.points=o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors=o3d.utility.Vector3dVector(colors)
    vis=o3d.visualization.Visualizer()
    vis.create_window("Semantic static MLS v1",width=1500,height=950)
    vis.add_geometry(cloud)
    opt=vis.get_render_option(); opt.background_color=np.array([0.02,0.02,0.02]); opt.point_size=a.point_size
    vis.reset_view_point(True)
    print(f"Loaded {len(xyz):,} displayed points from {len(tiles)} tiles.")
    print("Semantic colors: road=gray, lane=white, sidewalk=brick red, building=yellow, vegetation=green, pedestrian=purple.")
    vis.run(); vis.destroy_window()
if __name__=="__main__": main()
