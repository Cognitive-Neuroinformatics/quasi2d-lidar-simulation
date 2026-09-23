#!/usr/bin/env python3
import numpy as np
import open3d as o3d

ROOT="/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/waymo_lidargs_mls_study/recon_related/segment-17791493328130181905_1480_000_1500_000_with_camera_labels"

OLD=f"{ROOT}/static_recon_labels_old.npz"
NEW=f"{ROOT}/static_recon_labels.npz"

COLORS={
    -1:[0.5,0.5,0.5], 0:[0.5,0.5,0.5],
    1:[0.0,0.4,1.0], 2:[0.0,0.25,0.7], 3:[0.0,0.2,0.6], 4:[0.2,0.5,1.0],
    5:[1.0,0.0,1.0], 6:[0.8,0.0,1.0], 7:[0.6,0.0,0.8],
    8:[1.0,0.5,0.0], 9:[1.0,0.0,0.0], 10:[0.4,0.4,0.4], 11:[1.0,0.4,0.0],
    12:[0.8,0.0,1.0], 13:[1.0,0.0,1.0], 14:[1.0,0.85,0.0],
    15:[0.1,0.7,0.1], 16:[0.45,0.25,0.1], 17:[0.7,0.2,0.1],
    18:[0.25,0.25,0.25], 19:[1.0,1.0,1.0], 20:[0.55,0.45,0.3],
    21:[0.85,0.35,0.35], 22:[0.65,0.15,0.10]
}

def load(path):
    with np.load(path, allow_pickle=False) as d:
        print(f"\n{path}")
        print("Keys:", d.files)
        xyz=np.asarray(d["xyz"], dtype=np.float64)
        sem=np.asarray(d["semantic_id"], dtype=np.int16)
    print(f"Points: {len(xyz):,}")
    for sid in np.unique(sem):
        print(f"  semantic {sid:3d}: {np.count_nonzero(sem==sid):,}")
    return xyz, sem

old_xyz,old_sem=load(OLD)
new_xyz,new_sem=load(NEW)

pcd=o3d.geometry.PointCloud()
state={"dataset":"old","road_only":False}

def set_cloud():
    if state["dataset"]=="old":
        xyz,sem=old_xyz,old_sem
        name="OLD"
    else:
        xyz,sem=new_xyz,new_sem
        name="NEW"

    if state["road_only"]:
        keep=sem==18
        xyz=xyz[keep]
        sem=sem[keep]
        mode="ROAD ONLY"
    else:
        mode="ALL SEMANTICS"

    colors=np.asarray([COLORS.get(int(s),[0.7,0.7,0.7]) for s in sem], dtype=np.float64)
    pcd.points=o3d.utility.Vector3dVector(xyz)
    pcd.colors=o3d.utility.Vector3dVector(colors)
    print(f"\nShowing {name} | {mode} | {len(xyz):,} points")
    return False

def show_old(vis):
    state["dataset"]="old"
    set_cloud()
    vis.update_geometry(pcd)
    vis.reset_view_point(False)
    return False

def show_new(vis):
    state["dataset"]="new"
    set_cloud()
    vis.update_geometry(pcd)
    vis.reset_view_point(False)
    return False

def toggle_road(vis):
    state["road_only"]=not state["road_only"]
    set_cloud()
    vis.update_geometry(pcd)
    vis.reset_view_point(False)
    return False

set_cloud()

vis=o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Static reconstruction diagnostic",1600,900)
vis.add_geometry(pcd)

vis.register_key_callback(ord("1"),show_old)
vis.register_key_callback(ord("2"),show_new)
vis.register_key_callback(ord("R"),toggle_road)

opt=vis.get_render_option()
opt.point_size=1.5
opt.background_color=np.array([0.05,0.05,0.05])

print("\nCONTROLS")
print("1 : OLD static_recon_labels_old.npz")
print("2 : NEW static_recon_labels.npz")
print("R : toggle ROAD-only / full semantic cloud")

vis.run()
vis.destroy_window()