#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

SEMANTIC_COLORS = {
    0:(0.45,0.45,0.45), 1:(0.10,0.70,0.95), 2:(0.05,0.55,0.85), 3:(0.05,0.40,0.70),
    4:(0.10,0.30,0.65), 5:(0.85,0.45,0.10), 6:(0.95,0.60,0.15), 7:(0.65,0.20,0.85),
    8:(0.95,0.15,0.15), 9:(1.00,0.35,0.10), 10:(0.70,0.70,0.70), 11:(1.00,0.65,0.10),
    12:(0.95,0.75,0.10), 13:(0.80,0.60,0.10), 14:(0.95,0.85,0.15), 15:(0.10,0.65,0.20),
    16:(0.45,0.25,0.10), 17:(0.60,0.30,0.15), 18:(0.35,0.35,0.35), 19:(1.00,1.00,1.00),
    20:(0.55,0.50,0.45), 21:(0.85,0.45,0.45), 22:(0.75,0.25,0.20)
}
BLUE=np.asarray([0.10,0.45,1.00]); ORANGE=np.asarray([1.00,0.35,0.05]); GREEN=np.asarray([0.20,0.85,0.35])

def sample(xyz, sem, maximum, seed):
    if maximum <= 0 or len(xyz) <= maximum: return xyz, sem
    rng=np.random.default_rng(seed); idx=rng.choice(len(xyz), maximum, replace=False)
    return xyz[idx], sem[idx]

def sem_colors(sem):
    return np.asarray([SEMANTIC_COLORS.get(int(s),(0.8,0.8,0.8)) for s in sem], dtype=np.float64)

def cloud(xyz, colors):
    p=o3d.geometry.PointCloud()
    p.points=o3d.utility.Vector3dVector(np.asarray(xyz,np.float64))
    p.colors=o3d.utility.Vector3dVector(np.asarray(colors,np.float64))
    return p

def main():
    ap=argparse.ArgumentParser(description="Compare original and densified NPZ without under-sampling generated support.")
    ap.add_argument("--original", required=True, type=Path)
    ap.add_argument("--densified", required=True, type=Path)
    ap.add_argument("--max-original", type=int, default=1500000)
    ap.add_argument("--max-generated", type=int, default=1000000)
    ap.add_argument("--max-full-densified", type=int, default=2500000)
    ap.add_argument("--point-size", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=17)
    a=ap.parse_args()

    with np.load(a.original, allow_pickle=False) as d:
        ox=np.asarray(d["xyz"],np.float32); os=np.asarray(d["semantic_id"],np.int16)
    with np.load(a.densified, allow_pickle=False) as d:
        dx=np.asarray(d["xyz"],np.float32); ds=np.asarray(d["semantic_id"],np.int16)
        if "is_generated" not in d.files: raise KeyError("densified NPZ has no is_generated array")
        gen=np.asarray(d["is_generated"],np.uint8).astype(bool)

    gx,gs=dx[gen],ds[gen]
    print(f"Original total   : {len(ox):,}")
    print(f"Densified total  : {len(dx):,}")
    print(f"Generated total  : {len(gx):,}")
    print(f"Generated fraction of densified: {100.0*len(gx)/max(len(dx),1):.3f}%")

    oxs,oss=sample(ox,os,a.max_original,a.seed)
    gxs,gss=sample(gx,gs,a.max_generated,a.seed+1)
    dxs,dss=sample(dx,ds,a.max_full_densified,a.seed+2)

    oc=cloud(oxs,np.tile(BLUE,(len(oxs),1)))
    gc=cloud(gxs,np.tile(ORANGE,(len(gxs),1)))
    dc=cloud(dxs,np.tile(GREEN,(len(dxs),1)))

    osem=sem_colors(oss); gsem=sem_colors(gss); dsem=sem_colors(dss)
    state={"o":True,"g":True,"d":False,"sem":False}

    vis=o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Original + Generated Support",1600,900)
    vis.add_geometry(oc); vis.add_geometry(gc)
    opt=vis.get_render_option(); opt.point_size=a.point_size; opt.background_color=np.asarray([0.02,0.02,0.02])

    def toggle(v,key,obj):
        if state[key]: v.remove_geometry(obj,reset_bounding_box=False)
        else: v.add_geometry(obj,reset_bounding_box=False)
        state[key]=not state[key]; return False
    def recolor(v):
        state["sem"]=not state["sem"]
        oc.colors=o3d.utility.Vector3dVector(osem if state["sem"] else np.tile(BLUE,(len(oxs),1)))
        gc.colors=o3d.utility.Vector3dVector(gsem if state["sem"] else np.tile(ORANGE,(len(gxs),1)))
        dc.colors=o3d.utility.Vector3dVector(dsem if state["sem"] else np.tile(GREEN,(len(dxs),1)))
        v.update_geometry(oc); v.update_geometry(gc); v.update_geometry(dc)
        return False

    vis.register_key_callback(ord("1"),lambda v:toggle(v,"o",oc))
    vis.register_key_callback(ord("2"),lambda v:toggle(v,"d",dc))
    vis.register_key_callback(ord("3"),lambda v:toggle(v,"g",gc))
    vis.register_key_callback(ord("S"),recolor)
    vis.register_key_callback(ord("C"),recolor)

    print(f"Displayed original : {len(oxs):,}")
    print(f"Displayed generated: {len(gxs):,}  <-- sampled separately, not diluted by original points")
    print(f"Displayed full densified when enabled: {len(dxs):,}")
    print("Keys: 1 original, 2 full densified, 3 generated-only, S/C semantic vs comparison colors")
    vis.run(); vis.destroy_window()

if __name__=="__main__": main()