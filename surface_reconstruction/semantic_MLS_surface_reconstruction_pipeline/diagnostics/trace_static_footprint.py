#!/usr/bin/env python3
"""Trace suspicious object footprints backward through MLS -> final preprocessed cloud -> strict pre-densification residue."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SEMANTIC_NAMES={0:"UNDEFINED",1:"CAR",2:"TRUCK",3:"BUS",4:"OTHER_VEHICLE",5:"MOTORCYCLIST",6:"BICYCLIST",7:"PEDESTRIAN",8:"SIGN",9:"TRAFFIC_LIGHT",10:"POLE",11:"CONSTRUCTION_CONE",12:"BICYCLE",13:"MOTORCYCLE",14:"BUILDING",15:"VEGETATION",16:"TREE_TRUNK",17:"CURB",18:"ROAD",19:"LANE_MARKER",20:"OTHER_GROUND",21:"WALKABLE",22:"SIDEWALK"}
SEMANTIC_COLORS={0:[0.2,0.2,0.2],1:[1,0,0],2:[0.8,0.1,0.1],3:[0.7,0.1,0.1],4:[0.9,0.25,0.1],5:[1,0.55,0],6:[1,0.65,0],7:[1,0,0.8],8:[0,0.7,0],9:[1,1,0],10:[0.4,0.4,0.4],11:[1,0.5,0],12:[1,0.55,0],13:[1,0.45,0],14:[0.65,0.35,0.15],15:[0,0.65,0],16:[0.45,0.25,0.1],17:[0.2,0.4,1],18:[0.18,0.18,0.18],19:[0.45,0.45,1],20:[0.5,0.5,0.5],21:[0.9,0.35,0.35],22:[0.75,0.2,0.2]}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raycast",required=True,type=Path,help="One rendered sensor frame NPZ, e.g. front_center/points/017.npz")
    p.add_argument("--reconstruction-root",required=True,type=Path,help="Root containing static_manifest.json and static_tiles/")
    p.add_argument("--densified",required=True,type=Path,help="Final recon_related/<case>/static_recon_labels.npz used as MLS input")
    p.add_argument("--strict",required=True,type=Path,help="Strict pre-densification residue: preprocessing_residues/static_filter/static_recon_labels_strict.npz")
    p.add_argument("--radius",type=float,default=0.35,help="World-space crop radius around picked ghost cluster [m]")
    p.add_argument("--strict-near",type=float,default=0.05,help="Distance below which strict geometry is considered already present [m]")
    p.add_argument("--minimum-label-confidence",type=float,default=0.66)
    p.add_argument("--pick-color",choices=("semantic","source","instance"),default="semantic")
    p.add_argument("--hit-index",type=int,default=None,help="Skip GUI and use row index in raycast hit arrays")
    p.add_argument("--ray-index",type=int,default=None,help="Skip GUI and select the hit with this SCALA2 ray_index")
    p.add_argument("--center-world",type=float,nargs=3,default=None,metavar=("X","Y","Z"),help="Skip GUI and use an explicit world center")
    p.add_argument("--output-dir",type=Path,default=Path("ghost_trace"))
    return p.parse_args()

def load_raycast(path):
    need=("xyz","surface_xyz_world","semantic_id","instance_id","source_type","ray_index")
    with np.load(path,allow_pickle=False) as z:
        missing=[k for k in need if k not in z.files]
        if missing: raise KeyError(f"{path} missing {missing}")
        return {k:np.asarray(z[k]) for k in need}

def pick_hits(data,mode):
    try: import open3d as o3d
    except ImportError as e: raise RuntimeError("Interactive picking needs open3d. Install it or use --hit-index/--ray-index/--center-world.") from e
    xyz=np.asarray(data["xyz"],np.float64); sem=np.asarray(data["semantic_id"]); inst=np.asarray(data["instance_id"]); src=np.asarray(data["source_type"])
    if mode=="semantic": colors=np.asarray([SEMANTIC_COLORS.get(int(v),[0.2,0.2,0.2]) for v in sem],np.float64)
    elif mode=="source": colors=np.asarray([[0.1,0.35,0.95] if int(v)==0 else [1,0.2,0.05] for v in src],np.float64)
    else: colors=np.asarray([[0.35,0.35,0.35] if int(v)==0 else [1,0.1,0.7] for v in inst],np.float64)
    cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(xyz); cloud.colors=o3d.utility.Vector3dVector(colors)
    print(f"Open3D version: {getattr(o3d,'__version__','unknown')}")
    print("Pick one or more ghost points with SHIFT + LEFT CLICK. Selected points should be visibly marked. Press Q when finished.")
    if hasattr(o3d.visualization,"VisualizerWithVertexSelection"):
        vis=o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window("Pick ghost points: SHIFT+LEFT CLICK; Q when done",1400,900)
        vis.add_geometry(cloud)
        opt=vis.get_render_option(); opt.background_color=np.asarray([1,1,1]); opt.point_size=7.0
        vis.run()
        raw=vis.get_picked_points()
        picked=np.asarray([int(p.index) for p in raw],dtype=np.int64)
        vis.destroy_window()
    else:
        vis=o3d.visualization.VisualizerWithEditing()
        vis.create_window("Pick ghost points: SHIFT+LEFT CLICK; Q when done",1400,900)
        vis.add_geometry(cloud)
        opt=vis.get_render_option(); opt.background_color=np.asarray([1,1,1]); opt.point_size=7.0
        vis.run()
        picked=np.asarray(vis.get_picked_points(),dtype=np.int64)
        vis.destroy_window()
    if not len(picked): raise RuntimeError("No points were selected. If SHIFT+left click still does nothing, use --ray-index, --hit-index, or --center-world.")
    print(f"Picked {len(picked)} point(s): {picked.tolist()}")
    return picked

def sphere_aabb_intersects(center,radius,minimum,maximum):
    nearest=np.maximum(np.asarray(minimum,float),np.minimum(center,np.asarray(maximum,float))); return np.linalg.norm(nearest-center)<=radius

def crop_npz(path,center,radius,keys):
    with np.load(path,allow_pickle=False) as z:
        xyz=np.asarray(z["xyz"]); d2=np.sum((xyz-np.asarray(center,dtype=xyz.dtype))**2,axis=1); idx=np.flatnonzero(d2<=radius*radius)
        out={"global_index":idx,"xyz":xyz[idx]}
        for k in keys:
            if k in z.files: out[k]=np.asarray(z[k])[idx]
        meta={k:np.asarray(z[k]) for k in ("original_point_count","generated_point_count") if k in z.files}
    return out,meta

def load_static_mls_crop(root,center,radius):
    manifest_path=root/"static_manifest.json"
    with manifest_path.open() as f: manifest=json.load(f)
    parts=[]
    for tile in manifest["tiles"]:
        if not sphere_aabb_intersects(center,radius,tile["min_xyz"],tile["max_xyz"]): continue
        path=root/tile["file"]
        with np.load(path,allow_pickle=False) as z:
            xyz=np.asarray(z["xyz"]); d2=np.sum((xyz-np.asarray(center,dtype=xyz.dtype))**2,axis=1); m=d2<=radius*radius
            if np.any(m): parts.append({k:np.asarray(z[k])[m] for k in ("xyz","normal","intensity","semantic_id","ground_id","instance_id") if k in z.files})
    if not parts: return {"xyz":np.empty((0,3),np.float32)}
    keys=parts[0].keys(); return {k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}

def nearest_rows(query_xyz,crop,mask=None):
    xyz=np.asarray(crop["xyz"],np.float64)
    if mask is None: ids=np.arange(len(xyz),dtype=np.int64)
    else: ids=np.flatnonzero(mask)
    if not len(ids): return np.full(len(query_xyz),np.nan),np.full(len(query_xyz),-1,dtype=np.int64)
    dist,local=cKDTree(xyz[ids]).query(np.asarray(query_xyz,np.float64),k=1); return np.asarray(dist),ids[np.asarray(local,dtype=np.int64)]

def counts(values):
    if values is None or not len(values): return "{}"
    u,c=np.unique(values,return_counts=True); return "{"+", ".join(f"{int(a)}:{int(b)}" for a,b in zip(u,c))+"}"

def save_crop(path,crop):
    payload={k:v for k,v in crop.items() if isinstance(v,np.ndarray)}; np.savez_compressed(path,**payload)

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); ray=load_raycast(a.raycast)
    if a.center_world is not None: picked=np.arange(len(ray["xyz"]),dtype=np.int64); center=np.asarray(a.center_world,float)
    elif a.hit_index is not None: picked=np.asarray([a.hit_index],dtype=np.int64); center=np.asarray(ray["surface_xyz_world"][a.hit_index],float)
    elif a.ray_index is not None:
        match=np.flatnonzero(ray["ray_index"]==a.ray_index)
        if len(match)!=1: raise RuntimeError(f"ray_index {a.ray_index} matched {len(match)} hits")
        picked=match; center=np.asarray(ray["surface_xyz_world"][picked[0]],float)
    else:
        picked=pick_hits(ray,a.pick_color); center=np.median(np.asarray(ray["surface_xyz_world"][picked],float),axis=0)
    d2=np.sum((np.asarray(ray["surface_xyz_world"],float)-center)**2,axis=1); cluster=np.flatnonzero((d2<=a.radius*a.radius)&(ray["source_type"]==0)&(ray["instance_id"]==0))
    if not len(cluster): raise RuntimeError("No STATIC + instance_id=0 raycast hits inside the selected world-space radius")
    targets=np.asarray(ray["surface_xyz_world"][cluster],float); target_sem=np.asarray(ray["semantic_id"][cluster],np.int16)
    print(f"\nGhost center world: {center}"); print(f"Raycast cluster: {len(cluster):,} STATIC/instance-0 hits inside r={a.radius:.3f} m"); print(f"Raycast semantics: {counts(target_sem)} -> "+", ".join(f"{sid}:{SEMANTIC_NAMES.get(int(sid),'?')}" for sid in np.unique(target_sem)))

    mls=load_static_mls_crop(a.reconstruction_root,center,a.radius+0.40); print(f"Static MLS crop: {len(mls['xyz']):,} points")
    mls_dist,mls_idx=nearest_rows(targets,mls); mls_sem=np.asarray(mls.get("semantic_id",np.full(len(mls["xyz"]),-1)))[mls_idx]

    dens_keys=("semantic_id","ground_id","instance_id","intensity","label_confidence","observation_frame_index","is_generated","densification_source_type","ground_family","source_original_point_index")
    dens,dens_meta=crop_npz(a.densified,center,a.radius+0.60,dens_keys); print(f"Densified crop: {len(dens['xyz']):,} points")
    dens_idx=np.full(len(targets),-1,dtype=np.int64); dens_dist=np.full(len(targets),np.nan)
    for sid in np.unique(mls_sem):
        q=np.flatnonzero(mls_sem==sid); mask=np.ones(len(dens["xyz"]),dtype=bool)
        if "semantic_id" in dens: mask&=dens["semantic_id"]==sid
        if "instance_id" in dens: mask&=dens["instance_id"]==0
        if "label_confidence" in dens: mask&=dens["label_confidence"]>=a.minimum_label_confidence
        dist,idx=nearest_rows(targets[q],dens,mask); dens_dist[q]=dist; dens_idx[q]=idx

    strict_keys=("semantic_id","ground_id","instance_id","intensity","label_confidence","observation_frame_index")
    strict,_=crop_npz(a.strict,center,a.radius+0.60,strict_keys); print(f"Strict crop: {len(strict['xyz']):,} points")
    strict_dist,strict_idx=nearest_rows(targets,strict)

    valid_dens=dens_idx>=0; generated=np.full(len(targets),-1,dtype=np.int16); source_type=np.full(len(targets),-1,dtype=np.int16); obs=np.full(len(targets),-1,dtype=np.int32); source_original=np.full(len(targets),-1,dtype=np.int64)
    if "is_generated" in dens: generated[valid_dens]=dens["is_generated"][dens_idx[valid_dens]]
    if "densification_source_type" in dens: source_type[valid_dens]=dens["densification_source_type"][dens_idx[valid_dens]]
    if "observation_frame_index" in dens: obs[valid_dens]=dens["observation_frame_index"][dens_idx[valid_dens]]
    if "source_original_point_index" in dens: source_original[valid_dens]=dens["source_original_point_index"][dens_idx[valid_dens]]

    rows=[]
    for j,ri in enumerate(cluster):
        di=dens_idx[j]; si=strict_idx[j]
        rows.append({"raycast_hit_row":int(ri),"ray_index":int(ray["ray_index"][ri]),"semantic_id":int(ray["semantic_id"][ri]),"semantic_name":SEMANTIC_NAMES.get(int(ray["semantic_id"][ri]),"?"),"surface_x":float(targets[j,0]),"surface_y":float(targets[j,1]),"surface_z":float(targets[j,2]),"nearest_mls_m":float(mls_dist[j]),"mls_semantic_id":int(mls_sem[j]),"nearest_densified_m":float(dens_dist[j]),"densified_global_index":int(dens["global_index"][di]) if di>=0 else -1,"densified_is_generated":int(generated[j]),"densification_source_type":int(source_type[j]),"observation_frame_index":int(obs[j]),"source_original_point_index":int(source_original[j]),"nearest_strict_m":float(strict_dist[j]),"strict_global_index":int(strict["global_index"][si]) if si>=0 else -1,"strict_semantic_id":int(strict["semantic_id"][si]) if si>=0 and "semantic_id" in strict else -1,"strict_instance_id":int(strict["instance_id"][si]) if si>=0 and "instance_id" in strict else -1})
    csv_path=a.output_dir/"trace.csv"
    with csv_path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    save_crop(a.output_dir/"static_mls_crop.npz",mls); save_crop(a.output_dir/"densified_crop.npz",dens); save_crop(a.output_dir/"strict_crop.npz",strict)
    np.savez_compressed(a.output_dir/"raycast_ghost_cluster.npz",hit_row=cluster,xyz_sensor=ray["xyz"][cluster],surface_xyz_world=ray["surface_xyz_world"][cluster],semantic_id=ray["semantic_id"][cluster],instance_id=ray["instance_id"][cluster],source_type=ray["source_type"][cluster],ray_index=ray["ray_index"][cluster])

    g=generated[valid_dens]; original_frac=float(np.mean(g==0)) if len(g) else 0.0; generated_frac=float(np.mean(g==1)) if len(g) else 0.0; strict_close=float(np.mean(strict_dist<=a.strict_near)) if len(strict_dist) else 0.0
    print("\nTRACE RESULT")
    print(f"  nearest MLS distance median      : {np.nanmedian(mls_dist):.4f} m")
    print(f"  nearest densified distance median: {np.nanmedian(dens_dist):.4f} m")
    print(f"  densified source original        : {original_frac*100:.1f}%")
    print(f"  densified source generated       : {generated_frac*100:.1f}%")
    print(f"  nearest strict <= {a.strict_near:.3f} m       : {strict_close*100:.1f}%")
    if original_frac>=0.6 and strict_close>=0.6: verdict="The ghost is already supported by ORIGINAL strict-static points; investigate upstream static/dynamic filtering or semantic transfer."
    elif generated_frac>=0.6 and strict_close<0.6: verdict="The ghost is mainly GENERATED during densification; inspect ring/generic support generation around this location."
    elif strict_close<0.25 and np.nanmedian(dens_dist)>0.05: verdict="The pre-MLS inputs have weak support here; MLS smoothing/upsampling may be bridging into the ghost region."
    else: verdict="Mixed evidence. Inspect trace.csv and the three saved local crops; the nearest-source rows show which stage contributes each hit."
    print(f"  diagnosis                         : {verdict}")
    print(f"\nSaved: {csv_path}")
    print(f"       {a.output_dir/'raycast_ghost_cluster.npz'}")
    print(f"       {a.output_dir/'static_mls_crop.npz'}")
    print(f"       {a.output_dir/'densified_crop.npz'}")
    print(f"       {a.output_dir/'strict_crop.npz'}")

if __name__=="__main__": main()
