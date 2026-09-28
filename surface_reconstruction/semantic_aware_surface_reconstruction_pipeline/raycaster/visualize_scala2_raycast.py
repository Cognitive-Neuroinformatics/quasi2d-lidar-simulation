#!/usr/bin/env python3
"""Compact SCALA2 raycast viewer: semantics/instances/source/elevation, boxes, slope profile, camera export, GIF."""
from __future__ import annotations
import argparse,colorsys,json,math,pickle
from pathlib import Path
import numpy as np
from visualize_mls_surface import SEMANTIC_COLORS

BOX_EDGES=np.asarray([[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]],np.int32)
STATIC_BOX_COLOR=np.asarray([.05,.20,.90]);DYNAMIC_BOX_COLOR=np.asarray([.90,.08,.08])
GROUND_IDS=np.asarray([18,19,20,21,22],np.int16);DEFAULT_SLOPE_IDS=(18,19)

def require_open3d():
    try: import open3d as o3d
    except ImportError as e: raise RuntimeError("Install open3d") from e
    return o3d

def require_matplotlib():
    try: import matplotlib,matplotlib.pyplot as plt
    except ImportError as e: raise RuntimeError("Install matplotlib for Z/H diagnostics") from e
    return matplotlib,plt

def instance_color(v): return [0.35]*3 if v<=0 else colorsys.hsv_to_rgb((v*.618033988749895)%1,.8,.95)
def transform_points(p,T):
    p=np.asarray(p,float);T=np.asarray(T,float).reshape(4,4)
    return (np.c_[p,np.ones(len(p))]@T.T)[:,:3]

def sensor_to_world(data): return transform_points(data["xyz"],np.linalg.inv(np.asarray(data["world_to_sensor"],float).reshape(4,4)))

def local_horizontal(data):
    w=sensor_to_world(data);T=np.asarray(data["vehicle_to_world"],float).reshape(4,4)
    f=T[:2,0].astype(float);f/=max(np.linalg.norm(f),1e-12);l=np.asarray([-f[1],f[0]])
    dxy=w[:,:2]-T[:2,3];return w,dxy@f,dxy@l

def elevation_colors(data):
    mpl,_=require_matplotlib();w=sensor_to_world(data);m=np.isin(data["semantic_id"],GROUND_IDS);c=np.tile([.72,.72,.72],(len(w),1))
    if not np.any(m): return c
    z=w[m,2];lo,hi=np.percentile(z,[2,98]) if len(z)>1 else (float(z[0]),float(z[0])+1)
    if hi<=lo+1e-9: hi=lo+1
    try: cmap=mpl.colormaps["turbo"]
    except Exception:
        import matplotlib.cm as cm;cmap=cm.get_cmap("turbo")
    c[m]=np.asarray(cmap(np.clip((z-lo)/(hi-lo),0,1)))[:,:3];return c

def point_colors(data,mode):
    n=len(data["xyz"])
    if mode=="uniform": return np.tile([.05,.35,.95],(n,1))
    if mode=="semantic": return np.asarray([SEMANTIC_COLORS.get(int(v),[.15]*3) for v in data["semantic_id"]],float)
    if mode=="ground":
        p={-1:[.2]*3,0:[.95,.25,.15],1:[.1,.75,.2]};return np.asarray([p.get(int(v),[.2]*3) for v in data["ground_id"]],float)
    if mode=="instance": return np.asarray([instance_color(int(v)) for v in data["instance_id"]],float)
    if mode=="source":
        p={0:[.10,.35,.95],1:[1,.20,.05]};return np.asarray([p.get(int(v),[.2]*3) for v in data["source_type"]],float)
    if mode=="elevation": return elevation_colors(data)
    raise ValueError(mode)

def slope_profile(data,ids,half_width,bin_m,window_m,min_pts,max_range):
    w,s,d=local_horizontal(data);m=np.isin(data["semantic_id"],np.asarray(ids,np.int16))&np.isfinite(w).all(1)&(s>=.5)&(s<=max_range)&(np.abs(d)<=half_width)
    s,z=s[m],w[m,2]
    if len(s)<max(8,min_pts*2): raise RuntimeError(f"Only {len(s)} usable slope points")
    a=math.floor(float(s.min())/bin_m)*bin_m;b=math.ceil(float(s.max())/bin_m)*bin_m+bin_m;e=np.arange(a,b+.5*bin_m,bin_m);x=.5*(e[:-1]+e[1:])
    med=np.full(len(x),np.nan);cnt=np.zeros(len(x),np.int32)
    for i in range(len(x)):
        q=(s>=e[i])&(s<e[i+1]);cnt[i]=q.sum()
        if cnt[i]>=min_pts: med[i]=np.median(z[q])
    q=np.isfinite(med);x,z,cnt=x[q],med[q],cnt[q]
    if len(x)<4: raise RuntimeError("Too few valid longitudinal bins")
    zs=np.empty(len(x));g=np.empty(len(x));hw=max(.5*window_m,1.5*bin_m)
    for i,c in enumerate(x):
        q=np.abs(x-c)<=hw
        if q.sum()<3:
            k=np.argsort(np.abs(x-c))[:min(5,len(x))];q=np.zeros(len(x),bool);q[k]=1
        xx=x[q]-c;yy=z[q];deg=2 if len(xx)>=3 else 1;coef=np.polyfit(xx,yy,deg,w=np.sqrt(np.maximum(cnt[q],1)))
        zs[i]=np.polyval(coef,0);g[i]=100*(coef[1] if deg==2 else coef[0])
    return x,z,zs,g,len(s)

def show_slope(data,path,args):
    _,plt=require_matplotlib();x,z,zs,g,n=slope_profile(data,args.slope_semantic_ids,args.slope_half_width_m,args.slope_bin_m,args.slope_window_m,args.slope_min_points,args.slope_max_range_m)
    fig,ax=plt.subplots(2,1,sharex=True,figsize=(11,7))
    ax[0].scatter(x,z,s=18,alpha=.55,label="median raycast elevation");ax[0].plot(x,zs,lw=2,label="local quadratic profile");ax[0].set_ylabel("world z [m]");ax[0].set_title(f"Frame {path.stem} | IDs={args.slope_semantic_ids} | |d|≤{args.slope_half_width_m:g} m");ax[0].grid(alpha=.3);ax[0].legend()
    ax[1].plot(x,g,lw=2);ax[1].axhline(0,ls="--",lw=1);ax[1].set_xlabel("forward distance s [m]");ax[1].set_ylabel("grade [%]");ax[1].grid(alpha=.3);fig.tight_layout()
    print(f"\nSlope frame={path.stem}: selected={n:,},bins={len(x)},median={np.nanmedian(g):.2f}%,p05={np.nanpercentile(g,5):.2f}%,p95={np.nanpercentile(g,95):.2f}%,min={np.nanmin(g):.2f}%,max={np.nanmax(g):.2f}%")
    plt.show(block=False);plt.pause(.001)

def box_local_corners(l,w,h):
    x,y,z=.5*l,.5*w,.5*h
    return np.asarray([[-x,-y,-z],[-x,y,-z],[x,-y,-z],[x,y,-z],[-x,-y,z],[-x,y,z],[x,-y,z],[x,y,z]],float)

def make_box(o3d,corners,color):
    g=o3d.geometry.LineSet();g.points=o3d.utility.Vector3dVector(corners);g.lines=o3d.utility.Vector2iVector(BOX_EDGES);g.colors=o3d.utility.Vector3dVector(np.tile(color,(len(BOX_EDGES),1)));return g

def infer_caseid(p):
    for q in [p,*p.parents]:
        if q.name.startswith("segment-"): return q.name

def discover_tracks(input_dir,explicit=None,dataset_root=None,caseid=None):
    if explicit:
        p=explicit.expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    caseid=caseid or infer_caseid(input_dir)
    if not caseid:return None
    if dataset_root:
        p=dataset_root.expanduser().resolve()/"temp"/caseid/"stage_a_tracks.json";return p if p.is_file() else None
    for a in [input_dir,*input_dir.parents]:
        p=a/"temp"/caseid/"stage_a_tracks.json"
        if p.is_file(): return p

def discover_meta(input_dir,dataset_root=None,caseid=None):
    caseid=caseid or infer_caseid(input_dir)
    if not caseid:return None
    if dataset_root:
        p=dataset_root.expanduser().resolve()/"meta_infos"/f"{caseid}.pkl";return p if p.is_file() else None
    for a in [input_dir,*input_dir.parents]:
        p=a/"meta_infos"/f"{caseid}.pkl"
        if p.is_file(): return p

def tracks_from_meta(path):
    with path.open("rb") as f:p=pickle.load(f)
    out={}
    for oi,fr in enumerate(p.get("frames",[])):
        l=fr.get("obj_label",{});tok=np.asarray(l.get("gt_boxes_token",[]));inst=np.asarray(l.get("gt_box_instance_ids",[]),np.int32);sem=np.asarray(l.get("gt_box_semantic_ids",[]),np.int16);dyn=np.asarray(l.get("gt_box_is_dynamic",[]),bool);oid=np.asarray(l.get("gt_box_lidargs_object_ids",[]),np.int32);box=np.asarray(l.get("gt_boxes",[]),float).reshape(-1,7);pose=np.asarray(l.get("gt_box_pose_world",[]),float).reshape(-1,4,4)
        for i in range(min(map(len,[tok,inst,sem,dyn,oid,box,pose]))):
            k=int(inst[i]);t=out.setdefault(k,{"waymo_track_id":str(tok[i]),"instance_id":k,"semantic_id":int(sem[i]),"is_dynamic":bool(dyn[i]),"lidargs_object_id":int(oid[i]),"frames":{}})
            t["frames"][str(oi)]={"output_frame_index":oi,"source_frame_index":int(fr.get("source_frame_index",oi)),"box_vehicle":box[i].tolist(),"box_pose_world":pose[i].reshape(-1).tolist()}
    print(f"Loaded {len(out):,} tracks from {path}");return out

def load_tracks(path):
    if path is None:return {}
    with path.open() as f:p=json.load(f)
    raw=p.get("tracks",p);it=raw.items() if isinstance(raw,dict) else enumerate(raw);out={}
    for k,t in it:
        if not isinstance(t,dict):continue
        iid=t.get("instance_id")
        if iid is None:
            try:iid=int(k)
            except:continue
        out[int(iid)]=t
    print(f"Loaded {len(out):,} tracks from {path}");return out

def frame_indices(data,path):
    oi=int(np.asarray(data["output_frame_index"]).reshape(-1)[0]);si=int(np.asarray(data["source_frame_index"]).reshape(-1)[0])
    try:fi=int(path.stem)
    except:fi=oi
    return oi,si,fi

def find_frame(track,oi,si,fi):
    f=track.get("frames",{})
    for label,v in [("output_frame_index",oi),("source_frame_index",si),("file_index",fi)]:
        for k in (str(v),v):
            if k in f:return f[k],label
    return None,None

def box_pose_vehicle(b):
    b=np.asarray(b,float).reshape(-1)
    if b.size<7:return None
    c,s=np.cos(b[6]),np.sin(b[6]);T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=b[:3];return T

def box_geometry(fr,data):
    if not isinstance(fr,dict):return None,None
    b=np.asarray(fr.get("box_vehicle",[]),float).reshape(-1)
    if b.size<6:return None,None
    p=np.asarray(fr.get("box_pose_world",[]),float)
    if p.size==16:return b[3:6],p.reshape(4,4)
    T=box_pose_vehicle(b)
    return (None,None) if T is None else (b[3:6],np.asarray(data["vehicle_to_world"],float).reshape(4,4)@T)

def rendered_stats(data):
    out={}
    for iid in np.unique(data["instance_id"]):
        iid=int(iid)
        if iid<=0:continue
        m=data["instance_id"]==iid;v,c=np.unique(data["source_type"][m],return_counts=True)
        out[iid]={"points":int(m.sum()),"source_type":int(v[np.argmax(c)]),"source_counts":{int(a):int(b) for a,b in zip(v,c)}}
    return out

def boxes_for_frame(o3d,tracks,data,path,min_pts):
    oi,si,fi=frame_indices(data,path);W=np.asarray(data["world_to_sensor"],float).reshape(4,4);geoms=[];records=[]
    for iid,st in sorted(rendered_stats(data).items()):
        base={"instance_id":iid,**st}
        if st["points"]<min_pts:records.append({**base,"status":"below_threshold"});continue
        tr=tracks.get(iid)
        if tr is None:records.append({**base,"status":"track_not_found"});continue
        fr,key=find_frame(tr,oi,si,fi)
        if fr is None:records.append({**base,"status":"frame_box_not_found"});continue
        dim,T=box_geometry(fr,data)
        if dim is None:records.append({**base,"status":"invalid_box_geometry"});continue
        corners=transform_points(transform_points(box_local_corners(*dim),T),W);color=DYNAMIC_BOX_COLOR if st["source_type"]==1 else STATIC_BOX_COLOR
        geoms.append(make_box(o3d,corners,color));records.append({**base,"status":"drawn","frame_key_used":key})
    return geoms,records

def print_boxes(r):
    if not r:print("  boxes: no rendered instance_id > 0");return
    d=[x for x in r if x["status"]=="drawn"];print(f"  boxes drawn={len(d)} (static={sum(x['source_type']==0 for x in d)},dynamic={sum(x['source_type']==1 for x in d)})")
    for x in r:
        if x["status"]!="drawn":print(f"    skip instance={x['instance_id']} points={x['points']} source={x['source_type']} reason={x['status']}")

def load_frame(path):
    req=("xyz","semantic_id","ground_id","instance_id","source_type","source_object_id","mirror_side","range","world_to_sensor","vehicle_to_world","output_frame_index","source_frame_index")
    with np.load(path,allow_pickle=False) as d:
        miss=[k for k in req if k not in d.files]
        if miss:raise KeyError(f"{path} missing {miss}")
        return {k:np.asarray(d[k]) for k in req}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",required=True,type=Path);p.add_argument("--point-size",type=float,default=3);p.add_argument("--min-box-points",type=int,default=4)
    p.add_argument("--tracks-json",type=Path);p.add_argument("--dataset-root",type=Path);p.add_argument("--caseid");p.add_argument("--no-boxes",action="store_true")
    p.add_argument("--camera-file",type=Path);p.add_argument("--screenshot-dir",type=Path);p.add_argument("--image-format",choices=("png","jpg"),default="png")
    p.add_argument("--make-gif",action="store_true");p.add_argument("--gif-name",default="animation.gif");p.add_argument("--gif-fps",type=float,default=10);p.add_argument("--gif-loop",type=int,default=0)
    p.add_argument("--slope-semantic-ids",nargs="+",type=int,default=list(DEFAULT_SLOPE_IDS));p.add_argument("--slope-half-width-m",type=float,default=4);p.add_argument("--slope-bin-m",type=float,default=.5)
    p.add_argument("--slope-window-m",type=float,default=6);p.add_argument("--slope-min-points",type=int,default=3);p.add_argument("--slope-max-range-m",type=float,default=80)
    return p.parse_args()

def main():
    a=parse_args()
    if a.min_box_points<0 or a.gif_fps<=0 or a.gif_loop<0 or min(a.slope_half_width_m,a.slope_bin_m,a.slope_window_m,a.slope_max_range_m)<=0 or a.slope_min_points<1:raise ValueError("Invalid numeric argument")
    o3d=require_open3d();root=a.input_dir.expanduser().resolve();files=sorted(root.glob("*.npz"),key=lambda p:int(p.stem))
    if not files:raise FileNotFoundError(f"No NPZ frames in {root}")
    cam=a.camera_file.expanduser().resolve() if a.camera_file else root/"camera_view.json";shots=a.screenshot_dir.expanduser().resolve() if a.screenshot_dir else root/"fixed_view_frames"
    tp=discover_tracks(root,a.tracks_json,a.dataset_root,a.caseid);tracks=load_tracks(tp) if tp else {};mp=None if tracks else discover_meta(root,a.dataset_root,a.caseid)
    if not tracks and mp:tracks=tracks_from_meta(mp)
    if not tracks:print("WARNING: no track metadata; boxes unavailable")
    state={"index":0,"mode":"semantic","boxes":not a.no_boxes and bool(tracks),"camera":None};cloud=o3d.geometry.PointCloud();active=[]

    def remove_boxes(vis):
        while active:vis.remove_geometry(active.pop(),reset_bounding_box=False)
    def apply_cam(vis,p):
        try:vis.get_view_control().convert_from_pinhole_camera_parameters(p,allow_arbitrary=True)
        except TypeError:vis.get_view_control().convert_from_pinhole_camera_parameters(p)
    def save_cam(vis,quiet=False):
        p=vis.get_view_control().convert_to_pinhole_camera_parameters();cam.parent.mkdir(parents=True,exist_ok=True)
        if not o3d.io.write_pinhole_camera_parameters(str(cam),p):raise RuntimeError(f"Could not save {cam}")
        state["camera"]=p
        if not quiet:print(f"\nSaved camera: {cam}")
        return p
    def load_cam(vis):
        if not cam.is_file():print(f"\nNo saved camera: {cam}");return False
        state["camera"]=o3d.io.read_pinhole_camera_parameters(str(cam));apply_cam(vis,state["camera"]);vis.update_renderer();return False
    def update(vis,reset=False,verbose=True):
        path=files[state["index"]];d=load_frame(path);cloud.points=o3d.utility.Vector3dVector(d["xyz"].astype(float));cloud.colors=o3d.utility.Vector3dVector(point_colors(d,state["mode"]));vis.update_geometry(cloud);remove_boxes(vis);records=[]
        if state["boxes"] and tracks:
            g,records=boxes_for_frame(o3d,tracks,d,path,a.min_box_points)
            for x in g:vis.add_geometry(x,reset_bounding_box=False);active.append(x)
        if reset:vis.reset_view_point(True)
        if verbose:
            oi,si,_=frame_indices(d,path);ms=int(d["mirror_side"][0]) if len(d["mirror_side"]) else -1;r0=float(np.nanmin(d["range"])) if len(d["range"]) else np.nan;r1=float(np.nanmax(d["range"])) if len(d["range"]) else np.nan
            print(f"\nFrame {path.stem} ({state['index']+1}/{len(files)}),output={oi},source={si},MS{ms},view={state['mode']},hits={len(d['xyz']):,},range={r0:.2f}..{r1:.2f},boxes={'ON' if state['boxes'] else 'OFF'}")
            if state["boxes"]:print_boxes(records)
            print("  N/P next/prev | S semantic | G ground | I instance | O source | U uniform | Z elevation | H slope | B boxes")
            print("  V save camera | L load camera | E export all | F GIF")
        return False
    def move(k):
        def cb(vis):state["index"]=(state["index"]+k)%len(files);return update(vis)
        return cb
    def mode(name):
        def cb(vis):state["mode"]=name;return update(vis)
        return cb
    def boxes_cb(vis):
        state["boxes"]=not state["boxes"]
        if state["boxes"] and not tracks:state["boxes"]=False;print("Boxes unavailable")
        return update(vis)
    def slope_cb(vis):
        p=files[state["index"]]
        try:show_slope(load_frame(p),p,a)
        except Exception as e:print(f"\nSlope failed for {p.stem}: {e}")
        return False
    def gif():
        from PIL import Image
        paths=sorted(shots.glob(f"*.{a.image_format}"),key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem)
        if not paths:print(f"\nNo exported frames in {shots}");return False
        ims=[]
        for p in paths:
            with Image.open(p) as im:ims.append(im.convert("RGB").copy())
        out=shots/a.gif_name;ims[0].save(out,save_all=True,append_images=ims[1:],duration=max(1,round(1000/a.gif_fps)),loop=a.gif_loop,optimize=False);print(f"\nGIF: {out}");return False
    def export(vis):
        if state["camera"] is None:save_cam(vis,True)
        c=state["camera"];old=state["index"];shots.mkdir(parents=True,exist_ok=True)
        for i,p in enumerate(files):
            state["index"]=i;update(vis,verbose=False);apply_cam(vis,c);vis.update_renderer();out=shots/f"{p.stem}.{a.image_format}"
            if not vis.capture_screen_image(str(out),do_render=True):raise RuntimeError(out)
            if i==0 or (i+1)%10==0 or i+1==len(files):print(f"  saved {i+1}/{len(files)}")
        state["index"]=old;update(vis,verbose=False);apply_cam(vis,c);vis.update_renderer()
        if a.make_gif:gif()
        return False

    vis=o3d.visualization.VisualizerWithKeyCallback();vis.create_window(window_name="SCALA2 raycast viewer",width=1400,height=900);vis.add_geometry(cloud)
    for key,cb in [(ord("N"),move(1)),(262,move(1)),(ord("P"),move(-1)),(263,move(-1)),(ord("S"),mode("semantic")),(ord("G"),mode("ground")),(ord("I"),mode("instance")),(ord("O"),mode("source")),(ord("U"),mode("uniform")),(ord("Z"),mode("elevation")),(ord("H"),slope_cb),(ord("B"),boxes_cb),(ord("V"),lambda v:(save_cam(v),False)[1]),(ord("L"),load_cam),(ord("E"),export),(ord("F"),lambda v:gif())]:vis.register_key_callback(key,cb)
    opt=vis.get_render_option();opt.background_color=np.asarray([1.,1.,1.]);opt.point_size=a.point_size;update(vis,True);vis.run();vis.destroy_window()

if __name__=="__main__":main()
