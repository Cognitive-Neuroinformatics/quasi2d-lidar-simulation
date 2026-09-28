#!/usr/bin/env python3
"""Run hybrid reconstruction, shared SCALA-2 raycasting, post-hit noise, or noise ablations."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SENSORS=["front_left","front_center","front_right","rear_left","rear_center","rear_right"]

def load_json(p):
    with p.open() as f:return json.load(f)

def cfg_path(v):
    p=Path(v);return p if p.is_absolute() else ROOT/p

def case_name(v):
    n=Path(str(v).strip()).name;return n[:-9] if n.endswith(".tfrecord") else n

def read_cases(p):
    out=[];seen=set()
    for raw in p.read_text().splitlines():
        v=raw.split("#",1)[0].strip()
        if not v:continue
        c=case_name(v)
        if c not in seen:out.append(c);seen.add(c)
    return out

def run(cmd,log=None):
    print("\n$ "+" ".join(map(str,cmd)));t=time.perf_counter()
    if log is None:subprocess.run(cmd,check=True);return time.perf_counter()-t
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as f:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:print(line,end="");f.write(line)
        rc=p.wait()
    if rc:raise subprocess.CalledProcessError(rc,cmd)
    return time.perf_counter()-t

def ensure_pcl(exe):
    if exe.is_file():return
    run([str(ROOT/"scripts/build_pcl_mls.sh")])
    if not exe.is_file():raise FileNotFoundError(exe)

def roots(a,case):
    ds=a.dataset_root.expanduser().resolve()
    recon=a.reconstruction_root.expanduser().resolve() if getattr(a,"reconstruction_root",None) else ds/"semantic_aware_surface_reconstruction"/"hybrid_semantic_surface"/case
    if getattr(a,"raycast_root",None):ray=a.raycast_root.expanduser().resolve()
    else:ray=recon/(f"scala2_raycast_cuda_{a.precision}" if a.rasterizer=="cuda" else "scala2_raycast_cpu")
    return ds,recon,ray

def reconstruct_cmd(a,case,ds,recon,cfg):
    r=cfg["reconstruction"];exe=cfg_path(r["pcl_executable"]);ensure_pcl(exe)
    cmd=[sys.executable,str(ROOT/"reconstruction/hybrid/reconstruct_hybrid_semantic_surface.py"),
         "--dataset-root",str(ds),"--caseid",case,
         "--hybrid-config",str(cfg_path(r["hybrid_config"])),"--base-mls-config",str(cfg_path(r["base_mls_config"])),
         "--mls-script",str(ROOT/"reconstruction/mls/reconstruct_semantic_static_mls.py"),"--pcl-executable",str(exe),
         "--output-root",str(recon),"--tile-size",str(a.tile_size),"--tile-halo",str(a.tile_halo),
         "--minimum-label-confidence",str(a.min_label_confidence),"--surface-workers",str(a.surface_workers),
         "--pcl-threads",str(a.pcl_threads),"--mls-workers",str(a.mls_workers),"--attribute-workers",str(a.attribute_workers),
         "--npz-compression",a.npz_compression,"--include-dynamics" if a.include_dynamics else "--no-include-dynamics"]
    if a.force:cmd.append("--overwrite")
    if a.static_input:cmd+=["--static-input",str(a.static_input.expanduser().resolve())]
    if a.dynamic_input_root:cmd+=["--dynamic-input-root",str(a.dynamic_input_root.expanduser().resolve())]
    if a.pcl_work_root:cmd+=["--pcl-work-root",str(a.pcl_work_root.expanduser().resolve())]
    return cmd

def raycast_cmd(a,case,ds,recon,ray):
    shared=["--dataset-root",str(ds),"--caseid",case,"--reconstruction-root",str(recon),"--output-root",str(ray),
            "--sensors",*a.sensors,"--start-frame",str(a.start_frame),"--first-mirror-side",str(a.first_mirror_side),
            "--minimum-range",str(a.min_range),"--max-range",str(a.max_range),"--intersection-mode",a.intersection_mode,
            "--patch-radius",str(a.patch_radius),"--hit-radius",str(a.hit_radius),"--static-tile-cache",str(a.static_tile_cache),
            "--property-cache",str(a.property_cache),"--npz-compression",a.npz_compression,
            "--noise-output",a.noise_output,"--noise-model",a.noise_model,
            "--noise-range-sigma-m",str(a.noise_range_sigma),"--noise-azimuth-sigma-deg",str(a.noise_azimuth_sigma),
            "--noise-polar-sigma-deg",str(a.noise_polar_sigma),"--noise-seed",str(a.noise_seed),
            "--noise-incidence-max-angle-deg",str(a.noise_incidence_max_angle)]
    if a.end_frame is not None:shared+=["--end-frame",str(a.end_frame)]
    if a.static_only:shared.append("--static-only")
    if a.force:shared.append("--overwrite")
    if a.rasterizer=="cuda":
        return [sys.executable,str(ROOT/"raycaster/raycast_scala2_cuda.py"),*shared,
                "--point-batch-size",str(a.point_batch_size),"--gpu-cache-gb",str(a.gpu_cache_gb),
                "--devices",*a.devices,"--precision",a.precision]
    return [sys.executable,str(ROOT/"raycaster/raycast_scala2.py"),*shared,
            "--point-batch-size",str(a.cpu_point_batch_size),"--sensor-workers",str(a.sensor_workers),
            "--ckdtree-workers",str(a.ckdtree_workers),"--candidate-lookup",a.candidate_lookup,
            "--ray-neighbor-count",str(a.ray_neighbor_count)]

def run_scene(a,case,cfg,batch_log=None):
    ds,recon,ray=roots(a,case);logroot=batch_log or ds/"semantic_aware_surface_reconstruction"/"pipeline_logs"/case/time.strftime("%Y%m%d_%H%M%S");logroot.mkdir(parents=True,exist_ok=True)
    default_static=ds/"recon_related"/case/"static_recon_labels.npz"
    result={"case":case,"reconstruction_root":str(recon),"raycast_root":str(ray),"stage":a.stage,"timings_s":{},"status":"running"};t=time.perf_counter()
    try:
        if a.stage in ("all","reconstruct"):
            if not a.static_input and not default_static.is_file():raise FileNotFoundError(f"Missing preprocessing input: {default_static}")
            if (recon/"static_manifest.json").is_file() and not a.force:print(f"[{case}] reconstruction exists; skip. Use --force to rebuild.")
            else:result["timings_s"]["reconstruction"]=run(reconstruct_cmd(a,case,ds,recon,cfg),logroot/"reconstruction.log")
        if a.stage in ("all","raycast"):
            if not (recon/"static_manifest.json").is_file():raise FileNotFoundError(f"Missing reconstruction: {recon/'static_manifest.json'}")
            result["timings_s"]["raycast"]=run(raycast_cmd(a,case,ds,recon,ray),logroot/"raycast.log")
        result["status"]="completed"
    except Exception as e:result["status"]="failed";result["error"]=repr(e);raise
    finally:
        result["timings_s"]["total"]=time.perf_counter()-t;(logroot/"pipeline_summary.json").write_text(json.dumps(result,indent=2))
        print(f"\nSummary: {logroot/'pipeline_summary.json'}")
    return result

def add_noise(a,case,cfg):
    _,_,ray=roots(a,case);n=cfg["raycast"]["noise"];total=0
    for sensor in a.sensors:
        inp=ray/sensor/"points"
        if not inp.is_dir():raise FileNotFoundError(inp)
        cmd=[sys.executable,str(ROOT/"raycaster/add_scala2_noise.py"),"--input-dir",str(inp),
             "--noise-model",a.noise_model,"--range-sigma-m",str(a.noise_range_sigma),
             "--azimuth-sigma-deg",str(a.noise_azimuth_sigma),"--polar-sigma-deg",str(a.noise_polar_sigma),
             "--incidence-max-angle-deg",str(a.noise_incidence_max_angle),"--seed",str(a.noise_seed),
             "--npz-compression",a.npz_compression]
        if a.force:cmd.append("--overwrite")
        total+=run(cmd)
    print(f"\nNoise-only complete in {total:.1f}s: {ray}")

def run_ablation(a,case,cfg):
    _,_,ray=roots(a,case)
    cmd=[sys.executable,str(ROOT/"raycaster/generate_scala2_noise_ablation.py"),"--raycast-root",str(ray),
         "--sensors",*a.sensors,"--profiles",*a.profiles,"--range-sigma-m",str(a.noise_range_sigma),
         "--azimuth-sigma-deg",str(a.noise_azimuth_sigma),"--polar-sigma-deg",str(a.noise_polar_sigma),
         "--incidence-max-angle-deg",str(a.noise_incidence_max_angle),"--seed",str(a.noise_seed),
         "--npz-compression",a.npz_compression]
    if a.force:cmd.append("--overwrite")
    run(cmd)

def common(p,cfg):
    run_cfg=cfg["run"];r=cfg["reconstruction"];ray=cfg["raycast"];s=ray["shared"];n=ray["noise"];cu=ray["cuda"];cp=ray["cpu"]
    p.add_argument("--config",type=Path,default=ROOT/"configs/baseline_default.json")
    p.add_argument("--dataset-root",type=Path,default=Path(os.environ.get("WAYMO_SURFACE_ROOT",run_cfg["dataset_root"])))
    p.add_argument("--rasterizer",choices=["cuda","cpu"],default=run_cfg["rasterizer"]);p.add_argument("--reconstruction-root",type=Path);p.add_argument("--raycast-root",type=Path)
    p.add_argument("--npz-compression",choices=["stored","compressed"],default=run_cfg["npz_compression"]);p.add_argument("--force",action="store_true")
    p.add_argument("--sensors",nargs="+",choices=SENSORS,default=run_cfg["raycast_sensors"]);p.add_argument("--start-frame",type=int,default=run_cfg["raycast_start_frame"]);p.add_argument("--end-frame",type=int,default=run_cfg["raycast_end_frame"])
    p.add_argument("--intersection-mode",default=s["intersection_mode"]);p.add_argument("--patch-radius",type=float,default=s["patch_radius_m"]);p.add_argument("--hit-radius",type=float,default=s["hit_radius_m"])
    p.add_argument("--min-range",type=float,default=s["minimum_range_m"]);p.add_argument("--max-range",type=float,default=s["maximum_range_m"]);p.add_argument("--first-mirror-side",type=int,choices=[0,1],default=s["first_mirror_side"])
    p.add_argument("--static-tile-cache",type=int,default=s["static_tile_cache"]);p.add_argument("--property-cache",type=int,default=s["property_cache"]);p.add_argument("--static-only",action="store_true")
    p.add_argument("--noise-output",choices=["clean","noisy","both"],default=n["output"]);p.add_argument("--noise-model",choices=["datasheet_gaussian","incidence_secant"],default=n["model"])
    p.add_argument("--noise-range-sigma",type=float,default=n["range_sigma_m"]);p.add_argument("--noise-azimuth-sigma",type=float,default=n["azimuth_sigma_deg"]);p.add_argument("--noise-polar-sigma",type=float,default=n["polar_sigma_deg"])
    p.add_argument("--noise-seed",type=int,default=n["seed"]);p.add_argument("--noise-incidence-max-angle",type=float,default=n["incidence_max_angle_deg"])
    p.add_argument("--devices",nargs="+",default=cu["devices"]);p.add_argument("--precision",choices=["float32","float64"],default=cu["precision"]);p.add_argument("--point-batch-size",type=int,default=cu["point_batch_size"]);p.add_argument("--gpu-cache-gb",type=float,default=cu["gpu_cache_gb_per_gpu"])
    p.add_argument("--cpu-point-batch-size",type=int,default=cp["point_batch_size"]);p.add_argument("--sensor-workers",type=int,default=cp["sensor_workers"]);p.add_argument("--ckdtree-workers",type=int,default=cp["ckdtree_workers"]);p.add_argument("--candidate-lookup",default=cp["candidate_lookup"]);p.add_argument("--ray-neighbor-count",type=int,default=cp["ray_neighbor_count"])

def reconstruction_args(p,cfg):
    r=cfg["reconstruction"]
    p.add_argument("--static-input",type=Path);p.add_argument("--dynamic-input-root",type=Path);p.add_argument("--pcl-work-root",type=Path)
    p.add_argument("--tile-size",type=float,default=r["tile_size_m"]);p.add_argument("--tile-halo",type=float,default=r["tile_halo_m"]);p.add_argument("--min-label-confidence",type=float,default=r["minimum_label_confidence"])
    p.add_argument("--surface-workers",type=int,default=r["surface_workers"]);p.add_argument("--pcl-threads",type=int,default=r["pcl_threads"]);p.add_argument("--mls-workers",type=int,default=r["mls_workers"]);p.add_argument("--attribute-workers",type=int,default=r["attribute_workers"])
    p.add_argument("--include-dynamics",action=argparse.BooleanOptionalAction,default=r["include_dynamics"])

def main():
    pre=argparse.ArgumentParser(add_help=False);pre.add_argument("--config",type=Path,default=ROOT/"configs/baseline_default.json");known,_=pre.parse_known_args()
    cfg=load_json(known.config.expanduser().resolve())
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="mode",required=True)
    s=sub.add_parser("scene",help="Run one scene");s.add_argument("case");s.add_argument("--stage",choices=["all","reconstruct","raycast"],default="all");common(s,cfg);reconstruction_args(s,cfg)
    l=sub.add_parser("list",help="Run a scene-list sequentially");l.add_argument("--split-file",required=True,type=Path);l.add_argument("--stage",choices=["all","reconstruct","raycast"],default="all");l.add_argument("--continue-on-error",action=argparse.BooleanOptionalAction,default=True);common(l,cfg);reconstruction_args(l,cfg)
    n=sub.add_parser("noise",help="Add noise to existing clean raycasts without reraycasting");n.add_argument("case");common(n,cfg)
    b=sub.add_parser("ablation",help="Generate noise ablations from existing clean raycasts");b.add_argument("case");common(b,cfg);b.add_argument("--profiles",nargs="+",default=cfg["raycast"]["noise"]["ablation_profiles"])
    a=ap.parse_args()
    if a.mode=="scene":run_scene(a,case_name(a.case),cfg);return
    if a.mode=="noise":add_noise(a,case_name(a.case),cfg);return
    if a.mode=="ablation":run_ablation(a,case_name(a.case),cfg);return
    cases=read_cases(a.split_file.expanduser().resolve());batch=a.dataset_root.expanduser().resolve()/"semantic_aware_surface_reconstruction"/"batch_runs"/f"hybrid_{a.split_file.stem}";batch.mkdir(parents=True,exist_ok=True);rows=[]
    if any((a.reconstruction_root,a.raycast_root,a.static_input,a.dynamic_input_root)):raise ValueError("List mode uses standard per-case paths; do not set per-scene path overrides.")
    for i,c in enumerate(cases,1):
        print(f"\n{'='*80}\n[{i}/{len(cases)}] {c}\n{'='*80}")
        try:rows.append(run_scene(a,c,cfg,batch/"logs"/c))
        except Exception as e:
            rows.append({"case":c,"status":"failed","error":repr(e)})
            if not a.continue_on_error:break
    (batch/"manifest.json").write_text(json.dumps({"split_file":str(a.split_file.resolve()),"scenes":rows},indent=2));print(f"Batch manifest: {batch/'manifest.json'}")

if __name__=="__main__":main()
