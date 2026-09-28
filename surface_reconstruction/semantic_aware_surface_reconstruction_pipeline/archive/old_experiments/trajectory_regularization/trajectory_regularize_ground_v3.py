#!/usr/bin/env python3
"""Trajectory-guided post-densification ground regularization (v3).

Conservative design:
- observed ROAD/LANE points define the fitted surface;
- ego trajectory is only a longitudinal coordinate + weak boundary prior;
- two-pass robust fitting excludes observed outliers from the second fit;
- endpoint taper + grade cap suppress nonphysical fitted-profile end spikes;
- only trustworthy generated points are projected;
- generated points not explained by the model are ALWAYS KEPT at original XYZ;
- the first/last projection-end-margin metres are never projected because
  finite-trajectory nearest-point coordinates collapse scene geometry onto s=0
  or s=trajectory_length near the endpoints;
- observed LiDAR points remain unchanged unless --regularize-observed is passed.

Important: surface residual is an eligibility test for regularization, NOT a
criterion for deleting ROAD/LANE geometry.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

FAMILIES = {
    "road_lane": np.array([18, 19], dtype=np.int16),
    "other_ground": np.array([20], dtype=np.int16),
    "walkable_sidewalk": np.array([21, 22], dtype=np.int16),
}
STATUS = {
    0: "untouched_or_not_selected",
    1: "generated_projected",
    2: "generated_unexplained_moderate_kept",
    3: "generated_unexplained_severe_kept",
    4: "observed_trusted_unchanged",
    5: "observed_fit_outlier_unchanged",
    6: "observed_regularized_optional",
    7: "generated_endpoint_guard_unchanged",
}

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--caseid", required=True)
    ap.add_argument("--input-npz", type=Path, default=None)
    ap.add_argument("--output-npz", type=Path, default=None)
    ap.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=["road_lane"])
    ap.add_argument("--trajectory-sample-m", type=float, default=0.20)
    ap.add_argument("--trajectory-z-median-window-m", type=float, default=3.0)
    ap.add_argument("--trajectory-z-smooth-window-m", type=float, default=5.0)
    ap.add_argument("--fit-knot-spacing-m", type=float, default=0.50)
    ap.add_argument("--fit-half-window-m", type=float, default=0.75)
    ap.add_argument("--fit-lateral-max-m", type=float, default=12.0)
    ap.add_argument("--fit-max-points-per-knot", type=int, default=12000)
    ap.add_argument("--minimum-fit-points", type=int, default=80)
    ap.add_argument("--huber-delta-m", type=float, default=0.08)
    ap.add_argument("--irls-iterations", type=int, default=4)
    ap.add_argument("--refit-observed-max-residual-m", type=float, default=0.20)
    ap.add_argument("--profile-smooth-window-m", type=float, default=4.0)
    ap.add_argument("--cross-slope-smooth-window-m", type=float, default=5.0)
    ap.add_argument("--trajectory-prior-blend", type=float, default=0.20)
    ap.add_argument("--edge-taper-m", type=float, default=3.0)
    ap.add_argument("--max-abs-grade-percent", type=float, default=45.0)
    ap.add_argument("--max-abs-cross-slope-percent", type=float, default=20.0)
    ap.add_argument("--generated-project-max-residual-m", type=float, default=0.20)
    ap.add_argument("--generated-severe-residual-m", type=float, default=0.50)
    ap.add_argument("--max-generated-shift-m", type=float, default=0.20)
    ap.add_argument(
        "--projection-end-margin-m",
        type=float,
        default=2.0,
        help="Never project generated points within this arc-length margin of either finite trajectory endpoint.",
    )
    ap.add_argument(
        "--keep-rejected-generated",
        action="store_true",
        help="Deprecated V2 compatibility flag. V3 always keeps unexplained generated points.",
    )
    ap.add_argument("--regularize-observed", action="store_true")
    ap.add_argument("--observed-project-max-residual-m", type=float, default=0.08)
    ap.add_argument("--max-observed-shift-m", type=float, default=0.03)
    ap.add_argument("--generated-field", default=None)
    ap.add_argument("--compression", choices=["stored", "compressed"], default="stored")
    ap.add_argument("--report-dir", type=Path, default=None)
    return ap.parse_args()

def odd_window(meters, spacing, n, minimum=5):
    if n < 3: return 1
    w = max(minimum, int(round(meters / max(spacing, 1e-9))))
    if w % 2 == 0: w += 1
    if w > n: w = n if n % 2 else n - 1
    return max(1, w)

def smooth_signal(v, spacing, median_m, smooth_m):
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 5: return v.copy()
    mw = odd_window(median_m, spacing, len(v), 3)
    x = median_filter(v, size=mw, mode="nearest") if mw > 1 else v.copy()
    sw = odd_window(smooth_m, spacing, len(v), 5)
    if sw >= 5: x = savgol_filter(x, sw, 2, mode="interp")
    return x

def load_trajectory(root, caseid, sample_m, median_m, smooth_m):
    path = root / "laser_calibrations" / caseid / "laser_calibrations" / "laser_calibrations.npz"
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d: poses = np.asarray(d["frame_pose"], dtype=np.float64)
    ego = poses[:, :3, 3]
    s0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ego[:, :2], axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(s0) > 1e-6]); s0 = s0[keep]; ego = ego[keep]
    if len(s0) < 2 or s0[-1] < 1.0: raise RuntimeError("Insufficient XY ego motion.")
    z0 = smooth_signal(ego[:, 2], float(np.median(np.diff(s0))), median_m, smooth_m)
    s = np.arange(s0[0], s0[-1] + 0.5 * sample_m, sample_m)
    x = np.interp(s, s0, ego[:, 0]); y = np.interp(s, s0, ego[:, 1]); z = np.interp(s, s0, z0)
    dx = np.gradient(x, s); dy = np.gradient(y, s); n = np.hypot(dx, dy)
    bad = n < 1e-8
    if np.any(bad):
        good = np.flatnonzero(~bad)
        if not len(good): raise RuntimeError("Could not estimate trajectory tangent.")
        for i in np.flatnonzero(bad):
            j = good[np.argmin(np.abs(good - i))]; dx[i] = dx[j]; dy[i] = dy[j]
        n = np.hypot(dx, dy)
    tx = dx / n; ty = dy / n
    xyz = np.column_stack([x, y, z])
    return {"s": s, "xyz": xyz, "tx": tx, "ty": ty, "tree": cKDTree(xyz[:, :2])}

def trajectory_coords(xyz, traj):
    _, idx = traj["tree"].query(np.asarray(xyz[:, :2], dtype=np.float64), k=1, workers=-1)
    delta = xyz[:, :2] - traj["xyz"][idx, :2]; tx = traj["tx"][idx]; ty = traj["ty"][idx]
    d = delta[:, 0] * (-ty) + delta[:, 1] * tx
    return traj["s"][idx], d

def detect_generated(data, explicit):
    if explicit:
        if explicit not in data: raise KeyError(explicit)
        field = explicit
    else:
        field = next((k for k in ("is_generated", "densified_is_generated", "densification_source_type") if k in data), None)
    if field is None: return np.zeros(len(data["xyz"]), dtype=bool), None
    v = np.asarray(data[field]); return ((v != 0) if field == "densification_source_type" else v.astype(bool)), field

def robust_local_fit(s, d, z, knot, candidate, args):
    m = candidate & np.isfinite(s) & np.isfinite(d) & np.isfinite(z) & (np.abs(s-knot) <= args.fit_half_window_m) & (np.abs(d) <= args.fit_lateral_max_m)
    idx = np.flatnonzero(m)
    if len(idx) < args.minimum_fit_points: return None
    if len(idx) > args.fit_max_points_per_knot:
        idx = idx[np.linspace(0, len(idx)-1, args.fit_max_points_per_knot).astype(np.int64)]
    ss = s[idx]-knot; dd = d[idx]; zz = z[idx]
    A = np.column_stack([np.ones(len(idx)), ss, dd])
    w = np.ones(len(idx))
    for _ in range(max(1, args.irls_iterations)):
        rw = np.sqrt(w); coef, *_ = np.linalg.lstsq(A*rw[:,None], zz*rw, rcond=None)
        r = zz - A@coef; a = np.abs(r); w = np.ones_like(a); q = a > args.huber_delta_m; w[q] = args.huber_delta_m / np.maximum(a[q], 1e-12)
    r = zz - A@coef
    return coef, len(idx), float(np.median(np.abs(r))), float(np.percentile(np.abs(r),95))

def fill_missing(v):
    v = np.asarray(v, dtype=np.float64); g = np.isfinite(v)
    if not np.any(g): return v.copy()
    x = np.arange(len(v)); return np.interp(x, x[g], v[g])

def edge_guard_offset(offset, knots, edge_m):
    out = np.asarray(offset, dtype=np.float64).copy()
    if edge_m <= 0: return out
    s0, s1 = knots[0], knots[-1]
    lm = (knots >= s0+edge_m) & (knots <= s0+2*edge_m)
    rm = (knots <= s1-edge_m) & (knots >= s1-2*edge_m)
    if np.any(lm): out[knots < s0+edge_m] = np.median(out[lm])
    if np.any(rm): out[knots > s1-edge_m] = np.median(out[rm])
    return out

def edge_weight(knots, edge_m):
    if edge_m <= 0: return np.ones(len(knots))
    l = np.clip((knots-knots[0])/edge_m,0,1); r = np.clip((knots[-1]-knots)/edge_m,0,1); w = np.minimum(l,r)
    return w*w*(3-2*w)

def cap_grade(z, s, max_grade_percent):
    z = np.asarray(z, dtype=np.float64).copy(); g = max_grade_percent/100.0
    for _ in range(3):
        for i in range(1,len(z)):
            lim = g*(s[i]-s[i-1]); z[i] = np.clip(z[i], z[i-1]-lim, z[i-1]+lim)
        for i in range(len(z)-2,-1,-1):
            lim = g*(s[i+1]-s[i]); z[i] = np.clip(z[i], z[i+1]-lim, z[i+1]+lim)
    return z

def fit_surface(s,d,z,observed,args,traj):
    knots = np.arange(max(np.min(s),traj["s"][0]), min(np.max(s),traj["s"][-1])+0.5*args.fit_knot_spacing_m, args.fit_knot_spacing_m)
    def run(mask):
        c0=np.full(len(knots),np.nan); cd=np.full(len(knots),np.nan); cnt=np.zeros(len(knots),dtype=np.int64); med=np.full(len(knots),np.nan); p95=np.full(len(knots),np.nan)
        for i,k in enumerate(knots):
            q=robust_local_fit(s,d,z,k,mask,args)
            if q is not None: (coef,cnt[i],med[i],p95[i])=q; c0[i]=coef[0]; cd[i]=coef[2]
        if np.count_nonzero(np.isfinite(c0))<5: raise RuntimeError("Too few valid road-profile knots.")
        return {"c0_raw":c0,"cd_raw":cd,"count":cnt,"fit_med":med,"fit_p95":p95}
    def finalize(raw):
        c0=fill_missing(raw["c0_raw"]); cd=fill_missing(raw["cd_raw"]); ego=np.interp(knots,traj["s"],traj["xyz"][:,2])
        pw=odd_window(args.profile_smooth_window_m,args.fit_knot_spacing_m,len(knots),5); cw=odd_window(args.cross_slope_smooth_window_m,args.fit_knot_spacing_m,len(knots),5)
        direct=savgol_filter(c0,pw,2,mode="interp") if pw>=5 else c0.copy()
        off=c0-ego; off=median_filter(off,size=odd_window(args.profile_smooth_window_m,args.fit_knot_spacing_m,len(knots),3),mode="nearest")
        if pw>=5: off=savgol_filter(off,pw,2,mode="interp")
        off=edge_guard_offset(off,knots,args.edge_taper_m); prior=ego+off
        b=np.clip(args.trajectory_prior_blend,0,1); interior=(1-b)*direct+b*prior; ew=edge_weight(knots,args.edge_taper_m)
        center=ew*interior+(1-ew)*prior; center=cap_grade(center,knots,args.max_abs_grade_percent)
        cross=savgol_filter(cd,cw,2,mode="interp") if cw>=5 else cd.copy(); cross=np.clip(cross,-args.max_abs_cross_slope_percent/100,args.max_abs_cross_slope_percent/100)
        return {**raw,"knots":knots,"c0":center,"cd":cross,"ego_z":ego,"prior":prior,"edge_weight":ew}
    s1=finalize(run(observed)); z1=evaluate(s1,s,d); r1=z-z1; trusted=observed&(np.abs(r1)<=args.refit_observed_max_residual_m)
    if np.count_nonzero(trusted) < max(args.minimum_fit_points*10, int(0.15*np.count_nonzero(observed))): return s1, observed, False
    return finalize(run(trusted)), trusted, True

def evaluate(surface,s,d):
    return np.interp(s,surface["knots"],surface["c0"]) + np.interp(s,surface["knots"],surface["cd"])*d

def st(v):
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return {"count":0}
    return {"count":int(len(v)),"mean":float(np.mean(v)),"p50":float(np.percentile(v,50)),"p90":float(np.percentile(v,90)),"p95":float(np.percentile(v,95)),"p99":float(np.percentile(v,99)),"max":float(np.max(v))}

def plots(report_dir,family,surface,s,residual,status,shift,args):
    report_dir.mkdir(parents=True,exist_ok=True); k=surface["knots"]
    fig,ax=plt.subplots(figsize=(12,6)); ax.plot(k,surface["ego_z"],label="smoothed ego z"); ax.plot(k,surface["prior"],label="ego-relative road prior"); ax.plot(k,surface["c0"],label="regularized center profile"); g=np.isfinite(surface["c0_raw"]); ax.scatter(k[g],surface["c0_raw"][g],s=7,alpha=.35,label="pass-2 local robust fits"); ax.set(xlabel="trajectory arc length s [m]",ylabel="world z [m]",title=f"{family}: longitudinal profile"); ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(report_dir/f"{family}_01_longitudinal_profile.png",dpi=180); plt.close(fig)
    grade=100*np.gradient(surface["c0"],k); eg=100*np.gradient(surface["ego_z"],k); fig,ax=plt.subplots(figsize=(12,6)); ax.plot(k,grade,label="regularized road grade"); ax.plot(k,eg,label="ego trajectory grade"); ax.axhline(args.max_abs_grade_percent,ls="--",label="grade cap"); ax.axhline(-args.max_abs_grade_percent,ls="--"); ax.set(xlabel="s [m]",ylabel="grade [%]",title=f"{family}: grade profile"); ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(report_dir/f"{family}_02_grade_profile.png",dpi=180); plt.close(fig)
    r=np.abs(residual[np.isfinite(residual)]); hi=np.percentile(r,99.5) if len(r) else 1; fig,ax=plt.subplots(figsize=(12,6)); ax.hist(r[r<=hi],bins=120); ax.axvline(args.refit_observed_max_residual_m,ls="--",label="observed refit threshold"); ax.axvline(args.generated_project_max_residual_m,ls="--",label="generated projection threshold"); ax.set(xlabel="|input z - fitted surface z| [m]",ylabel="count",title=f"{family}: residual to fitted surface (trimmed p99.5)"); ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(report_dir/f"{family}_03_residual_histogram.png",dpi=180); plt.close(fig)
    moved=np.abs(shift)>0; fig,ax=plt.subplots(figsize=(12,6)); ax.hist(np.abs(shift[moved]),bins=100); ax.set(xlabel="|applied z shift| [m]",ylabel="count",title=f"{family}: applied shift"); ax.grid(); fig.tight_layout(); fig.savefig(report_dir/f"{family}_04_applied_shift_histogram.png",dpi=180); plt.close(fig)
    edges=np.arange(np.min(s),np.max(s)+1,1); centers=.5*(edges[:-1]+edges[1:]); bi=np.digitize(s,edges)-1; fig,ax=plt.subplots(figsize=(12,6));
    for code in (1,2,3,4,5,6):
        c=np.array([np.count_nonzero((bi==i)&(status==code)) for i in range(len(centers))]);
        if np.any(c): ax.plot(centers,c,label=f"{code}: {STATUS[code]}")
    ax.set(xlabel="s [m]",ylabel="points / 1 m",title=f"{family}: regularization status along trajectory"); ax.grid(); ax.legend(); fig.tight_layout(); fig.savefig(report_dir/f"{family}_05_status_vs_s.png",dpi=180); plt.close(fig)

def main():
    args=parse_args(); root=args.dataset_root.resolve(); case=args.caseid
    inp=args.input_npz.resolve() if args.input_npz else root/"recon_related"/case/"static_recon_labels.npz"
    out=args.output_npz.resolve() if args.output_npz else root/"recon_related"/case/"static_recon_labels_trajectory_regularized_v3.npz"
    report=args.report_dir.resolve() if args.report_dir else root/"recon_related"/case/"trajectory_ground_regularization_v3"
    if not inp.is_file(): raise FileNotFoundError(inp)
    with np.load(inp,allow_pickle=False) as d: data={k:np.asarray(d[k]) for k in d.files}
    xyz=np.asarray(data["xyz"],dtype=np.float64).copy(); sem=np.asarray(data["semantic_id"],dtype=np.int16); n=len(xyz)
    generated,gen_field=detect_generated(data,args.generated_field); traj=load_trajectory(root,case,args.trajectory_sample_m,args.trajectory_z_median_window_m,args.trajectory_z_smooth_window_m)
    status=np.zeros(n,dtype=np.uint8); fitted=np.full(n,np.nan,dtype=np.float32); residual_all=np.full(n,np.nan,dtype=np.float32); shift_all=np.zeros(n,dtype=np.float32); reports={}
    print("="*78); print("TRAJECTORY-GUIDED GROUND REGULARIZATION V3"); print("="*78); print("Input:",inp); print("Generated field:",gen_field); print(f"Trajectory length: {traj['s'][-1]:.2f} m")
    for family in args.families:
        ids=FAMILIES[family]; gi=np.flatnonzero(np.isin(sem,ids)); fam=xyz[gi]; s,d=trajectory_coords(fam,traj); corridor=np.abs(d)<=args.fit_lateral_max_m; obs=~generated[gi]; fitmask=corridor&obs
        surface,trusted_fit,refit=fit_surface(s,d,fam[:,2],fitmask,args,traj); zsurf=evaluate(surface,s,d); residual=fam[:,2]-zsurf; ar=np.abs(residual)
        trusted_obs=corridor&obs&(ar<=args.refit_observed_max_residual_m)
        obs_out=corridor&obs&~trusted_obs
        gen_corr=corridor&generated[gi]

        s_min=float(traj["s"][0])+args.projection_end_margin_m
        s_max=float(traj["s"][-1])-args.projection_end_margin_m
        if s_max <= s_min:
            raise ValueError(
                f"--projection-end-margin-m={args.projection_end_margin_m} leaves no projection interior "
                f"for trajectory length {traj['s'][-1]-traj['s'][0]:.3f} m"
            )
        projection_interior=(s>=s_min)&(s<=s_max)
        gen_endpoint=gen_corr&~projection_interior
        gen_model=gen_corr&projection_interior

        gen_proj=gen_model&(ar<=args.generated_project_max_residual_m)
        gen_mod=gen_model&(ar>args.generated_project_max_residual_m)&(ar<=args.generated_severe_residual_m)
        gen_sev=gen_model&(ar>args.generated_severe_residual_m)

        fs=np.zeros(len(gi),dtype=np.uint8)
        fs[gen_proj]=1
        fs[gen_mod]=2
        fs[gen_sev]=3
        fs[trusted_obs]=4
        fs[obs_out]=5
        fs[gen_endpoint]=7

        shift=np.zeros(len(gi))
        shift[gen_proj]=np.clip(
            zsurf[gen_proj]-fam[gen_proj,2],
            -args.max_generated_shift_m,
            args.max_generated_shift_m,
        )
        if args.regularize_observed:
            om=trusted_obs&(ar<=args.observed_project_max_residual_m); shift[om]=np.clip(zsurf[om]-fam[om,2],-args.max_observed_shift_m,args.max_observed_shift_m); fs[om]=6
        xyz[gi,2]+=shift
        status[gi]=fs; fitted[gi]=zsurf.astype(np.float32); residual_all[gi]=residual.astype(np.float32); shift_all[gi]=shift.astype(np.float32)
        grade=100*np.gradient(surface["c0"],surface["knots"]); reports[family]={"semantic_ids":ids.tolist(),"points_total":int(len(gi)),"corridor_points":int(np.count_nonzero(corridor)),"observed_points":int(np.count_nonzero(obs)),"generated_points":int(np.count_nonzero(generated[gi])),"trusted_observed":int(np.count_nonzero(trusted_obs)),"observed_fit_outliers":int(np.count_nonzero(obs_out)),"generated_projected":int(np.count_nonzero(gen_proj)),"generated_unexplained_moderate_kept":int(np.count_nonzero(gen_mod)),"generated_unexplained_severe_kept":int(np.count_nonzero(gen_sev)),"generated_endpoint_guard_unchanged":int(np.count_nonzero(gen_endpoint)),"generated_dropped":0,"projection_interior_s_m":[float(s_min),float(s_max)],"refit_used":bool(refit),"residual_abs_m_all_corridor":st(ar[corridor]),"residual_abs_m_trusted_observed":st(ar[trusted_obs]),"residual_abs_m_generated_before_projection":st(ar[gen_corr]),"residual_abs_m_generated_projection_interior":st(ar[gen_model]),"applied_abs_shift_m":st(np.abs(shift[np.abs(shift)>0])),"grade_percent_abs":st(np.abs(grade)),"cross_slope_percent_abs":st(100*np.abs(surface["cd"])),"valid_knot_fraction":float(np.mean(np.isfinite(surface["c0_raw"]))) }
        plots(report,family,surface,s,residual,fs,shift,args)
        print(
            f"[{family}] projected generated={np.count_nonzero(gen_proj):,}, "
            f"unexplained kept={np.count_nonzero(gen_mod|gen_sev):,}, "
            f"endpoint-guard unchanged={np.count_nonzero(gen_endpoint):,}, "
            f"observed outliers={np.count_nonzero(obs_out):,}"
        )
        print(f"[{family}] grade abs p95={reports[family]['grade_percent_abs']['p95']:.2f}% max={reports[family]['grade_percent_abs']['max']:.2f}%")
    output={}
    for k,v in data.items(): output[k]=v
    output["xyz"]=xyz.astype(np.float32)
    output["trajectory_source_index"]=np.arange(n,dtype=np.int64)
    output["trajectory_regularization_status"]=status
    output["trajectory_fitted_ground_z"]=fitted
    output["trajectory_surface_residual_before_m"]=residual_all
    output["trajectory_applied_z_shift_m"]=shift_all
    out.parent.mkdir(parents=True,exist_ok=True); (np.savez_compressed if args.compression=="compressed" else np.savez)(out,**output)
    report.mkdir(parents=True,exist_ok=True); params={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}; summary={"caseid":case,"input_npz":str(inp),"output_npz":str(out),"input_points":int(n),"output_points":int(n),"dropped_generated_points":0,"generated_field":gen_field,"trajectory_length_m":float(traj["s"][-1]),"projection_end_margin_m":float(args.projection_end_margin_m),"status_codes":{str(k):v for k,v in STATUS.items()},"families":reports,"parameters":params}; rp=report/"trajectory_ground_regularization_v3_report.json"; rp.write_text(json.dumps(summary,indent=2))
    print("DONE"); print("Output:",out); print("Dropped generated points: 0 (V3 always preserves unexplained geometry)"); print("Report:",rp)

if __name__=="__main__": main()