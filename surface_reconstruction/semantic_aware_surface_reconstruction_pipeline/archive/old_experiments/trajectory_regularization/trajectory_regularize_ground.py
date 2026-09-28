#!/usr/bin/env python3
"""Trajectory-guided post-densification ground regularization.

The ego trajectory is used only as a longitudinal coordinate / weak prior.
Actual LiDAR ground measurements determine the fitted surface.

Default behavior:
- regularize ROAD + LANE_MARKER only (semantic 18,19)
- prefer observed points as fitting anchors
- move generated/densified points only
- preserve all point-aligned NPZ arrays
- leave CURB untouched as a boundary

Output can be passed directly as --static-input to
reconstruct_semantic_static_mls.py.
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
    ap.add_argument("--profile-smooth-window-m", type=float, default=4.0)
    ap.add_argument("--cross-slope-smooth-window-m", type=float, default=5.0)
    ap.add_argument("--trajectory-prior-blend", type=float, default=0.25)
    ap.add_argument("--max-generated-shift-m", type=float, default=0.15)
    ap.add_argument("--regularize-observed", action="store_true")
    ap.add_argument("--max-observed-shift-m", type=float, default=0.03)
    ap.add_argument("--generated-field", default=None)
    ap.add_argument("--compression", choices=["stored", "compressed"], default="stored")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--max-plot-points", type=int, default=150000)
    return ap.parse_args()

def odd_window(meters, spacing, n, minimum=5):
    if n < 3: return 1
    w = max(minimum, int(round(meters / max(spacing, 1e-6))))
    if w % 2 == 0: w += 1
    if w > n: w = n if n % 2 else n - 1
    return max(1, w)

def smooth_1d(v, spacing, med_m, smooth_m):
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 5: return v.copy()
    mw = odd_window(med_m, spacing, len(v), 3)
    x = median_filter(v, size=mw, mode="nearest") if mw > 1 else v
    sw = odd_window(smooth_m, spacing, len(v), 5)
    return savgol_filter(x, sw, min(2, sw-1), mode="interp") if sw >= 5 else x

def load_trajectory(root, caseid, sample_m, med_m, smooth_m):
    p = root / "laser_calibrations" / caseid / "laser_calibrations" / "laser_calibrations.npz"
    if not p.is_file(): raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as d:
        poses = np.asarray(d["frame_pose"], dtype=np.float64)
    xyz = poses[:, :3, 3]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    s0, xyz0 = s[keep], xyz[keep]
    if len(s0) < 2: raise RuntimeError("Trajectory has insufficient XY motion.")
    spacing = float(np.median(np.diff(s0)))
    z0 = smooth_1d(xyz0[:, 2], spacing, med_m, smooth_m)
    sd = np.arange(s0[0], s0[-1] + 0.5*sample_m, sample_m)
    xd, yd, zd = [np.interp(sd, s0, arr) for arr in (xyz0[:,0], xyz0[:,1], z0)]
    dx, dy = np.gradient(xd, sd), np.gradient(yd, sd)
    n = np.hypot(dx, dy)
    good = n > 1e-8
    if not np.all(good):
        valid = np.flatnonzero(good)
        if not len(valid): raise RuntimeError("Cannot estimate trajectory tangent.")
        dx[~good] = np.interp(np.flatnonzero(~good), valid, dx[valid])
        dy[~good] = np.interp(np.flatnonzero(~good), valid, dy[valid])
        n = np.hypot(dx, dy)
    tx, ty = dx/n, dy/n
    return {"s": sd, "xyz": np.column_stack([xd,yd,zd]), "tx": tx, "ty": ty, "tree": cKDTree(np.column_stack([xd,yd]))}

def traj_coords(xyz, traj):
    _, idx = traj["tree"].query(np.asarray(xyz[:, :2], dtype=np.float64), k=1, workers=-1)
    ref = traj["xyz"][idx, :2]
    delta = xyz[:, :2] - ref
    tx, ty = traj["tx"][idx], traj["ty"][idx]
    d = delta[:,0]*(-ty) + delta[:,1]*tx
    return traj["s"][idx], d

def generated_mask(data, explicit=None):
    field = explicit
    if field is None:
        field = next((x for x in ("is_generated","densified_is_generated","densification_source_type") if x in data), None)
    if field is None: return np.zeros(len(data["xyz"]), dtype=bool), None
    if field not in data: raise KeyError(field)
    v = np.asarray(data[field])
    return ((v != 0) if field == "densification_source_type" else v.astype(bool)), field

def robust_fit(s, d, z, knot, args):
    m = (np.abs(s-knot) <= args.fit_half_window_m) & (np.abs(d) <= args.fit_lateral_max_m) & np.isfinite(z)
    idx = np.flatnonzero(m)
    if len(idx) < args.minimum_fit_points: return None
    if len(idx) > args.fit_max_points_per_knot:
        idx = idx[np.linspace(0, len(idx)-1, args.fit_max_points_per_knot).astype(np.int64)]
    ds, dd, zz = s[idx]-knot, d[idx], z[idx]
    A = np.column_stack([np.ones(len(idx)), ds, dd, dd*dd])
    w = np.ones(len(idx))
    coef = np.zeros(4)
    for _ in range(max(1,args.irls_iterations)):
        r = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A*r[:,None], zz*r, rcond=None)
        e = np.abs(zz - A@coef)
        w[:] = 1.0
        bad = e > args.huber_delta_m
        w[bad] = args.huber_delta_m / np.maximum(e[bad], 1e-12)
    res = np.abs(zz - A@coef)
    return coef, len(idx), float(np.median(res)), float(np.percentile(res,95))

def fill_missing(v):
    v = np.asarray(v, dtype=np.float64)
    good = np.isfinite(v)
    if not np.any(good): return v
    x = np.arange(len(v))
    return np.interp(x, x[good], v[good])

def fit_surface(fit_xyz, s, d, traj, args):
    knots = np.arange(max(s.min(), traj["s"][0]), min(s.max(), traj["s"][-1]) + 0.25*args.fit_knot_spacing_m, args.fit_knot_spacing_m)
    c0 = np.full(len(knots), np.nan); cd = c0.copy(); cdd = c0.copy(); cs = c0.copy(); nfit = np.zeros(len(knots), np.int32)
    med = c0.copy(); p95 = c0.copy()
    for i,k in enumerate(knots):
        out = robust_fit(s,d,fit_xyz[:,2],k,args)
        if out is None: continue
        coef, n, m, q = out
        c0[i], cs[i], cd[i], cdd[i] = coef
        nfit[i], med[i], p95[i] = n,m,q
    if np.count_nonzero(np.isfinite(c0)) < 5:
        raise RuntimeError("Too few valid surface knots. Increase --fit-half-window-m or lower --minimum-fit-points.")
    c0i, cdi, cddi = map(fill_missing, (c0,cd,cdd))
    ego_z = np.interp(knots, traj["s"], traj["xyz"][:,2])
    pw = odd_window(args.profile_smooth_window_m, args.fit_knot_spacing_m, len(knots), 5)
    cw = odd_window(args.cross_slope_smooth_window_m, args.fit_knot_spacing_m, len(knots), 5)
    direct = savgol_filter(c0i,pw,2,mode="interp") if pw>=5 else c0i
    offset = c0i - ego_z
    offset = savgol_filter(offset,pw,2,mode="interp") if pw>=5 else offset
    prior = ego_z + offset
    a = float(np.clip(args.trajectory_prior_blend,0,1))
    c0f = (1-a)*direct + a*prior
    cdf = savgol_filter(cdi,cw,2,mode="interp") if cw>=5 else cdi
    cddf = savgol_filter(cddi,cw,2,mode="interp") if cw>=5 else cddi
    return {"knots":knots,"c0_raw":c0,"c0":c0f,"cd":cdf,"cdd":cddf,"ego_z":ego_z,"nfit":nfit,"med":med,"p95":p95}

def eval_surface(sf, s, d):
    c0 = np.interp(s,sf["knots"],sf["c0"])
    cd = np.interp(s,sf["knots"],sf["cd"])
    cdd = np.interp(s,sf["knots"],sf["cdd"])
    return c0 + cd*d + cdd*d*d

def stats(v):
    v = np.asarray(v,dtype=np.float64); v=v[np.isfinite(v)]
    if not len(v): return {"count":0}
    return {"count":int(len(v)),"mean":float(v.mean()),"p50":float(np.percentile(v,50)),"p90":float(np.percentile(v,90)),"p95":float(np.percentile(v,95)),"p99":float(np.percentile(v,99)),"max":float(v.max())}

def make_plots(report_dir, family, sf, residual, shift):
    report_dir.mkdir(parents=True, exist_ok=True)
    k = sf["knots"]
    fig,ax=plt.subplots(figsize=(12,6)); ax.plot(k,sf["ego_z"],label="smoothed ego z"); ax.plot(k,sf["c0"],label="regularized center profile"); g=np.isfinite(sf["c0_raw"]); ax.scatter(k[g],sf["c0_raw"][g],s=8,alpha=.4,label="local robust fits"); ax.set_xlabel("trajectory arc length s [m]"); ax.set_ylabel("world z [m]"); ax.grid(True); ax.legend(); ax.set_title(f"{family}: longitudinal profile"); fig.tight_layout(); fig.savefig(report_dir/f"{family}_01_profile.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6)); ax.plot(k,100*np.gradient(sf["c0"],k),label="regularized road grade"); ax.plot(k,100*np.gradient(sf["ego_z"],k),label="ego trajectory grade"); ax.axhline(0,lw=1); ax.set_xlabel("s [m]"); ax.set_ylabel("grade [%]"); ax.grid(True); ax.legend(); ax.set_title(f"{family}: grade profile"); fig.tight_layout(); fig.savefig(report_dir/f"{family}_02_grade.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6)); ax.hist(np.abs(residual[np.isfinite(residual)]),bins=120); ax.set_xlabel("|input z - fitted surface z| [m]"); ax.set_ylabel("count"); ax.grid(True); ax.set_title(f"{family}: residual to fitted surface"); fig.tight_layout(); fig.savefig(report_dir/f"{family}_03_residual.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6)); ax.hist(np.abs(shift[np.isfinite(shift)]),bins=120); ax.set_xlabel("|applied z shift| [m]"); ax.set_ylabel("count"); ax.grid(True); ax.set_title(f"{family}: applied regularization shift"); fig.tight_layout(); fig.savefig(report_dir/f"{family}_04_shift.png",dpi=180); plt.close(fig)

def main():
    args=parse_args(); root=args.dataset_root.resolve(); case=args.caseid
    inp=(args.input_npz.resolve() if args.input_npz else root/"recon_related"/case/"static_recon_labels.npz")
    out=(args.output_npz.resolve() if args.output_npz else root/"recon_related"/case/"static_recon_labels_trajectory_regularized.npz")
    report=(args.report_dir.resolve() if args.report_dir else root/"recon_related"/case/"trajectory_ground_regularization")
    if not inp.is_file(): raise FileNotFoundError(inp)
    with np.load(inp,allow_pickle=False) as d: data={k:np.asarray(d[k]) for k in d.files}
    xyz=np.asarray(data["xyz"],dtype=np.float64).copy(); sem=np.asarray(data["semantic_id"],dtype=np.int16)
    gen, gen_field=generated_mask(data,args.generated_field)
    traj=load_trajectory(root,case,args.trajectory_sample_m,args.trajectory_z_median_window_m,args.trajectory_z_smooth_window_m)
    moved=np.zeros(len(xyz),dtype=np.uint8); fitted_z=np.full(len(xyz),np.nan,np.float32); applied=np.zeros(len(xyz),np.float32)
    rep={"caseid":case,"input_npz":str(inp),"output_npz":str(out),"generated_field":gen_field,"trajectory_length_m":float(traj["s"][-1]),"families":{}}
    print(f"Trajectory length: {traj['s'][-1]:.2f} m")
    for family in args.families:
        idx=np.flatnonzero(np.isin(sem,FAMILIES[family]));
        if not len(idx): continue
        p=xyz[idx]; s,d=traj_coords(p,traj); eligible=np.abs(d)<=args.fit_lateral_max_m
        obs=~gen[idx]; fitmask=eligible & obs
        source="observed only"
        if np.count_nonzero(fitmask) < args.minimum_fit_points*5:
            fitmask=eligible; source="all eligible points"
        sf=fit_surface(p[fitmask],s[fitmask],d[fitmask],traj,args)
        zfit=eval_surface(sf,s,d); residual=p[:,2]-zfit
        movemask=eligible & gen[idx]
        if args.regularize_observed: movemask |= eligible & (~gen[idx])
        maxshift=np.where(gen[idx],args.max_generated_shift_m,args.max_observed_shift_m)
        shift=np.clip(zfit-p[:,2],-maxshift,maxshift); shift[~movemask]=0
        xyz[idx,2]+=shift; moved[idx[movemask]]=1; fitted_z[idx]=zfit.astype(np.float32); applied[idx]=shift.astype(np.float32)
        rep["families"][family]={"semantic_ids":FAMILIES[family].tolist(),"points":int(len(idx)),"eligible":int(np.count_nonzero(eligible)),"fit_points":int(np.count_nonzero(fitmask)),"fit_source":source,"moved":int(np.count_nonzero(movemask)),"input_abs_residual_m":stats(np.abs(residual[eligible])),"applied_abs_shift_m":stats(np.abs(shift[movemask])),"valid_knot_fraction":float(np.mean(np.isfinite(sf["c0_raw"]))),"grade_abs_percent":stats(np.abs(100*np.gradient(sf["c0"],sf["knots"]))) }
        make_plots(report,family,sf,residual,shift)
        print(f"[{family}] points={len(idx):,} fit={np.count_nonzero(fitmask):,} moved={np.count_nonzero(movemask):,} residual_p95={rep['families'][family]['input_abs_residual_m']['p95']:.3f} m shift_p95={rep['families'][family]['applied_abs_shift_m']['p95']:.3f} m")
    data["xyz"]=xyz.astype(np.float32); data["trajectory_regularized"]=moved; data["trajectory_fitted_ground_z"]=fitted_z; data["trajectory_applied_z_shift_m"]=applied
    out.parent.mkdir(parents=True,exist_ok=True)
    (np.savez_compressed if args.compression=="compressed" else np.savez)(out,**data)
    report.mkdir(parents=True,exist_ok=True); (report/"trajectory_ground_regularization_report.json").write_text(json.dumps(rep,indent=2))
    print(f"Output: {out}\nReport: {report}")

if __name__=="__main__": main()