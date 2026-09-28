#!/usr/bin/env python3
"""
Hybrid semantic surface reconstruction for Waymo -> SCALA-2 raycasting.

Static scene routing
--------------------
18,19  ROAD/LANE          -> overlapping local explicit quadratic surfaces
20     OTHER_GROUND       -> overlapping local explicit quadratic surfaces
21,22  WALKABLE/SIDEWALK  -> overlapping local explicit quadratic surfaces
17     CURB               -> explicit boundary curves / vertical faces
other configured static semantics -> existing semantic PCL-MLS pipeline
static foreground instances       -> existing instance-aware PCL-MLS pipeline

Dynamic tracks
--------------
Dynamic objects remain object-local and are reconstructed by the existing
semantic PCL-MLS stage. The resulting dynamic_objects/<id>/mls_surface.npz
folder is copied into the final hybrid reconstruction root so the existing
raycaster can use it without modification.

The final reconstruction root uses the same static_manifest.json tile contract
and the same six required surface arrays:
    xyz, normal, intensity, semantic_id, ground_id, instance_id

This stage is trajectory-independent. The explicit ground surface model is
spatial (XY local patches), so the same code works for stationary, flat,
sloped, uphill/downhill, and moving-ego scenes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_closing
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


SURFACE_ARRAYS = ("xyz", "normal", "intensity", "semantic_id", "ground_id", "instance_id")
EXPLICIT_IDS = frozenset((17, 18, 19, 20, 21, 22))
SURFACE_TYPE = {
    "road_lane": 10,
    "other_ground": 11,
    "walkable_sidewalk": 12,
    "curb_boundary": 20,
    "mls_non_ground": 30,
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--caseid", required=True)
    ap.add_argument("--static-input", type=Path, default=None)
    ap.add_argument("--dynamic-input-root", type=Path, default=None)

    ap.add_argument("--hybrid-config", required=True, type=Path)
    ap.add_argument("--base-mls-config", required=True, type=Path)
    ap.add_argument("--mls-script", required=True, type=Path)
    ap.add_argument("--pcl-executable", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)

    ap.add_argument("--tile-size", type=float, default=25.0)
    ap.add_argument("--tile-halo", type=float, default=0.5)
    ap.add_argument("--minimum-label-confidence", type=float, default=0.66)
    ap.add_argument("--surface-workers", type=int, default=8)
    ap.add_argument("--pcl-threads", type=int, default=8)
    ap.add_argument("--mls-workers", type=int, default=3)
    ap.add_argument("--attribute-workers", type=int, default=1)
    ap.add_argument("--pcl-work-root", type=Path, default=None)
    ap.add_argument("--npz-compression", choices=["stored", "compressed"], default="stored")

    ap.add_argument("--include-dynamics", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--skip-mls", action="store_true", help="Diagnostic: build only explicit ground/curb surfaces.")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def save_npz(path: Path, arrays: dict[str, np.ndarray], compression: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "stored":
        np.savez(path, **arrays)
    elif compression == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError(compression)


def load_static_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        missing = sorted({"xyz", "intensity", "semantic_id", "ground_id", "instance_id"} - set(d.files))
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}")
        out = {
            "xyz": np.asarray(d["xyz"], dtype=np.float32),
            "intensity": np.asarray(d["intensity"], dtype=np.float32),
            "semantic_id": np.asarray(d["semantic_id"], dtype=np.int16),
            "ground_id": np.asarray(d["ground_id"], dtype=np.int8),
            "instance_id": np.asarray(d["instance_id"], dtype=np.int32),
        }
        for key, dtype in (
            ("label_confidence", np.float32),
            ("observation_frame_index", np.int32),
            ("is_generated", np.uint8),
            ("densified_is_generated", np.uint8),
            ("densification_source_type", np.int16),
        ):
            if key in d.files:
                out[key] = np.asarray(d[key], dtype=dtype)
    n = len(out["xyz"])
    if out["xyz"].shape != (n, 3):
        raise ValueError("xyz must have shape (N,3)")
    for k, v in out.items():
        if v.ndim >= 1 and len(v) != n:
            raise ValueError(f"Aligned-array mismatch: {k}")
    finite = np.isfinite(out["xyz"]).all(axis=1)
    if not np.all(finite):
        out = {k: (v[finite] if v.ndim >= 1 and len(v) == n else v) for k, v in out.items()}
    return out


def generated_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    n = len(data["xyz"])
    if "is_generated" in data:
        return np.asarray(data["is_generated"]).astype(bool)
    if "densified_is_generated" in data:
        return np.asarray(data["densified_is_generated"]).astype(bool)
    if "densification_source_type" in data:
        return np.asarray(data["densification_source_type"]) != 0
    return np.zeros(n, dtype=bool)


def family_source_mask(data, semantic_ids, minimum_confidence):
    sem = data["semantic_id"]
    mask = np.isin(sem, np.asarray(semantic_ids, dtype=np.int16))
    if "label_confidence" in data:
        mask &= data["label_confidence"] >= minimum_confidence
    return mask


def robust_quadratic_fit(u, v, z, base_weight, huber_delta, iterations):
    A = np.column_stack([np.ones(len(u)), u, v, u*u, u*v, v*v]).astype(np.float64)
    z = np.asarray(z, dtype=np.float64)
    w0 = np.asarray(base_weight, dtype=np.float64)
    w = w0.copy()
    coef = np.zeros(6, dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        root = np.sqrt(np.maximum(w, 1e-12))
        coef, *_ = np.linalg.lstsq(A * root[:, None], z * root, rcond=None)
        r = z - A @ coef
        ar = np.abs(r)
        huber = np.ones_like(ar)
        bad = ar > huber_delta
        huber[bad] = huber_delta / np.maximum(ar[bad], 1e-12)
        w = w0 * huber

    residual = z - A @ coef
    return coef, residual


def eval_quadratic(coef, u, v):
    a,b,c,d,e,f = coef
    z = a + b*u + c*v + d*u*u + e*u*v + f*v*v
    dzdu = b + 2.0*d*u + e*v
    dzdv = c + e*u + 2.0*f*v
    normal = np.column_stack([-dzdu, -dzdv, np.ones(len(u), dtype=np.float64)])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    flip = normal[:,2] < 0
    normal[flip] *= -1
    return z, normal


def patch_centers_from_points(xy, stride):
    qx = np.floor(xy[:,0] / stride).astype(np.int64)
    qy = np.floor(xy[:,1] / stride).astype(np.int64)
    pairs = np.column_stack([qx, qy])
    unique = np.unique(pairs, axis=0)
    centers = np.column_stack([
        (unique[:,0] + 0.5) * stride,
        (unique[:,1] + 0.5) * stride,
    ])
    return centers


def process_ground_patch(center, xyz, base_weight, source_tree, family_cfg):
    patch_size = float(family_cfg["patch_size_m"])
    half = 0.5 * patch_size
    support_cell = float(family_cfg["support_cell_m"])
    fine = float(family_cfg["sample_spacing_m"])

    radius = math.sqrt(2.0) * half + support_cell
    idx = np.asarray(source_tree.query_ball_point(center, radius), dtype=np.int64)
    if len(idx) < int(family_cfg["minimum_patch_points"]):
        return None

    p = xyz[idx].astype(np.float64, copy=False)
    u = p[:,0] - center[0]
    v = p[:,1] - center[1]
    inside = (np.abs(u) <= half) & (np.abs(v) <= half)
    if np.count_nonzero(inside) < int(family_cfg["minimum_patch_points"]):
        return None

    idx = idx[inside]
    u = u[inside]
    v = v[inside]
    z = xyz[idx,2].astype(np.float64, copy=False)
    bw = base_weight[idx]

    coef, residual = robust_quadratic_fit(
        u, v, z, bw,
        float(family_cfg["huber_delta_m"]),
        int(family_cfg["irls_iterations"]),
    )
    p95 = float(np.percentile(np.abs(residual), 95))
    if p95 > float(family_cfg["maximum_patch_p95_residual_m"]):
        return None

    # Support occupancy. Only sample where this semantic family is actually observed.
    edges = np.arange(-half, half + support_cell * 0.5, support_cell)
    if edges[-1] < half:
        edges = np.r_[edges, half]
    hist, u_edges, v_edges = np.histogram2d(u, v, bins=[edges, edges])
    occ = hist >= int(family_cfg["support_min_count"])
    iterations = int(family_cfg.get("support_close_iterations", 1))
    if iterations > 0:
        occ = binary_closing(occ, structure=np.ones((3,3), dtype=bool), iterations=iterations)

    fu = np.arange(-half, half + fine * 0.5, fine)
    fv = np.arange(-half, half + fine * 0.5, fine)
    U, V = np.meshgrid(fu, fv, indexing="ij")
    uf = U.ravel()
    vf = V.ravel()

    ui = np.floor((uf - u_edges[0]) / support_cell).astype(np.int64)
    vi = np.floor((vf - v_edges[0]) / support_cell).astype(np.int64)
    valid = (ui >= 0) & (ui < occ.shape[0]) & (vi >= 0) & (vi < occ.shape[1])
    keep = np.zeros(len(uf), dtype=bool)
    keep[valid] = occ[ui[valid], vi[valid]]
    if not np.any(keep):
        return None

    uf = uf[keep]
    vf = vf[keep]
    zf, nf = eval_quadratic(coef, uf, vf)
    xyf = np.column_stack([center[0] + uf, center[1] + vf])

    # Smooth raised-cosine blending in overlap.
    qu = np.clip(np.abs(uf) / max(half, 1e-9), 0.0, 1.0)
    qv = np.clip(np.abs(vf) / max(half, 1e-9), 0.0, 1.0)
    weight = (0.5 * (1.0 + np.cos(np.pi * qu))) * (0.5 * (1.0 + np.cos(np.pi * qv)))
    weight = np.maximum(weight, 1e-4)

    return {
        "xy": xyf,
        "z": zf,
        "normal": nf,
        "weight": weight,
        "p95_residual_m": p95,
        "source_points": int(len(idx)),
    }


def reduce_patch_samples(parts, fine):
    if not parts:
        return None

    xy = np.concatenate([p["xy"] for p in parts], axis=0)
    z = np.concatenate([p["z"] for p in parts], axis=0)
    normal = np.concatenate([p["normal"] for p in parts], axis=0)
    w = np.concatenate([p["weight"] for p in parts], axis=0)

    qx = np.rint(xy[:,0] / fine).astype(np.int64)
    qy = np.rint(xy[:,1] / fine).astype(np.int64)
    order = np.lexsort((qy, qx))
    qx = qx[order]
    qy = qy[order]
    z = z[order]
    normal = normal[order]
    w = w[order]

    start = np.r_[0, np.flatnonzero((np.diff(qx) != 0) | (np.diff(qy) != 0)) + 1]
    qx_u = qx[start]
    qy_u = qy[start]

    wsum = np.add.reduceat(w, start)
    zsum = np.add.reduceat(w * z, start)
    nx = np.add.reduceat(w * normal[:,0], start)
    ny = np.add.reduceat(w * normal[:,1], start)
    nz = np.add.reduceat(w * normal[:,2], start)

    xyz = np.column_stack([
        qx_u.astype(np.float64) * fine,
        qy_u.astype(np.float64) * fine,
        zsum / np.maximum(wsum, 1e-12),
    ])
    n = np.column_stack([nx,ny,nz])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    flip = n[:,2] < 0
    n[flip] *= -1
    return xyz, n


def build_explicit_ground_family(data, family_cfg, minimum_confidence, workers):
    name = family_cfg["name"]
    ids = family_cfg["semantic_ids"]
    mask = family_source_mask(data, ids, minimum_confidence)
    source_idx = np.flatnonzero(mask)
    if len(source_idx) < int(family_cfg["minimum_patch_points"]):
        return None, {"name": name, "status": "too_few_points", "source_points": int(len(source_idx))}

    xyz = data["xyz"][source_idx].astype(np.float64, copy=False)
    gen = generated_mask(data)[source_idx]
    generated_weight = float(family_cfg.get("generated_weight", 0.25))
    base_weight = np.where(gen, generated_weight, 1.0).astype(np.float64)

    source_tree = cKDTree(xyz[:,:2])
    centers = patch_centers_from_points(xyz[:,:2], float(family_cfg["patch_stride_m"]))
    parts = []
    skipped = 0

    def job(center):
        return process_ground_patch(center, xyz, base_weight, source_tree, family_cfg)

    if workers <= 1:
        for c in centers:
            p = job(c)
            if p is None:
                skipped += 1
            else:
                parts.append(p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(job, c) for c in centers]
            for fut in as_completed(futures):
                p = fut.result()
                if p is None:
                    skipped += 1
                else:
                    parts.append(p)

    reduced = reduce_patch_samples(parts, float(family_cfg["sample_spacing_m"]))
    if reduced is None:
        return None, {"name": name, "status": "no_valid_patches", "source_points": int(len(source_idx))}

    out_xyz, out_normal = reduced

    # Transfer semantic/intensity from nearest source point in XY.
    _, nearest = source_tree.query(out_xyz[:,:2], k=1, workers=-1)
    nearest = np.asarray(nearest, dtype=np.int64)
    src_global = source_idx[nearest]

    surface = {
        "xyz": out_xyz.astype(np.float32),
        "normal": out_normal.astype(np.float32),
        "intensity": data["intensity"][src_global].astype(np.float32, copy=False),
        "semantic_id": data["semantic_id"][src_global].astype(np.int16, copy=False),
        "ground_id": np.ones(len(out_xyz), dtype=np.int8),
        "instance_id": np.zeros(len(out_xyz), dtype=np.int32),
        "surface_type": np.full(len(out_xyz), SURFACE_TYPE[name], dtype=np.uint8),
    }
    report = {
        "name": name,
        "status": "completed",
        "semantic_ids": [int(x) for x in ids],
        "source_points": int(len(source_idx)),
        "patch_centers": int(len(centers)),
        "valid_patches": int(len(parts)),
        "skipped_patches": int(skipped),
        "output_points": int(len(out_xyz)),
        "patch_p95_residual_m": {
            "p50": float(np.percentile([p["p95_residual_m"] for p in parts], 50)),
            "p95": float(np.percentile([p["p95_residual_m"] for p in parts], 95)),
            "max": float(np.max([p["p95_residual_m"] for p in parts])),
        },
    }
    return surface, report


def voxelize_curb(data, mask, voxel):
    idx = np.flatnonzero(mask)
    xyz = data["xyz"][idx].astype(np.float64)
    intensity = data["intensity"][idx].astype(np.float64)
    qx = np.floor(xyz[:,0] / voxel).astype(np.int64)
    qy = np.floor(xyz[:,1] / voxel).astype(np.int64)
    order = np.lexsort((qy,qx))
    qx = qx[order]; qy = qy[order]
    xyz = xyz[order]; intensity = intensity[order]
    start = np.r_[0, np.flatnonzero((np.diff(qx)!=0)|(np.diff(qy)!=0))+1]

    count = np.diff(np.r_[start, len(xyz)])
    sx = np.add.reduceat(xyz[:,0], start) / count
    sy = np.add.reduceat(xyz[:,1], start) / count
    sz = np.add.reduceat(xyz[:,2], start) / count
    si = np.add.reduceat(intensity, start) / count
    return np.column_stack([sx,sy,sz]), si.astype(np.float32)


def extract_mst_chains(points_xy, link_radius, min_nodes, min_length):
    if len(points_xy) < min_nodes:
        return []
    tree = cKDTree(points_xy)
    pairs = tree.query_pairs(link_radius, output_type="ndarray")
    if len(pairs) == 0:
        return []
    dist = np.linalg.norm(points_xy[pairs[:,0]] - points_xy[pairs[:,1]], axis=1)
    n = len(points_xy)
    graph = csr_matrix((dist, (pairs[:,0], pairs[:,1])), shape=(n,n))
    graph = graph + graph.T
    ncomp, labels = connected_components(graph, directed=False)
    chains = []

    for comp in range(ncomp):
        nodes = np.flatnonzero(labels == comp)
        if len(nodes) < min_nodes:
            continue
        sub = graph[nodes][:,nodes]
        mst = minimum_spanning_tree(sub)
        mst = (mst + mst.T).tocsr()
        deg = np.diff(mst.indptr)
        anchors = np.flatnonzero(deg != 2)
        if len(anchors) < 2:
            continue

        visited = set()
        for a in anchors:
            neigh = mst.indices[mst.indptr[a]:mst.indptr[a+1]]
            for b in neigh:
                edge = (min(int(a),int(b)), max(int(a),int(b)))
                if edge in visited:
                    continue
                path = [int(a)]
                prev = int(a)
                cur = int(b)
                visited.add(edge)
                while True:
                    path.append(cur)
                    if deg[cur] != 2:
                        break
                    nn = mst.indices[mst.indptr[cur]:mst.indptr[cur+1]]
                    nxt = int(nn[0] if int(nn[1]) == prev else nn[1])
                    edge2 = (min(cur,nxt), max(cur,nxt))
                    if edge2 in visited:
                        break
                    visited.add(edge2)
                    prev, cur = cur, nxt

                global_nodes = nodes[np.asarray(path, dtype=np.int64)]
                if len(global_nodes) < 3:
                    continue
                seg = points_xy[global_nodes]
                length = float(np.sum(np.linalg.norm(np.diff(seg,axis=0),axis=1)))
                if length >= min_length:
                    chains.append(global_nodes)
    return chains


class GroundFamilySampler:
    def __init__(self, surfaces):
        self.entries = []
        for name, surface in surfaces.items():
            if surface is None or len(surface["xyz"]) == 0:
                continue
            self.entries.append({
                "name": name,
                "xyz": np.asarray(surface["xyz"], dtype=np.float64),
                "tree": cKDTree(np.asarray(surface["xyz"][:,:2], dtype=np.float64)),
                "semantic_id": surface["semantic_id"],
            })

    def query(self, xy, maximum_distance):
        n = len(xy)
        best_dist = np.full(n, np.inf, dtype=np.float64)
        best_z = np.full(n, np.nan, dtype=np.float64)
        best_sem = np.full(n, -1, dtype=np.int16)
        for entry in self.entries:
            dist, idx = entry["tree"].query(xy, k=1, workers=-1)
            take = dist < best_dist
            best_dist[take] = dist[take]
            best_z[take] = entry["xyz"][idx[take],2]
            best_sem[take] = entry["semantic_id"][idx[take]]
        valid = best_dist <= maximum_distance
        return best_z, best_sem, best_dist, valid


def reconstruct_curb(data, ground_surfaces, curb_cfg, minimum_confidence):
    mask = family_source_mask(data, [17], minimum_confidence)
    if np.count_nonzero(mask) < int(curb_cfg["minimum_component_nodes"]):
        return None, {"name":"curb_boundary","status":"too_few_points","source_points":int(np.count_nonzero(mask))}

    vox_xyz, vox_intensity = voxelize_curb(data, mask, float(curb_cfg["voxel_size_m"]))
    chains = extract_mst_chains(
        vox_xyz[:,:2],
        float(curb_cfg["link_radius_m"]),
        int(curb_cfg["minimum_component_nodes"]),
        float(curb_cfg["minimum_curve_length_m"]),
    )
    if not chains:
        return None, {"name":"curb_boundary","status":"no_valid_curves","source_points":int(np.count_nonzero(mask))}

    sampler = GroundFamilySampler(ground_surfaces)
    observed_tree = cKDTree(vox_xyz[:,:2])

    xyz_parts=[]; normal_parts=[]; intensity_parts=[]; sem_parts=[]; ground_parts=[]; inst_parts=[]
    curve_records=[]

    spacing = float(curb_cfg["curve_sample_spacing_m"])
    vertical_spacing = float(curb_cfg["vertical_sample_spacing_m"])
    probe = float(curb_cfg["side_probe_m"])
    query_max = float(curb_cfg["ground_query_max_distance_m"])
    min_h = float(curb_cfg["minimum_curb_height_m"])
    max_h = float(curb_cfg["maximum_curb_height_m"])

    for chain_id, chain in enumerate(chains):
        pts = vox_xyz[chain]
        ds = np.linalg.norm(np.diff(pts[:,:2],axis=0),axis=1)
        t = np.r_[0.0, np.cumsum(ds)]
        good = np.r_[True, np.diff(t) > 1e-6]
        pts = pts[good]; t = t[good]
        if len(t) < 3 or t[-1] < float(curb_cfg["minimum_curve_length_m"]):
            continue

        fx = PchipInterpolator(t, pts[:,0])
        fy = PchipInterpolator(t, pts[:,1])
        ts = np.arange(0.0, t[-1] + 0.5*spacing, spacing)
        cx = fx(ts); cy = fy(ts)
        dx = fx.derivative()(ts); dy = fy.derivative()(ts)
        dn = np.hypot(dx,dy)
        valid_t = dn > 1e-10
        if not np.any(valid_t):
            continue
        cx=cx[valid_t]; cy=cy[valid_t]; dx=dx[valid_t]/dn[valid_t]; dy=dy[valid_t]/dn[valid_t]
        center_xy = np.column_stack([cx,cy])
        left_n = np.column_stack([-dy,dx])

        plus = center_xy + probe*left_n
        minus = center_xy - probe*left_n
        zp, semp, dp, vp = sampler.query(plus, query_max)
        zm, semm, dm, vm = sampler.query(minus, query_max)

        _, near_obs = observed_tree.query(center_xy, k=1, workers=-1)
        zobs = vox_xyz[near_obs,2]
        iobs = vox_intensity[near_obs]

        out_count = 0
        for i in range(len(center_xy)):
            zlo = np.nan
            zhi = np.nan
            if vp[i] and vm[i]:
                lo = min(zp[i], zm[i]); hi = max(zp[i], zm[i])
                h = hi-lo
                if min_h <= h <= max_h:
                    zlo, zhi = lo, hi
            if not np.isfinite(zlo):
                candidates = []
                if vp[i]:
                    candidates.append(float(zp[i]))
                if vm[i]:
                    candidates.append(float(zm[i]))
                if candidates:
                    zg = min(candidates, key=lambda zz: abs(zz-float(zobs[i])))
                    h = abs(float(zobs[i])-zg)
                    if min_h <= h <= max_h:
                        zlo, zhi = min(zg,float(zobs[i])), max(zg,float(zobs[i]))

            if np.isfinite(zlo):
                zs = np.arange(zlo, zhi + 0.5*vertical_spacing, vertical_spacing)
                if len(zs) == 0:
                    zs = np.array([0.5*(zlo+zhi)])
            else:
                zs = np.array([float(zobs[i])])

            nn = np.array([left_n[i,0],left_n[i,1],0.0],dtype=np.float64)
            nn /= max(np.linalg.norm(nn),1e-12)
            xyz_parts.append(np.column_stack([
                np.full(len(zs),center_xy[i,0]),
                np.full(len(zs),center_xy[i,1]),
                zs,
            ]))
            normal_parts.append(np.repeat(nn[None,:],len(zs),axis=0))
            intensity_parts.append(np.full(len(zs),iobs[i],dtype=np.float32))
            sem_parts.append(np.full(len(zs),17,dtype=np.int16))
            ground_parts.append(np.ones(len(zs),dtype=np.int8))
            inst_parts.append(np.zeros(len(zs),dtype=np.int32))
            out_count += len(zs)

        curve_records.append({"chain_id":chain_id,"length_m":float(t[-1]),"samples":int(len(center_xy)),"output_points":int(out_count)})

    if not xyz_parts:
        return None, {"name":"curb_boundary","status":"no_output","source_points":int(np.count_nonzero(mask))}

    out_xyz=np.concatenate(xyz_parts).astype(np.float32)
    out_normal=np.concatenate(normal_parts).astype(np.float32)
    surface={
        "xyz":out_xyz,
        "normal":out_normal,
        "intensity":np.concatenate(intensity_parts).astype(np.float32),
        "semantic_id":np.concatenate(sem_parts).astype(np.int16),
        "ground_id":np.concatenate(ground_parts).astype(np.int8),
        "instance_id":np.concatenate(inst_parts).astype(np.int32),
        "surface_type":np.full(len(out_xyz),SURFACE_TYPE["curb_boundary"],dtype=np.uint8),
    }
    report={
        "name":"curb_boundary",
        "status":"completed",
        "source_points":int(np.count_nonzero(mask)),
        "voxelized_points":int(len(vox_xyz)),
        "curves":int(len(curve_records)),
        "output_points":int(len(out_xyz)),
        "curve_records":curve_records,
    }
    return surface, report


def make_non_ground_config(base_path: Path, output_path: Path):
    config = json.loads(base_path.read_text())
    new_groups=[]
    for group in config.get("background_groups",[]):
        g=copy.deepcopy(group)
        kept=[int(x) for x in g.get("semantic_ids",[]) if int(x) not in EXPLICIT_IDS]
        if kept:
            g["semantic_ids"]=kept
            new_groups.append(g)
    config["background_groups"]=new_groups
    config["configuration_name"]=str(config.get("configuration_name","mls"))+"_hybrid_non_ground"
    config.setdefault("notes",[])
    config["notes"]=[
        "Ground-like semantics 17-22 are reconstructed by the hybrid explicit-surface stage and removed from this MLS sub-stage."
    ]+list(config["notes"])
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(config,indent=2))
    return config


def run_non_ground_mls(args, temp_root, temp_config):
    stages=["background","static_objects"]
    if args.include_dynamics:
        stages.append("dynamic_objects")
    cmd=[
        os.environ.get("PYTHON","python"),
        str(args.mls_script.resolve()),
        "--dataset-root",str(args.dataset_root.resolve()),
        "--caseid",args.caseid,
        "--static-input",str(args.static_input.resolve()),
        "--pcl-executable",str(args.pcl_executable.resolve()),
        "--config",str(temp_config.resolve()),
        "--output-root",str(temp_root.resolve()),
        "--stages",*stages,
        "--tile-size",str(args.tile_size),
        "--tile-halo",str(args.tile_halo),
        "--pcl-threads",str(args.pcl_threads),
        "--mls-workers",str(args.mls_workers),
        "--attribute-workers",str(args.attribute_workers),
        "--minimum-label-confidence",str(args.minimum_label_confidence),
        "--no-exclude-points-in-tracked-boxes",
        "--npz-compression",args.npz_compression,
        "--overwrite",
    ]
    if args.dynamic_input_root is not None:
        cmd += ["--dynamic-input-root",str(args.dynamic_input_root.resolve())]
    if args.pcl_work_root is not None:
        cmd += ["--work-root",str(args.pcl_work_root.resolve())]

    env=os.environ.copy()
    env["OMP_NUM_THREADS"]=str(args.pcl_threads)
    env["OPENBLAS_NUM_THREADS"]="1"
    env["MKL_NUM_THREADS"]="1"
    env["NUMEXPR_NUM_THREADS"]="1"
    print("\nLaunching non-ground/static-object MLS:")
    print(" ".join(cmd))
    subprocess.run(cmd,check=True,env=env)


def tile_key_from_xy(xy, tile_size):
    ix=np.floor(xy[:,0]/tile_size).astype(np.int32)
    iy=np.floor(xy[:,1]/tile_size).astype(np.int32)
    return ix,iy


def group_surface_by_tile(surface, tile_size):
    result={}
    if surface is None or len(surface["xyz"])==0:
        return result
    ix,iy=tile_key_from_xy(surface["xyz"][:,:2],tile_size)
    pairs=np.column_stack([ix,iy])
    unique,inv=np.unique(pairs,axis=0,return_inverse=True)
    for j,(tx,ty) in enumerate(unique):
        rows=np.flatnonzero(inv==j)
        result[(int(tx),int(ty))]={k:v[rows] for k,v in surface.items()}
    return result


def load_mls_tile(path):
    with np.load(path,allow_pickle=False) as d:
        out={name:np.asarray(d[name]) for name in SURFACE_ARRAYS}
    out["surface_type"]=np.full(len(out["xyz"]),SURFACE_TYPE["mls_non_ground"],dtype=np.uint8)
    return out


def concatenate_surface_parts(parts):
    parts=[p for p in parts if p is not None and len(p["xyz"])]
    if not parts:
        return None
    keys=list(SURFACE_ARRAYS)+["surface_type"]
    return {k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}


def semantic_counts(ids):
    v,c=np.unique(ids,return_counts=True)
    return {str(int(a)):int(b) for a,b in zip(v,c)}


def merge_final_tiles(args, explicit_surfaces, curb_surface, mls_root):
    explicit_by_tile={}
    for surface in list(explicit_surfaces.values())+[curb_surface]:
        for key,part in group_surface_by_tile(surface,args.tile_size).items():
            explicit_by_tile.setdefault(key,[]).append(part)

    mls_manifest=None
    mls_tile_map={}
    if not args.skip_mls:
        mp=mls_root/"static_manifest.json"
        mls_manifest=json.loads(mp.read_text())
        for tile in mls_manifest["tiles"]:
            mls_tile_map[tuple(tile["tile_index"])]=tile

    all_keys=sorted(set(explicit_by_tile)|set(mls_tile_map))
    final_tiles=[]
    static_dir=args.output_root/"static_tiles"
    static_dir.mkdir(parents=True,exist_ok=True)

    for number,key in enumerate(all_keys,1):
        parts=[]
        parts.extend(explicit_by_tile.get(key,[]))
        if key in mls_tile_map:
            tile=mls_tile_map[key]
            parts.append(load_mls_tile(mls_root/tile["file"]))
        merged=concatenate_surface_parts(parts)
        if merged is None:
            continue
        tx,ty=key
        rel=Path("static_tiles")/f"tile_{tx}_{ty}.npz"
        save_npz(args.output_root/rel,merged,args.npz_compression)
        xyz=merged["xyz"]
        final_tiles.append({
            "tile_index":[tx,ty],
            "file":str(rel),
            "core_min_xy":[tx*args.tile_size,ty*args.tile_size],
            "core_max_xy":[(tx+1)*args.tile_size,(ty+1)*args.tile_size],
            "min_xyz":np.min(xyz,axis=0).astype(float).tolist(),
            "max_xyz":np.max(xyz,axis=0).astype(float).tolist(),
            "point_count":int(len(xyz)),
            "semantic_counts":semantic_counts(merged["semantic_id"]),
        })
        if number%20==0 or number==len(all_keys):
            print(f"Merge tiles {number}/{len(all_keys)}")

    return final_tiles, mls_manifest


def main():
    args=parse_args()
    args.dataset_root=args.dataset_root.resolve()
    args.hybrid_config=args.hybrid_config.resolve()
    args.base_mls_config=args.base_mls_config.resolve()
    args.mls_script=args.mls_script.resolve()
    args.pcl_executable=args.pcl_executable.resolve()
    args.output_root=args.output_root.resolve()
    args.static_input=(args.static_input.resolve() if args.static_input else args.dataset_root/"recon_related"/args.caseid/"static_recon_labels.npz")
    args.dynamic_input_root=args.dynamic_input_root.resolve() if args.dynamic_input_root else None
    args.pcl_work_root=args.pcl_work_root.resolve() if args.pcl_work_root else None

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} is not empty; use --overwrite")
    args.output_root.mkdir(parents=True,exist_ok=True)

    config=json.loads(args.hybrid_config.read_text())
    data=load_static_npz(args.static_input)
    started=time.time()
    timings={}

    print("="*80)
    print("HYBRID SEMANTIC SURFACE RECONSTRUCTION")
    print("="*80)
    print("Case        :",args.caseid)
    print("Static input:",args.static_input)
    print("Output      :",args.output_root)

    # 1) Explicit local ground-like surfaces.
    t=time.time()
    explicit_surfaces={}
    ground_reports=[]
    for family_cfg in config["ground_families"]:
        print(f"\nEXPLICIT GROUND FAMILY: {family_cfg['name']}")
        surface,report=build_explicit_ground_family(
            data,family_cfg,args.minimum_label_confidence,args.surface_workers
        )
        explicit_surfaces[family_cfg["name"]]=surface
        ground_reports.append(report)
        print(json.dumps({k:v for k,v in report.items() if k!="patch_p95_residual_m"},indent=2))
    timings["explicit_ground_s"]=time.time()-t

    # 2) Curb boundary reconstruction.
    t=time.time()
    print("\nCURB BOUNDARY RECONSTRUCTION")
    curb_surface,curb_report=reconstruct_curb(
        data,explicit_surfaces,config["curb"],args.minimum_label_confidence
    )
    print(json.dumps({k:v for k,v in curb_report.items() if k!="curve_records"},indent=2))
    timings["curb_s"]=time.time()-t

    # 3) Existing MLS for everything else + dynamic object-local models.
    temp_root=args.output_root/"_non_ground_mls_stage"
    temp_cfg=args.output_root/"_non_ground_mls_config.json"
    mls_manifest=None
    if not args.skip_mls:
        t=time.time()
        make_non_ground_config(args.base_mls_config,temp_cfg)
        run_non_ground_mls(args,temp_root,temp_cfg)
        timings["non_ground_mls_s"]=time.time()-t

    # 4) Merge into raycaster-compatible final tile store.
    t=time.time()
    final_tiles,mls_manifest=merge_final_tiles(args,explicit_surfaces,curb_surface,temp_root)
    timings["merge_tiles_s"]=time.time()-t

    # 5) Preserve object-local dynamic models exactly where current raycaster expects them.
    dynamic_models=0
    if not args.skip_mls and args.include_dynamics:
        src=temp_root/"dynamic_objects"
        if src.is_dir():
            dst=args.output_root/"dynamic_objects"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src,dst)
            dynamic_models=sum(1 for p in dst.iterdir() if p.is_dir() and (p/"mls_surface.npz").is_file())
        dm=temp_root/"dynamic_manifest.json"
        if dm.is_file():
            shutil.copy2(dm,args.output_root/"dynamic_manifest.json")

    manifest={
        "complete":True,
        "format_version":1,
        "case":args.caseid,
        "method":"hybrid_semantic_explicit_local_plus_pcl_mls",
        "coordinate_frame":"Waymo world",
        "surface_arrays":list(SURFACE_ARRAYS),
        "extra_surface_arrays":["surface_type"],
        "surface_type_codes":{str(v):k for k,v in SURFACE_TYPE.items()},
        "tile_size_m":args.tile_size,
        "tile_halo_m":args.tile_halo,
        "tiles":final_tiles,
        "counts":{
            "static_source_points":int(len(data["xyz"])),
            "static_output_points":int(sum(t["point_count"] for t in final_tiles)),
            "dynamic_object_models":int(dynamic_models),
            "explicit_ground_points":int(sum(len(s["xyz"]) for s in explicit_surfaces.values() if s is not None)),
            "curb_boundary_points":int(0 if curb_surface is None else len(curb_surface["xyz"])),
        },
        "hybrid_config":config,
    }
    (args.output_root/"static_manifest.json").write_text(json.dumps(manifest,indent=2))

    report={
        "case":args.caseid,
        "method":manifest["method"],
        "elapsed_seconds":time.time()-started,
        "stage_timings_seconds":timings,
        "ground_families":ground_reports,
        "curb":curb_report,
        "counts":manifest["counts"],
        "static_manifest":str(args.output_root/"static_manifest.json"),
    }
    (args.output_root/"hybrid_reconstruction_report.json").write_text(json.dumps(report,indent=2))

    if not args.keep_intermediate and temp_root.exists():
        shutil.rmtree(temp_root,ignore_errors=True)
    if not args.keep_intermediate and temp_cfg.exists():
        temp_cfg.unlink(missing_ok=True)

    print("\n"+"="*80)
    print("HYBRID RECONSTRUCTION COMPLETE")
    print("="*80)
    print(f"Static tiles          : {len(final_tiles)}")
    print(f"Static output points  : {manifest['counts']['static_output_points']:,}")
    print(f"Explicit ground       : {manifest['counts']['explicit_ground_points']:,}")
    print(f"Curb boundary         : {manifest['counts']['curb_boundary_points']:,}")
    print(f"Dynamic object models : {dynamic_models}")
    print(f"Output root           : {args.output_root}")
    print(f"Manifest              : {args.output_root/'static_manifest.json'}")


if __name__=="__main__":
    main()
