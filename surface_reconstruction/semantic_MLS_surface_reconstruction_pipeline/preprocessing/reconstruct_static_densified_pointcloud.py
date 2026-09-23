#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import time
import zipfile
from pathlib import Path

from numpy.lib import format as npy_format

import numpy as np
from scipy.spatial import cKDTree

GROUND_FAMILIES = {1: (18, 19), 2: (20,), 3: (21, 22)}
FAMILY_DEFAULT_SEMANTIC = {1: 18, 2: 20, 3: 22}
GROUND_SEMANTICS = sorted({s for vals in GROUND_FAMILIES.values() for s in vals})
CURB_SEMANTIC = 17

_SEMANTIC_FAMILY_LUT = np.zeros(256, dtype=np.int8)
for _family_id, _semantic_ids in GROUND_FAMILIES.items():
    for _semantic_id in _semantic_ids:
        _SEMANTIC_FAMILY_LUT[int(_semantic_id)] = int(_family_id)


def semantic_family(sem):
    """Vectorized Waymo semantic -> ground-family mapping."""
    values = np.asarray(sem)
    out = np.zeros(values.shape, dtype=np.int8)
    valid = (values >= 0) & (values < len(_SEMANTIC_FAMILY_LUT))
    out[valid] = _SEMANTIC_FAMILY_LUT[values[valid].astype(np.int64, copy=False)]
    return out


def voxel_centroids(points, labels, voxel):
    # voxel <= 0 means: use the complete point cloud exactly as provided.
    if voxel <= 0:
        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(labels, dtype=np.int16),
        )

    xyzs, sems = [], []
    for sid in np.unique(labels):
        pts = points[labels == sid]
        if len(pts) == 0:
            continue
        keys = np.floor(pts / voxel).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inv).astype(np.float64)
        c = np.column_stack([np.bincount(inv, weights=pts[:, d], minlength=len(counts)) / counts for d in range(3)]).astype(np.float32)
        xyzs.append(c); sems.append(np.full(len(c), int(sid), dtype=np.int16))
    if not xyzs:
        return np.empty((0, 3), np.float32), np.empty((0,), np.int16)
    return np.concatenate(xyzs, axis=0), np.concatenate(sems, axis=0)


def load_sensor_origins(root, case):
    meta_path = os.path.join(root, "meta_infos", case + ".pkl")
    cal_path = os.path.join(root, "laser_calibrations", case, "laser_calibrations", "laser_calibrations.npz")
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    with np.load(cal_path, allow_pickle=False) as d:
        top_to_vehicle = np.asarray(d["extrinsic"][0], dtype=np.float64)
    origins = []
    for frame in data["frames"]:
        vehicle_to_world = np.asarray(frame.get("optimized_pose", frame["lidar2world"]), dtype=np.float64)
        origins.append((vehicle_to_world @ top_to_vehicle)[:3, 3])
    return np.asarray(origins, dtype=np.float64)


def trajectory_stats(origins):
    center = np.median(origins, axis=0)
    dxy = np.linalg.norm(origins[:, :2] - center[None, :2], axis=1)
    return {
        "center": center,
        "p95_xy_spread_m": float(np.percentile(dxy, 95)),
        "max_xy_spread_m": float(dxy.max(initial=0.0)),
        "path_length_xy_m": float(np.linalg.norm(np.diff(origins[:, :2], axis=0), axis=1).sum()) if len(origins) > 1 else 0.0,
    }


def _robust_along_ring_spacing(phi_values, radius_m, duplicate_threshold_m=0.003,
                               min_unique_samples=3):
    """Estimate physical sampling spacing ALONG one detected ring run.

    Important:
      - This does NOT voxelize/downsample the reconstruction cloud.
      - It is used only to estimate the target density for NEW support.
      - Near-identical repeated temporal observations are collapsed in 1-D
        along-ring arc length so stationary-frame repetition does not masquerade
        as true sub-millimetre spatial sampling.

    The ring samples are ordered by azimuth. Their tangential arc coordinates
    are approximately s = r * phi. Near-identical s values are clustered, and
    spacing is measured between consecutive unique spatial samples.
    """
    phi = np.asarray(phi_values, dtype=np.float64)
    if len(phi) < 2 or radius_m <= 1e-9:
        return 0.0, int(len(phi)), 0

    s = np.sort(radius_m * phi)

    # Collapse only nearly identical temporal repeats. This is NOT a voxel grid.
    if duplicate_threshold_m > 0 and len(s) > 1:
        cuts = np.flatnonzero(np.diff(s) > duplicate_threshold_m) + 1
        groups = np.split(s, cuts)
        unique_s = np.asarray([np.median(g) for g in groups], dtype=np.float64)
    else:
        unique_s = s

    if len(unique_s) < max(2, int(min_unique_samples)):
        return 0.0, int(len(unique_s)), max(0, len(unique_s) - 1)

    ds = np.diff(unique_s)
    ds = ds[np.isfinite(ds) & (ds > max(1e-9, duplicate_threshold_m))]
    if len(ds) == 0:
        return 0.0, int(len(unique_s)), 0

    # Missed rays/occlusions appear as large integer-multiple gaps. Estimate the
    # base sampling from the lower part of the spacing distribution and remove
    # large missing-sample gaps before taking the final robust median.
    base = float(np.percentile(ds, 30))
    upper = max(2.5 * base, base + 2.0 * max(duplicate_threshold_m, 1e-6))
    good = ds[ds <= upper]
    if len(good) == 0:
        good = ds

    spacing = float(np.median(good))
    return spacing, int(len(unique_s)), int(len(good))


def radial_runs_with_z(r_values, z_values, phi_values, split_gap, min_points,
                       density_duplicate_threshold_m=0.003,
                       density_min_unique_samples=3):
    if len(r_values) == 0:
        return []

    order = np.argsort(r_values)
    r = np.asarray(r_values, dtype=np.float64)[order]
    z = np.asarray(z_values, dtype=np.float64)[order]
    phi = np.asarray(phi_values, dtype=np.float64)[order]

    cuts = np.flatnonzero(np.diff(r) > split_gap) + 1
    groups = np.split(np.arange(len(r)), cuts)
    out = []

    for g in groups:
        if len(g) < min_points:
            continue

        rr, zz, pp = r[g], z[g], phi[g]
        r_med = float(np.median(rr))

        along_spacing, unique_count, spacing_samples = _robust_along_ring_spacing(
            pp, r_med,
            duplicate_threshold_m=density_duplicate_threshold_m,
            min_unique_samples=density_min_unique_samples,
        )

        out.append({
            "r_min": float(rr.min()),
            "r_max": float(rr.max()),
            "r": r_med,
            "z": float(np.median(zz)),
            "z_raw": float(np.median(zz)),
            "count": int(len(g)),
            "z_p10": float(np.percentile(zz, 10)),
            "z_p90": float(np.percentile(zz, 90)),
            "along_spacing_m": float(along_spacing),
            "along_unique_samples": int(unique_count),
            "along_spacing_samples": int(spacing_samples),
        })

    return out



def _local_run_sep(runs, j):
    vals = []
    if j > 0:
        vals.append(runs[j]["r"] - runs[j - 1]["r"])
    if j + 1 < len(runs):
        vals.append(runs[j + 1]["r"] - runs[j]["r"])
    vals = [v for v in vals if v > 1e-6]
    return min(vals) if vals else 0.5


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, 1.4826 * mad


def clean_ring_anchors(runs_by_sector, nsec, loo_floor_m=0.03, loo_sigma=6.0,
                       neighbor_half_window=1, neighbor_z_tol_m=0.05, neighbor_min_matches=1):
    """Reject isolated bad ring-height anchors without changing valid anchor Z.

    This method deliberately does NOT smooth Z across azimuth. Each accepted anchor keeps
    the median Z measured in its own azimuth sector.

    A ring is first marked as a radial leave-one-out outlier when it is a local
    maximum of the residual to the line through the previous/next rings. That
    candidate is rejected only when neighbouring sectors also disagree in Z.
    If neighbouring support is unavailable, only an extreme radial outlier is
    rejected. This preserves coherent real road dips/slopes that repeat across
    adjacent azimuths while removing isolated semantic/range artifacts.
    """
    total_anchors = 0
    candidate_spikes = 0
    neighbor_confirmed_rejected = 0
    isolated_extreme_rejected = 0

    # Compute same-sector leave-one-out residuals but do not reject yet.
    for s, runs in runs_by_sector.items():
        total_anchors += len(runs)
        residual = np.full(len(runs), np.nan, dtype=np.float64)
        for i in range(1, len(runs) - 1):
            a, b, c = runs[i - 1], runs[i], runs[i + 1]
            den = c["r"] - a["r"]
            if den <= 1e-6:
                continue
            t = (b["r"] - a["r"]) / den
            pred = a["z_raw"] + t * (c["z_raw"] - a["z_raw"])
            residual[i] = abs(b["z_raw"] - pred)

        finite = residual[np.isfinite(residual)]
        med, scale = _robust_scale(finite)
        threshold = max(float(loo_floor_m), med + float(loo_sigma) * scale)

        for i, run in enumerate(runs):
            run["valid"] = True
            run["z"] = run["z_raw"]  # no azimuthal Z smoothing
            run["loo_residual_m"] = float(residual[i]) if np.isfinite(residual[i]) else 0.0
            run["loo_threshold_m"] = float(threshold)
            run["neighbor_dz_m"] = 0.0
            run["neighbor_matches"] = 0
            run["loo_candidate"] = False
            run["reject_reason"] = ""

        for i in range(1, len(runs) - 1):
            ri = residual[i]
            if not np.isfinite(ri) or ri <= threshold:
                continue
            rprev = residual[i - 1] if np.isfinite(residual[i - 1]) else -np.inf
            rnext = residual[i + 1] if np.isfinite(residual[i + 1]) else -np.inf
            if ri >= rprev and ri >= rnext:
                runs[i]["loo_candidate"] = True
                candidate_spikes += 1

    # Validate LOO candidates against nearby sectors. Neighbouring sectors are
    # evidence only; they never change the current sector's measured Z.
    for s, runs in runs_by_sector.items():
        for j, run in enumerate(runs):
            if not run["loo_candidate"]:
                continue

            sep = _local_run_sep(runs, j)
            radius_tol = float(np.clip(0.40 * sep, 0.06, 1.25))
            neighbor_z = []

            if neighbor_half_window > 0:
                for ds in range(-neighbor_half_window, neighbor_half_window + 1):
                    if ds == 0:
                        continue
                    nruns = runs_by_sector.get((s + ds) % nsec)
                    if not nruns:
                        continue
                    q = min(nruns, key=lambda x: abs(x["r"] - run["r"]))
                    if abs(q["r"] - run["r"]) <= radius_tol:
                        neighbor_z.append(q["z_raw"])

            run["neighbor_matches"] = int(len(neighbor_z))

            if len(neighbor_z) >= neighbor_min_matches:
                dz = abs(run["z_raw"] - float(np.median(neighbor_z)))
                run["neighbor_dz_m"] = float(dz)
                if dz > neighbor_z_tol_m:
                    run["valid"] = False
                    run["reject_reason"] = "loo_plus_neighbor"
                    neighbor_confirmed_rejected += 1
            else:
                # No cross-azimuth evidence: reject only a very strong isolated
                # spike. This avoids deleting genuine local road curvature.
                extreme_threshold = max(0.10, 2.0 * run["loo_threshold_m"])
                if run["loo_residual_m"] > extreme_threshold:
                    run["valid"] = False
                    run["reject_reason"] = "loo_extreme_no_neighbor"
                    isolated_extreme_rejected += 1

    cleaned = {}
    for s, runs in runs_by_sector.items():
        good = [q for q in runs if q["valid"]]
        if len(good) >= 2:
            cleaned[s] = good

    total_rejected = neighbor_confirmed_rejected + isolated_extreme_rejected
    meta = {
        "total_anchors": int(total_anchors),
        "loo_candidates": int(candidate_spikes),
        "neighbor_confirmed_rejected": int(neighbor_confirmed_rejected),
        "isolated_extreme_rejected": int(isolated_extreme_rejected),
        "total_rejected": int(total_rejected),
        "loo_floor_m": float(loo_floor_m),
        "loo_sigma": float(loo_sigma),
        "neighbor_half_window": int(neighbor_half_window),
        "neighbor_z_tol_m": float(neighbor_z_tol_m),
        "neighbor_min_matches": int(neighbor_min_matches),
    }
    return cleaned, meta


def build_ring_gap_intervals(raw_points, origin_xy, sector_deg, split_gap, fill_spacing, profile_bin_m,
                             max_gap_factor, min_points_per_run, min_sector_points,
                             max_abs_grade_percent, loo_floor_m, loo_sigma,
                             neighbor_half_window, neighbor_z_tol_m, neighbor_min_matches,
                             spacing_mode="fixed", adaptive_scale=1.0,
                             adaptive_min=0.01, adaptive_max=0.08,
                             density_duplicate_threshold_m=0.003,
                             density_min_unique_samples=3):
    """Detect ring gaps using clean same-sector measured Z anchors.

    Key behaviour:
      - no azimuthal smoothing/overwriting of ring Z;
      - isolated radial Z spikes are rejected by leave-one-out consistency;
      - neighbouring sectors only validate anchors;
      - support Z is interpolated between clean anchors from this exact sector.
    """
    if len(raw_points) < 20:
        return {}, {"sectors": 0, "accepted_gaps": 0, "profile_bins": {}, "anchor_cleaning": {}}

    dx = raw_points[:, 0] - origin_xy[0]
    dy = raw_points[:, 1] - origin_xy[1]
    r = np.hypot(dx, dy)
    phi = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    dphi = math.radians(sector_deg)
    nsec = max(1, int(math.ceil(2 * math.pi / dphi)))
    sid = np.minimum((phi / dphi).astype(np.int32), nsec - 1)

    # Group sectors once. The previous implementation evaluated ``sid == s``
    # for every one of ~720 sectors, repeatedly scanning the full family cloud.
    # Sorting once is exactly equivalent because radial_runs_with_z sorts by
    # radius internally and therefore does not depend on original point order.
    raw_runs_by_sector = {}
    order = np.argsort(sid, kind="stable")
    sid_sorted = sid[order]
    unique_sector, starts, counts = np.unique(
        sid_sorted, return_index=True, return_counts=True
    )
    for s, begin, count in zip(unique_sector, starts, counts):
        if int(count) < min_sector_points:
            continue
        idx = order[begin:begin + count]
        runs = radial_runs_with_z(
            r[idx], raw_points[idx, 2], phi[idx], split_gap, min_points_per_run,
            density_duplicate_threshold_m=density_duplicate_threshold_m,
            density_min_unique_samples=density_min_unique_samples,
        )
        if len(runs) >= 2:
            raw_runs_by_sector[int(s)] = runs

    runs_by_sector, clean_meta = clean_ring_anchors(
        raw_runs_by_sector, nsec,
        loo_floor_m=loo_floor_m, loo_sigma=loo_sigma,
        neighbor_half_window=neighbor_half_window,
        neighbor_z_tol_m=neighbor_z_tol_m,
        neighbor_min_matches=neighbor_min_matches)

    def pair_fill_spacing(a, b):
        if spacing_mode != "along_ring_density":
            return float(fill_spacing)

        values = np.asarray(
            [a.get("along_spacing_m", 0.0), b.get("along_spacing_m", 0.0)],
            dtype=np.float64,
        )
        values = values[np.isfinite(values) & (values > 0)]
        measured = float(np.median(values)) if len(values) else float(fill_spacing)
        return float(np.clip(adaptive_scale * measured, adaptive_min, adaptive_max))

    raw_records = []
    grade_rejected = 0
    for s, runs in runs_by_sector.items():
        for a, b in zip(runs[:-1], runs[1:]):
            target_spacing = pair_fill_spacing(a, b)
            gap = b["r_min"] - a["r_max"]
            if gap <= 1.15 * target_spacing:
                continue
            dr_anchor = b["r"] - a["r"]
            if dr_anchor <= 1e-6:
                continue
            grade = 100.0 * (b["z"] - a["z"]) / dr_anchor
            if abs(grade) > max_abs_grade_percent:
                grade_rejected += 1
                continue
            raw_records.append({
                "sector": s, "inner": a["r_max"], "outer": b["r_min"],
                "mid": 0.5 * (a["r_max"] + b["r_min"]), "gap": gap,
                "r0": a["r"], "z0": a["z"], "r1": b["r"], "z1": b["z"],
                "grade_percent": grade,
                "inner_along_spacing_m": float(a.get("along_spacing_m", 0.0)),
                "outer_along_spacing_m": float(b.get("along_spacing_m", 0.0)),
                "inner_unique_samples": int(a.get("along_unique_samples", 0)),
                "outer_unique_samples": int(b.get("along_unique_samples", 0)),
                "target_fill_spacing_m": float(target_spacing),
                "anchor_loo_max_m": max(a.get("loo_residual_m", 0.0), b.get("loo_residual_m", 0.0)),
                "anchor_neighbor_max_m": max(a.get("neighbor_dz_m", 0.0), b.get("neighbor_dz_m", 0.0)),
            })

    if not raw_records:
        return {}, {
            "sectors": len(runs_by_sector), "accepted_gaps": 0, "profile_bins": {},
            "anchor_cleaning": clean_meta, "grade_rejected": int(grade_rejected)}

    mids = np.asarray([x["mid"] for x in raw_records], dtype=np.float64)
    gaps = np.asarray([x["gap"] for x in raw_records], dtype=np.float64)
    radial_bin = np.floor(mids / profile_bin_m).astype(np.int32)
    profile = {}
    global_expected = float(np.percentile(gaps, 30))
    for b in np.unique(radial_bin):
        g = gaps[radial_bin == b]
        if len(g) >= 6:
            profile[int(b)] = float(np.percentile(g, 30))

    accepted, accepted_gaps, grades = {}, [], []
    gap_ratio_rejected = 0
    for rec in raw_records:
        b = int(math.floor(rec["mid"] / profile_bin_m))
        expected = max(profile.get(b, global_expected), rec.get("target_fill_spacing_m", fill_spacing))
        ratio = rec["gap"] / expected
        if ratio > max_gap_factor:
            gap_ratio_rejected += 1
            continue
        rec = dict(rec)
        rec["expected"] = expected
        rec["ratio"] = ratio
        accepted.setdefault(rec["sector"], []).append(rec)
        accepted_gaps.append(rec["gap"])
        grades.append(rec["grade_percent"])

    meta = {
        "sectors_raw": len(raw_runs_by_sector),
        "sectors_clean": len(runs_by_sector),
        "raw_gaps": len(raw_records),
        "accepted_gaps": int(sum(len(v) for v in accepted.values())),
        "grade_rejected": int(grade_rejected),
        "gap_ratio_rejected": int(gap_ratio_rejected),
        "median_accepted_gap_m": float(np.median(accepted_gaps)) if accepted_gaps else 0.0,
        "global_expected_gap_m": global_expected,
        "anchor_grade_p50_percent": float(np.median(np.abs(grades))) if grades else 0.0,
        "anchor_grade_p95_percent": float(np.percentile(np.abs(grades), 95)) if grades else 0.0,
        "profile_bins": {str(k): v for k, v in sorted(profile.items())},
        "anchor_cleaning": clean_meta,
        "spacing_mode": spacing_mode,
        "density_duplicate_threshold_m": float(density_duplicate_threshold_m),
        "density_min_unique_samples": int(density_min_unique_samples),
    }
    return accepted, meta


def _z_from_interval(rec, radius):
    if radius < rec["r0"] or radius > rec["r1"]:
        return None
    t = (radius - rec["r0"]) / max(rec["r1"] - rec["r0"], 1e-9)
    return rec["z0"] + t * (rec["z1"] - rec["z0"])


def _neighbor_interval_z(intervals, sector, radius):
    recs = intervals.get(sector)
    if not recs:
        return None
    # Choose the interval that actually brackets this radius.
    for rec in recs:
        z = _z_from_interval(rec, radius)
        if z is not None:
            return z
    return None


def _neighbor_interval_z_many(intervals, sector, radii):
    """Vectorized equivalent of repeated _neighbor_interval_z calls."""
    radii = np.asarray(radii, dtype=np.float64)
    out = np.full(radii.shape, np.nan, dtype=np.float64)
    recs = intervals.get(sector)
    if not recs or len(radii) == 0:
        return out
    remaining = np.ones(radii.shape, dtype=bool)
    for rec in recs:
        mask = remaining & (radii >= rec["r0"]) & (radii <= rec["r1"])
        if not np.any(mask):
            continue
        denom = max(rec["r1"] - rec["r0"], 1e-9)
        t = (radii[mask] - rec["r0"]) / denom
        out[mask] = rec["z0"] + t * (rec["z1"] - rec["z0"])
        remaining[mask] = False
        if not np.any(remaining):
            break
    return out


def build_local_ring_spacing_estimator(points_xy, density_voxel=0.01, nn_neighbors=3, query_neighbors=16):
    """Build a robust local spacing estimator from observed ring points.

    Temporal accumulation can place many nearly identical observations on top of
    one another. Therefore density is estimated only after a fine XY voxel
    deduplication. For each retained observed point we estimate its local
    point-to-point spacing from its nearest spatial neighbours.

    Returns a dictionary containing the reference points, their local spacing,
    and a cKDTree used to query spacing around the two rings bracketing a gap.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) < 3:
        return None

    if density_voxel > 0:
        keys = np.floor(points_xy / density_voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        reference = points_xy[np.sort(first)]
    else:
        reference = points_xy

    if len(reference) < 3:
        return None

    tree = cKDTree(reference)
    k = min(max(2, int(nn_neighbors) + 1), len(reference))
    distances, _ = tree.query(reference, k=k, workers=-1)
    if distances.ndim == 1:
        distances = distances[:, None]

    neighbour_distances = distances[:, 1:]
    neighbour_distances = np.where(neighbour_distances > 1e-9, neighbour_distances, np.nan)
    local_spacing = np.nanmedian(neighbour_distances, axis=1)

    finite = np.isfinite(local_spacing) & (local_spacing > 0)
    if not np.any(finite):
        return None

    fallback = float(np.median(local_spacing[finite]))
    local_spacing[~finite] = fallback

    return {
        "points_xy": reference,
        "tree": tree,
        "local_spacing": local_spacing.astype(np.float32),
        "query_neighbors": max(1, int(query_neighbors)),
        "density_voxel": float(density_voxel),
        "global_median_spacing": fallback,
    }


def estimate_gap_fill_spacing(estimator, origin_xy, sector, sector_deg, r0, r1,
                              fixed_spacing, scale, minimum, maximum):
    """Estimate target spacing from the two observed rings surrounding one gap.

    We query the local observed density at the inner-ring anchor and outer-ring
    anchor. Their robust median spacing becomes the target spacing for the
    interpolated surface between them.
    """
    if estimator is None:
        return float(np.clip(fixed_spacing, minimum, maximum))

    dphi = math.radians(sector_deg)
    phi = (float(sector) + 0.5) * dphi

    anchors = np.asarray([
        [origin_xy[0] + r0 * math.cos(phi), origin_xy[1] + r0 * math.sin(phi)],
        [origin_xy[0] + r1 * math.cos(phi), origin_xy[1] + r1 * math.sin(phi)],
    ], dtype=np.float64)

    k = min(estimator["query_neighbors"], len(estimator["points_xy"]))
    _, idx = estimator["tree"].query(anchors, k=k, workers=-1)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]

    values = estimator["local_spacing"][idx].reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values):
        measured = float(np.median(values))
    else:
        measured = float(estimator["global_median_spacing"])

    return float(np.clip(scale * measured, minimum, maximum))


def ring_candidates_direct_z(intervals, origin_xy, sector_deg, fill_spacing, max_candidates,
                             spacing_mode="fixed", spacing_estimator=None,
                             adaptive_scale=1.0, adaptive_min=0.01, adaptive_max=0.08,
                             radial_spacing_scale=1.0):
    """Generate support with exact same-sector ring interpolation.

    In fixed mode, every accepted gap uses ``fill_spacing``.

    In local_ring_density mode, every gap gets its own target spacing estimated
    from the observed point density on the two rings that bracket that gap.
    Dense observed rings therefore receive dense interpolated support, while
    naturally sparse regions remain sparser.

    Radial slope comes directly from the bracketing rings. Tangential slope is
    estimated, when possible, from accepted neighbouring-sector surfaces at the
    same radius.
    """
    if not intervals:
        empty3 = np.empty((0, 3), np.float32)
        empty1 = np.empty((0,), np.float32)
        return empty3, empty3.copy(), *(empty1.copy() for _ in range(13))

    dphi = math.radians(sector_deg)
    nsec = max(1, int(math.ceil(2 * math.pi / dphi)))
    xyzs, normals = [], []
    gaps, expected, ratios, r0s, z0s, r1s, z1s, ts, grades, loo_maxs, neigh_maxs, used_spacings, radial_spacings = ([] for _ in range(13))
    count = 0

    for s, recs in intervals.items():
        phi0 = s * dphi
        phi1 = min((s + 1) * dphi, 2 * math.pi)

        for rec in recs:
            if spacing_mode == "along_ring_density":
                tangential_spacing = float(np.clip(
                    rec.get("target_fill_spacing_m", fill_spacing),
                    adaptive_min, adaptive_max
                ))
            elif spacing_mode == "local_ring_density":
                tangential_spacing = estimate_gap_fill_spacing(
                    spacing_estimator, origin_xy, s, sector_deg,
                    rec["r0"], rec["r1"], fill_spacing,
                    adaptive_scale, adaptive_min, adaptive_max
                )
            else:
                tangential_spacing = float(fill_spacing)

            radial_spacing = float(np.clip(
                radial_spacing_scale * tangential_spacing,
                adaptive_min, adaptive_max
            ))

            width = rec["outer"] - rec["inner"]
            n_r = max(0, int(math.ceil(width / radial_spacing)) - 1)
            if n_r <= 0:
                continue

            rr = np.linspace(rec["inner"], rec["outer"], n_r + 2, dtype=np.float64)[1:-1]
            r_mid = 0.5 * (rec["inner"] + rec["outer"])
            arc = max(r_mid * (phi1 - phi0), 1e-6)
            n_phi = max(1, int(math.ceil(arc / tangential_spacing)))
            pp = np.linspace(phi0, phi1, n_phi, endpoint=False, dtype=np.float64) + 0.5 * (phi1 - phi0) / n_phi
            R, P = np.meshgrid(rr, pp, indexing="ij")
            rv, pv = R.ravel(), P.ravel()

            if max_candidates > 0:
                remaining = max_candidates - count
                if remaining <= 0:
                    break
                if len(rv) > remaining:
                    rv = rv[:remaining]
                    pv = pv[:remaining]

            t = np.clip((rv - rec["r0"]) / max(rec["r1"] - rec["r0"], 1e-9), 0.0, 1.0)
            z = rec["z0"] + t * (rec["z1"] - rec["z0"])
            x = origin_xy[0] + rv * np.cos(pv)
            y = origin_xy[1] + rv * np.sin(pv)
            xyz = np.column_stack([x, y, z]).astype(np.float32)

            radial_slope = (rec["z1"] - rec["z0"]) / max(rec["r1"] - rec["r0"], 1e-9)
            tangential_slope = np.zeros_like(rv)
            # Exact vectorized equivalent of the old per-candidate Python loop.
            zm = _neighbor_interval_z_many(intervals, (s - 1) % nsec, rv)
            zp = _neighbor_interval_z_many(intervals, (s + 1) % nsec, rv)
            ds = np.maximum(rv * dphi, 1e-6)
            has_m = np.isfinite(zm)
            has_p = np.isfinite(zp)
            both = has_m & has_p
            only_p = (~has_m) & has_p
            only_m = has_m & (~has_p)
            tangential_slope[both] = (zp[both] - zm[both]) / (2.0 * ds[both])
            tangential_slope[only_p] = (zp[only_p] - z[only_p]) / ds[only_p]
            tangential_slope[only_m] = (z[only_m] - zm[only_m]) / ds[only_m]

            grad_x = radial_slope * np.cos(pv) - tangential_slope * np.sin(pv)
            grad_y = radial_slope * np.sin(pv) + tangential_slope * np.cos(pv)
            n = np.column_stack([-grad_x, -grad_y, np.ones_like(grad_x)])
            n /= np.linalg.norm(n, axis=1, keepdims=True)

            xyzs.append(xyz)
            normals.append(n.astype(np.float32))
            N = len(xyz)
            gaps.append(np.full(N, rec["gap"], np.float32))
            expected.append(np.full(N, rec["expected"], np.float32))
            ratios.append(np.full(N, rec["ratio"], np.float32))
            r0s.append(np.full(N, rec["r0"], np.float32))
            z0s.append(np.full(N, rec["z0"], np.float32))
            r1s.append(np.full(N, rec["r1"], np.float32))
            z1s.append(np.full(N, rec["z1"], np.float32))
            ts.append(t.astype(np.float32))
            grades.append(np.full(N, rec["grade_percent"], np.float32))
            loo_maxs.append(np.full(N, rec.get("anchor_loo_max_m", 0.0), np.float32))
            neigh_maxs.append(np.full(N, rec.get("anchor_neighbor_max_m", 0.0), np.float32))
            used_spacings.append(np.full(N, tangential_spacing, np.float32))
            radial_spacings.append(np.full(N, radial_spacing, np.float32))

            count += N
            if max_candidates > 0 and count >= max_candidates:
                break

        if max_candidates > 0 and count >= max_candidates:
            break

    if not xyzs:
        empty3 = np.empty((0, 3), np.float32)
        empty1 = np.empty((0,), np.float32)
        return empty3, empty3.copy(), *(empty1.copy() for _ in range(13))

    xyz = np.concatenate(xyzs, axis=0)
    normal = np.concatenate(normals, axis=0)
    scalars = [np.concatenate(x, axis=0)
               for x in (gaps, expected, ratios, r0s, z0s, r1s, z1s, ts, grades,
                         loo_maxs, neigh_maxs, used_spacings, radial_spacings)]
    return xyz, normal, *scalars

def generic_coverage_candidates(points_xy, spacing, radius, knn, min_quadrants, max_candidates):
    if len(points_xy) < knn:
        return np.empty((0, 2), np.float32)
    lo = points_xy.min(axis=0) - radius; hi = points_xy.max(axis=0) + radius
    nx = int(math.floor((hi[0] - lo[0]) / spacing)) + 1; ny = int(math.floor((hi[1] - lo[1]) / spacing)) + 1
    tree = cKDTree(points_xy); accepted = []; accepted_count = 0
    chunk_rows = max(1, int(250000 / max(nx, 1)))
    xs = lo[0] + np.arange(nx, dtype=np.float64) * spacing
    for y0 in range(0, ny, chunk_rows):
        y1 = min(ny, y0 + chunk_rows)
        ys = lo[1] + np.arange(y0, y1, dtype=np.float64) * spacing
        X, Y = np.meshgrid(xs, ys, indexing="xy"); q = np.column_stack([X.ravel(), Y.ravel()])
        dist, idx = tree.query(q, k=min(knn, len(points_xy)), workers=-1)
        if dist.ndim == 1: dist = dist[:, None]; idx = idx[:, None]
        nearest = dist[:, 0]; mask = (nearest >= 0.95 * spacing) & (nearest <= radius)
        if not np.any(mask): continue
        qq, dd, ii = q[mask], dist[mask], idx[mask]
        rel = points_xy[ii] - qq[:, None, :]; valid = dd <= radius
        quad = (rel[..., 0] >= 0).astype(np.int8) + 2 * (rel[..., 1] >= 0).astype(np.int8)
        qcount = np.zeros(len(qq), dtype=np.int8)
        for qid in range(4): qcount += np.any(valid & (quad == qid), axis=1)
        keep = qcount >= min_quadrants
        if np.any(keep):
            block = qq[keep].astype(np.float32)
            accepted.append(block)
            accepted_count += len(block)
            if max_candidates > 0 and accepted_count >= max_candidates: break
    if not accepted:
        return np.empty((0, 2), np.float32)
    result = np.concatenate(accepted, axis=0)
    return result[:max_candidates] if max_candidates > 0 else result


def fit_generic_candidates(cand_xy, family_points, family_tree, k, rmse_max, support_radius, min_sep,
                           boundary_tree_xy=None, boundary_radius=0.12, chunk=50000):
    if len(cand_xy) == 0:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0,), np.float32), np.empty((0,), np.float32))
    xyz_out, normals_out, rmse_out, nearest_out = [], [], [], []
    kk = min(k, len(family_points))
    for begin in range(0, len(cand_xy), chunk):
        qxy = cand_xy[begin:begin + chunk]
        dxy, idx = family_tree.query(qxy, k=kk, workers=-1)
        if kk == 1: dxy = dxy[:, None]; idx = idx[:, None]
        nearest = dxy[:, 0]; enough = (nearest >= min_sep) & (dxy[:, -1] <= support_radius)
        if not np.any(enough): continue
        qxy2, idx2 = qxy[enough], idx[enough]
        neigh = family_points[idx2]; mean = neigh.mean(axis=1); centered = neigh - mean[:, None, :]
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(1, kk)
        eigvals, eigvecs = np.linalg.eigh(cov); normal = eigvecs[:, :, 0]
        normal[normal[:, 2] < 0] *= -1.0
        rmse = np.sqrt(np.maximum(eigvals[:, 0], 0.0)); nz = normal[:, 2]
        valid = (rmse <= rmse_max) & (np.abs(nz) >= 0.25)
        if not np.any(valid): continue
        qxy3, m, n = qxy2[valid], mean[valid], normal[valid]
        z = m[:, 2] - (n[:, 0] * (qxy3[:, 0] - m[:, 0]) + n[:, 1] * (qxy3[:, 1] - m[:, 1])) / n[:, 2]
        xyz3 = np.column_stack([qxy3, z]).astype(np.float32)
        keep = np.ones(len(xyz3), dtype=bool)
        if boundary_tree_xy is not None:
            bd, _ = boundary_tree_xy.query(xyz3[:, :2], k=1, workers=-1); keep &= bd >= boundary_radius
        if np.any(keep):
            xyz_out.append(xyz3[keep]); normals_out.append(n[keep].astype(np.float32))
            rmse_out.append(rmse[valid][keep].astype(np.float32)); nearest_out.append(nearest[enough][valid][keep].astype(np.float32))
    if not xyz_out:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0,), np.float32), np.empty((0,), np.float32))
    return np.concatenate(xyz_out), np.concatenate(normals_out), np.concatenate(rmse_out), np.concatenate(nearest_out)


def filter_ring_support(xyz, family_tree_xy, min_sep, boundary_tree_xy=None, boundary_radius=0.12):
    if len(xyz) == 0:
        return np.empty((0,), dtype=bool), np.empty((0,), np.float32)
    nearest, _ = family_tree_xy.query(xyz[:, :2], k=1, workers=-1)
    keep = nearest >= min_sep
    if boundary_tree_xy is not None:
        bd, _ = boundary_tree_xy.query(xyz[:, :2], k=1, workers=-1); keep &= bd >= boundary_radius
    return keep, nearest.astype(np.float32)


def deduplicate_candidates(xyz, source, spacing, priority=None):
    """Vectorized exact equivalent of the former Python set-based dedup."""
    if len(xyz) == 0:
        return np.empty((0,), np.int64)
    cells = np.floor(xyz[:, :2] / spacing + 0.5).astype(np.int64)
    if priority is None:
        priority = np.zeros(len(xyz), dtype=np.float32)
    # Preserve the original winner ordering: source ascending, then priority
    # descending. np.unique is only used to find the first cell occurrence in
    # that ranked order; sorting its returned positions restores loop order.
    rank = np.lexsort((-priority, source))
    ranked_cells = np.ascontiguousarray(cells[rank])
    key_dtype = np.dtype((np.void, ranked_cells.dtype.itemsize * ranked_cells.shape[1]))
    packed = ranked_cells.view(key_dtype).reshape(-1)
    _, first = np.unique(packed, return_index=True)
    return rank[np.sort(first)].astype(np.int64, copy=False)




def assign_support_source_indices(original_xyz, original_sem, support_xyz, support_sem):
    """Map every generated support point to one ORIGINAL point.

    Preference:
      1. nearest original point with the exact same semantic_id;
      2. nearest original point from the same ground family.

    The returned original-point index is then used to copy every aligned
    per-point attribute (ground_id, instance_id, label_confidence,
    observation_frame_index, intensity, etc.) into the generated point.
    """
    if len(support_xyz) == 0:
        return np.empty(0, dtype=np.int64)

    # Keep the stored float32 geometry instead of eagerly duplicating the full
    # original + support clouds as float64. scipy cKDTree converts each source
    # subset/query to double internally, yielding the same distances/indices
    # while substantially reducing peak RAM for 20M+ point scenes.
    original_xyz = np.asarray(original_xyz)
    original_sem = np.asarray(original_sem, dtype=np.int16)
    support_xyz = np.asarray(support_xyz)
    support_sem = np.asarray(support_sem, dtype=np.int16)

    result = np.full(len(support_xyz), -1, dtype=np.int64)
    original_family = semantic_family(original_sem)

    for sid in np.unique(support_sem):
        qmask = support_sem == sid
        same = original_sem == sid

        if np.any(same):
            source_indices = np.flatnonzero(same)
            tree = cKDTree(original_xyz[source_indices])
            _, local = tree.query(support_xyz[qmask], k=1, workers=-1)
            result[qmask] = source_indices[np.asarray(local, dtype=np.int64)]
            continue

        family_id = 0
        for fam, values in GROUND_FAMILIES.items():
            if int(sid) in values:
                family_id = fam
                break

        same_family = original_family == family_id
        if family_id > 0 and np.any(same_family):
            source_indices = np.flatnonzero(same_family)
            tree = cKDTree(original_xyz[source_indices])
            _, local = tree.query(support_xyz[qmask], k=1, workers=-1)
            result[qmask] = source_indices[np.asarray(local, dtype=np.int64)]
            continue

        raise RuntimeError(
            f"Could not find an original semantic/family source for generated semantic_id={int(sid)}"
        )

    if np.any(result < 0):
        raise RuntimeError("Some generated support points could not be mapped to an original point.")
    return result


def _point_aligned_arrays(npz_dict, n_points):
    """Return all arrays whose first dimension is the point dimension."""
    out = {}
    for key, value in npz_dict.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n_points:
            out[key] = arr
    return out


def _refresh_data_labeled_rows(rows, columns, xyz, metadata):
    """Update copied data_labeled rows so their XYZ and known fields match new points."""
    rows = np.asarray(rows).copy()
    names = [str(x) for x in np.asarray(columns).reshape(-1)]
    if rows.ndim != 2 or rows.shape[1] != len(names):
        return rows

    xyz_names = {
        "x": 0, "x_world": 0, "x_vehicle": 0, "x_object": 0,
        "y": 1, "y_world": 1, "y_vehicle": 1, "y_object": 1,
        "z": 2, "z_world": 2, "z_vehicle": 2, "z_object": 2,
    }

    for j, name in enumerate(names):
        if name in xyz_names:
            rows[:, j] = xyz[:, xyz_names[name]]
        elif name in metadata:
            values = np.asarray(metadata[name])
            if values.ndim == 1 and len(values) == len(rows):
                rows[:, j] = values
    return rows


def build_densified_npz_arrays(strict_path, support_xyz, support_sem, support_source, support_family, source_index=None):
    """Build ORIGINAL + generated support while preserving every original NPZ attribute.

    Every point-aligned array in the original strict NPZ is retained. Generated
    points inherit all such attributes from their nearest original point of the
    same semantic class (or same ground family as fallback).

    xyz and semantic_id are then replaced by the generated geometry/class.
    data_labeled is refreshed when data_labeled_columns is available, so its
    XYZ columns are not stale.
    """
    with np.load(strict_path, allow_pickle=False) as d:
        original = {key: np.asarray(d[key]) for key in d.files}

    if "xyz" not in original or "semantic_id" not in original:
        raise KeyError("Strict NPZ must contain at least xyz and semantic_id.")

    original_xyz = np.asarray(original["xyz"], dtype=np.float32)
    original_sem = np.asarray(original["semantic_id"], dtype=np.int16)
    n_original = len(original_xyz)
    n_support = len(support_xyz)

    if source_index is None:
        source_index = assign_support_source_indices(
            original_xyz, original_sem, support_xyz, support_sem
        )
    else:
        source_index = np.asarray(source_index, dtype=np.int64)
        if len(source_index) != n_support:
            raise ValueError("precomputed source_index length does not match support")

    point_arrays = _point_aligned_arrays(original, n_original)
    merged = {}

    # Preserve all non-point arrays exactly as they were.
    for key, value in original.items():
        if key not in point_arrays:
            merged[key] = value

    # Geometry and semantic class of generated support are authoritative.
    merged["xyz"] = np.concatenate(
        [original_xyz, np.asarray(support_xyz, dtype=np.float32)], axis=0
    )
    merged["semantic_id"] = np.concatenate(
        [original_sem, np.asarray(support_sem, dtype=np.int16)], axis=0
    )

    # Preserve every other aligned field without constructing a second dict of
    # all generated metadata at once. This materially reduces peak RAM.
    for key, value in point_arrays.items():
        if key in ("xyz", "semantic_id", "data_labeled"):
            continue
        destination = np.empty((n_original + n_support,) + value.shape[1:], dtype=value.dtype)
        destination[:n_original] = value
        destination[n_original:] = value[source_index]
        merged[key] = destination

    # data_labeled contains coordinates, so simply copying the source row would
    # leave stale XYZ. Copy it, then refresh all known columns.
    if "data_labeled" in point_arrays:
        support_rows = point_arrays["data_labeled"][source_index].copy()
        if "data_labeled_columns" in original:
            refresh_meta = {
                key: value[source_index]
                for key, value in point_arrays.items()
                if key not in ("xyz", "semantic_id", "data_labeled")
            }
            refresh_meta["semantic_id"] = np.asarray(support_sem, dtype=np.int16)
            support_rows = _refresh_data_labeled_rows(
                support_rows,
                original["data_labeled_columns"],
                np.asarray(support_xyz, dtype=np.float32),
                refresh_meta,
            )
        merged["data_labeled"] = np.concatenate(
            [point_arrays["data_labeled"], support_rows], axis=0
        )

    # Explicit provenance for the densification itself.
    merged["is_generated"] = np.concatenate([
        np.zeros(n_original, dtype=np.uint8),
        np.ones(n_support, dtype=np.uint8),
    ])
    merged["densification_source_type"] = np.concatenate([
        np.zeros(n_original, dtype=np.int8),
        np.asarray(support_source, dtype=np.int8),
    ])
    merged["ground_family"] = np.concatenate([
        semantic_family(original_sem).astype(np.int8),
        np.asarray(support_family, dtype=np.int8),
    ])
    merged["source_original_point_index"] = np.concatenate([
        np.arange(n_original, dtype=np.int64),
        source_index.astype(np.int64),
    ])
    merged["original_point_count"] = np.int64(n_original)
    merged["generated_point_count"] = np.int64(n_support)
    merged["metadata_transfer_note"] = np.asarray(
        "Generated support inherits every aligned original NPZ attribute from "
        "the nearest original point of matching semantic class/family."
    )

    return merged, source_index


def _pcd_dtype():
    return np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("semantic_id", "<i2"), ("intensity", "<f4")
    ])


def _write_pcd_chunk(f, xyz, semantic, intensity):
    dtype = _pcd_dtype()
    points = np.empty(len(xyz), dtype=dtype)
    points["x"] = xyz[:, 0]
    points["y"] = xyz[:, 1]
    points["z"] = xyz[:, 2]
    points["semantic_id"] = semantic
    points["intensity"] = intensity
    points.tofile(f)


def export_densified_pcd(strict_path, support_xyz, support_sem, support_source_index,
                         out_pcd, chunk_size=1000000):
    """Write original strict cloud + generated support to a lightweight binary PCD.

    The complete metadata is retained in the densified NPZ. PCD remains limited
    to x/y/z/semantic_id/intensity for compatibility with downstream PCL tools.
    """
    out_pcd = Path(out_pcd)
    out_pcd.parent.mkdir(parents=True, exist_ok=True)

    with np.load(strict_path, allow_pickle=False) as d:
        original_xyz = np.asarray(d["xyz"], dtype=np.float32)
        original_sem = np.asarray(d["semantic_id"], dtype=np.int16)
        original_intensity = np.asarray(d["intensity"], dtype=np.float32)

    support_intensity = original_intensity[np.asarray(support_source_index, dtype=np.int64)]
    n_original = len(original_xyz)
    n_support = len(support_xyz)
    n_total = n_original + n_support

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z semantic_id intensity
SIZE 4 4 4 2 4
TYPE F F F I F
COUNT 1 1 1 1 1
WIDTH {n_total}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n_total}
DATA binary
"""
    with out_pcd.open("wb") as f:
        f.write(header.encode("ascii"))
        for begin in range(0, n_original, chunk_size):
            end = min(begin + chunk_size, n_original)
            _write_pcd_chunk(
                f, original_xyz[begin:end], original_sem[begin:end],
                original_intensity[begin:end]
            )
        for begin in range(0, n_support, chunk_size):
            end = min(begin + chunk_size, n_support)
            _write_pcd_chunk(
                f, support_xyz[begin:end], support_sem[begin:end],
                support_intensity[begin:end]
            )

    print(f"[densified export] original strict points : {n_original:,}")
    print(f"[densified export] generated support      : {n_support:,}")
    print(f"[densified export] final densified cloud  : {n_total:,}")
    print(f"[densified export] binary PCD             : {out_pcd}")


def save_npz(path, arrays, compression="compressed"):
    if compression == "stored":
        np.savez(path, **arrays)
    elif compression == "compressed":
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError(f"Unknown NPZ compression mode: {compression}")


def _zip_compression(mode):
    if mode == "stored":
        return zipfile.ZIP_STORED
    if mode == "compressed":
        return zipfile.ZIP_DEFLATED
    raise ValueError(f"Unknown NPZ compression mode: {mode}")


def _write_array_entry(archive, name, array):
    """Write one ordinary array into an NPZ archive without pickling."""
    with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
        npy_format.write_array(stream, np.asarray(array), allow_pickle=False)


def _open_streamed_array(archive, name, dtype, shape):
    """Open an NPZ entry and emit a C-order NPY header for streamed rows."""
    stream = archive.open(f"{name}.npy", "w", force_zip64=True)
    npy_format.write_array_header_2_0(stream, {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": tuple(shape),
    })
    return stream


def _write_c_array(stream, array):
    array = np.ascontiguousarray(array)
    if array.size:
        stream.write(array.tobytes(order="C"))


def _stream_original_plus_mapped(stream, original, source_index, chunk_size):
    """Write original rows followed by source-index-mapped generated rows."""
    original = np.asarray(original)
    for begin in range(0, len(original), chunk_size):
        _write_c_array(stream, original[begin:begin + chunk_size])
    for begin in range(0, len(source_index), chunk_size):
        rows = source_index[begin:begin + chunk_size]
        _write_c_array(stream, original[rows])


def export_densified_npz(strict_path, support_xyz, support_sem, support_source,
                         support_family, out_npz, source_index=None,
                         compression="compressed", chunk_size=1000000):
    """Stream ORIGINAL + support into an NPZ while preserving all attributes.

    The previous implementation materialized every merged original+generated
    point-aligned array simultaneously before calling np.savez. On a 40M-point
    scene that can consume several extra GB of RAM. This writer produces the
    same loaded arrays one key at a time and writes generated metadata in
    chunks, so peak memory is governed by one source attribute plus a chunk.
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    support_xyz = np.asarray(support_xyz, dtype=np.float32)
    support_sem = np.asarray(support_sem, dtype=np.int16)
    support_source = np.asarray(support_source, dtype=np.int8)
    support_family = np.asarray(support_family, dtype=np.int8)

    with np.load(strict_path, allow_pickle=False) as strict:
        if "xyz" not in strict.files or "semantic_id" not in strict.files:
            raise KeyError("Strict NPZ must contain at least xyz and semantic_id.")
        original_xyz = np.asarray(strict["xyz"], dtype=np.float32)
        original_sem = np.asarray(strict["semantic_id"], dtype=np.int16)
        n_original = len(original_xyz)
        n_support = len(support_xyz)

        if source_index is None:
            source_index = assign_support_source_indices(
                original_xyz, original_sem, support_xyz, support_sem
            )
        source_index = np.asarray(source_index, dtype=np.int64)
        if len(source_index) != n_support:
            raise ValueError("source_index length does not match generated support")

        override_keys = {
            "xyz", "semantic_id", "data_labeled",
            "is_generated", "densification_source_type", "ground_family",
            "source_original_point_index", "original_point_count",
            "generated_point_count", "metadata_transfer_note",
        }

        with zipfile.ZipFile(
            out_npz,
            mode="w",
            compression=_zip_compression(compression),
            allowZip64=True,
        ) as archive:
            # Authoritative geometry and semantic class.
            with _open_streamed_array(
                archive, "xyz", np.float32, (n_original + n_support, 3)
            ) as stream:
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(stream, original_xyz[begin:begin + chunk_size])
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, support_xyz[begin:begin + chunk_size])

            with _open_streamed_array(
                archive, "semantic_id", np.int16, (n_original + n_support,)
            ) as stream:
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(stream, original_sem[begin:begin + chunk_size])
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, support_sem[begin:begin + chunk_size])

            # Preserve every original array. Point-aligned arrays inherit their
            # generated rows from source_index; non-point arrays are unchanged.
            for key in strict.files:
                if key in override_keys:
                    continue
                value = np.asarray(strict[key])
                if value.ndim >= 1 and value.shape[0] == n_original:
                    shape = (n_original + n_support,) + value.shape[1:]
                    with _open_streamed_array(
                        archive, key, value.dtype, shape
                    ) as stream:
                        _stream_original_plus_mapped(
                            stream, value, source_index, chunk_size
                        )
                else:
                    _write_array_entry(archive, key, value)

            # data_labeled is point-aligned but contains coordinates. Copy the
            # source row, then refresh XYZ and any columns backed by known
            # point-aligned 1-D metadata exactly as the original implementation.
            if "data_labeled" in strict.files:
                data_labeled = np.asarray(strict["data_labeled"])
                if data_labeled.ndim < 1 or data_labeled.shape[0] != n_original:
                    _write_array_entry(archive, "data_labeled", data_labeled)
                else:
                    columns = (
                        np.asarray(strict["data_labeled_columns"])
                        if "data_labeled_columns" in strict.files
                        else None
                    )
                    refresh_arrays = {}
                    if columns is not None:
                        names = {str(x) for x in columns.reshape(-1)}
                        for name in names:
                            if name in {"xyz", "semantic_id", "data_labeled"}:
                                continue
                            if name in strict.files:
                                candidate = np.asarray(strict[name])
                                if candidate.ndim == 1 and len(candidate) == n_original:
                                    refresh_arrays[name] = candidate

                    shape = (n_original + n_support,) + data_labeled.shape[1:]
                    with _open_streamed_array(
                        archive, "data_labeled", data_labeled.dtype, shape
                    ) as stream:
                        for begin in range(0, n_original, chunk_size):
                            _write_c_array(
                                stream, data_labeled[begin:begin + chunk_size]
                            )
                        for begin in range(0, n_support, chunk_size):
                            end = min(begin + chunk_size, n_support)
                            src = source_index[begin:end]
                            rows = data_labeled[src].copy()
                            if columns is not None:
                                metadata = {
                                    name: values[src]
                                    for name, values in refresh_arrays.items()
                                }
                                metadata["semantic_id"] = support_sem[begin:end]
                                rows = _refresh_data_labeled_rows(
                                    rows,
                                    columns,
                                    support_xyz[begin:end],
                                    metadata,
                                )
                            _write_c_array(stream, rows)

            # Densification provenance.
            with _open_streamed_array(
                archive, "is_generated", np.uint8, (n_original + n_support,)
            ) as stream:
                zero = np.zeros(min(chunk_size, max(n_original, 1)), dtype=np.uint8)
                one = np.ones(min(chunk_size, max(n_support, 1)), dtype=np.uint8)
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(stream, zero[:min(chunk_size, n_original - begin)])
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, one[:min(chunk_size, n_support - begin)])

            with _open_streamed_array(
                archive, "densification_source_type", np.int8,
                (n_original + n_support,)
            ) as stream:
                zero = np.zeros(min(chunk_size, max(n_original, 1)), dtype=np.int8)
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(stream, zero[:min(chunk_size, n_original - begin)])
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, support_source[begin:begin + chunk_size])

            with _open_streamed_array(
                archive, "ground_family", np.int8, (n_original + n_support,)
            ) as stream:
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(
                        stream,
                        semantic_family(original_sem[begin:begin + chunk_size]).astype(
                            np.int8, copy=False
                        ),
                    )
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, support_family[begin:begin + chunk_size])

            with _open_streamed_array(
                archive, "source_original_point_index", np.int64,
                (n_original + n_support,)
            ) as stream:
                for begin in range(0, n_original, chunk_size):
                    _write_c_array(
                        stream,
                        np.arange(
                            begin,
                            min(begin + chunk_size, n_original),
                            dtype=np.int64,
                        ),
                    )
                for begin in range(0, n_support, chunk_size):
                    _write_c_array(stream, source_index[begin:begin + chunk_size])

            _write_array_entry(archive, "original_point_count", np.int64(n_original))
            _write_array_entry(archive, "generated_point_count", np.int64(n_support))
            _write_array_entry(
                archive,
                "metadata_transfer_note",
                np.asarray(
                    "Generated support inherits every aligned original NPZ attribute "
                    "from the nearest original point of matching semantic class/family."
                ),
            )

    print(f"[densified export] densified NPZ            : {out_npz}")
    print(f"[densified export] streaming writer         : yes ({compression})")
    print(f"[densified export] original strict points   : {n_original:,}")
    print(f"[densified export] generated support        : {n_support:,}")
    return source_index

def main():
    ap = argparse.ArgumentParser("Robust same-sector ring interpolation with anchor outlier rejection")
    ap.add_argument("-s", "--source_path", required=True); ap.add_argument("--caseid", required=True); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--analysis_voxel", type=float, default=0.10)
    ap.add_argument("--fill_spacing", type=float, default=0.10, help="Fixed spacing used in fixed mode and by generic coverage")
    ap.add_argument("--fill_spacing_mode", choices=["fixed", "local_ring_density", "along_ring_density"], default="along_ring_density", help="fixed; generic XY-NN adaptive; or density measured along the two actual rings bracketing each gap")
    ap.add_argument("--adaptive_density_voxel", type=float, default=0.01, help="Fine XY deduplication used only to estimate observed local ring density")
    ap.add_argument("--adaptive_density_neighbors", type=int, default=3, help="Nearest neighbours used to estimate observed point spacing")
    ap.add_argument("--adaptive_query_neighbors", type=int, default=16, help="Local density samples queried around each bracketing ring anchor")
    ap.add_argument("--adaptive_fill_scale", type=float, default=1.0, help="Multiply measured local ring spacing by this factor")
    ap.add_argument("--adaptive_fill_min", type=float, default=0.01, help="Minimum allowed adaptive fill spacing in metres")
    ap.add_argument("--adaptive_fill_max", type=float, default=0.08, help="Maximum allowed adaptive fill spacing in metres")
    ap.add_argument("--ring_density_duplicate_threshold", type=float, default=0.003, help="Only for density estimation: collapse nearly identical repeated samples along a ring within this tangential distance")
    ap.add_argument("--ring_density_min_unique_samples", type=int, default=3, help="Minimum unique along-ring samples required for a direct spacing estimate")
    ap.add_argument("--radial_spacing_scale", type=float, default=1.0, help="Radial fill spacing = this factor x measured along-ring spacing")
    ap.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True, help="Remove near-duplicate generated candidates caused by overlapping generators")
    ap.add_argument("--dedup_spacing", type=float, default=0.0, help="Generated-point dedup cell size. 0=derive automatically")
    ap.add_argument(
        "--full_resolution",
        action="store_true",
        help=(
            "Use the complete original ground point cloud for geometry analysis and "
            "density estimation: no analysis voxelization, no density-estimation "
            "voxelization, no generated-point deduplication, no minimum-separation "
            "thinning, and no point-count caps."
        ),
    )
    ap.add_argument(
        "--disable_safety_filters",
        action="store_true",
        help=(
            "Also disable ring-anchor outlier rejection, grade rejection, abnormal-gap "
            "rejection, and semantic-boundary rejection. This is intentionally permissive."
        ),
    )
    ap.add_argument("--ring_mode", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--ring_origin_spread_max", type=float, default=1.0)
    ap.add_argument("--ring_sector_deg", type=float, default=0.5); ap.add_argument("--ring_split_gap", type=float, default=0.16)
    ap.add_argument("--ring_profile_bin_m", type=float, default=5.0); ap.add_argument("--ring_max_gap_factor", type=float, default=2.75)
    ap.add_argument("--ring_min_points_per_run", type=int, default=8); ap.add_argument("--ring_min_sector_points", type=int, default=24)
    ap.add_argument("--ring_max_abs_grade_percent", type=float, default=12.0, help="Safety reject for obviously mismatched adjacent clean ring anchors")
    ap.add_argument("--ring_anchor_loo_floor_m", type=float, default=0.03, help="Minimum leave-one-out Z-spike rejection threshold")
    ap.add_argument("--ring_anchor_loo_sigma", type=float, default=6.0, help="Robust MAD multiplier for radial anchor-spike rejection")
    ap.add_argument("--ring_neighbor_half_window", type=int, default=1, help="Neighbouring 0.5deg sectors used only to validate anchor Z")
    ap.add_argument("--ring_neighbor_z_tol_m", type=float, default=0.05, help="Reject anchor if matched neighbouring-sector Z differs by more than this")
    ap.add_argument("--ring_neighbor_min_matches", type=int, default=1)
    ap.add_argument("--generic_coverage", action="store_true"); ap.add_argument("--coverage_radius", type=float, default=0.55)
    ap.add_argument("--coverage_knn", type=int, default=12); ap.add_argument("--coverage_min_quadrants", type=int, default=3)
    ap.add_argument("--plane_knn", type=int, default=16); ap.add_argument("--plane_rmse_max", type=float, default=0.06)
    ap.add_argument("--plane_support_radius", type=float, default=1.25); ap.add_argument("--boundary_radius", type=float, default=0.12)
    ap.add_argument("--min_separation", type=float, default=0.065)
    ap.add_argument("--max_candidates_per_family", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--max_support", type=int, default=0, help="0 = unlimited; no global support cap")
    ap.add_argument("--strict_path", default=None, help="Optional explicit strict NPZ. Default: <source_path>/recon_related/<case>/static_filter/static_recon_labels_strict.npz")
    ap.add_argument("--densified_pcd", default=None, help="Export ORIGINAL strict cloud + generated support to this binary PCD")
    ap.add_argument("--densified_npz", default=None, help="Export merged densified NPZ while preserving all original NPZ attributes")
    ap.add_argument("--npz_compression", choices=["compressed", "stored"], default="compressed", help="NPZ storage mode; stored is exact but much faster/larger")
    ap.add_argument("--timing_json", default=None, help="Optional JSON file with detailed CPU-stage wall times")
    ap.add_argument("--pcd_chunk_size", type=int, default=1000000)
    args = ap.parse_args()
    if args.fill_spacing <= 0:
        ap.error("--fill_spacing must be > 0")
    if args.adaptive_density_voxel < 0:
        ap.error("--adaptive_density_voxel must be >= 0")
    if args.adaptive_fill_min <= 0 or args.adaptive_fill_max <= 0 or args.adaptive_fill_min > args.adaptive_fill_max:
        ap.error("adaptive fill limits must satisfy 0 < min <= max")
    if args.adaptive_fill_scale <= 0:
        ap.error("--adaptive_fill_scale must be > 0")
    if args.ring_density_duplicate_threshold < 0:
        ap.error("--ring_density_duplicate_threshold must be >= 0")
    if args.ring_density_min_unique_samples < 2:
        ap.error("--ring_density_min_unique_samples must be >= 2")
    if args.radial_spacing_scale <= 0:
        ap.error("--radial_spacing_scale must be > 0")
    if args.max_candidates_per_family < 0 or args.max_support < 0:
        ap.error("candidate/support caps must be >= 0; use 0 for unlimited")

    if args.full_resolution:
        # Density/thinning filters are disabled. The original ground cloud is used
        # directly for analysis and local-density estimation.
        args.analysis_voxel = 0.0
        args.adaptive_density_voxel = 0.0
        args.min_separation = 0.0
        args.deduplicate = False
        args.max_candidates_per_family = 0
        args.max_support = 0

    strict_path = args.strict_path or os.path.join(args.source_path, "recon_related", args.caseid, "static_filter", "static_recon_labels_strict.npz")
    timing = {}
    total_started = time.perf_counter()
    stage_started = time.perf_counter()
    with np.load(strict_path, allow_pickle=False) as d:
        xyz_all = np.asarray(d["xyz"], dtype=np.float32)
        sem_all = np.asarray(d["semantic_id"], dtype=np.int16)
        if "coordinate_frame" in d:
            frame = str(np.asarray(d["coordinate_frame"]).reshape(-1)[0])
            if frame.lower() != "world": raise ValueError(f"Expected world coordinates, got {frame!r}")
    valid = np.isfinite(xyz_all).all(axis=1) & np.isin(sem_all, GROUND_SEMANTICS + [CURB_SEMANTIC])
    xyz = xyz_all[valid]
    sem = sem_all[valid]
    del xyz_all, sem_all
    timing["load_filter_input_s"] = time.perf_counter() - stage_started
    raw_fam = semantic_family(sem)
    print(f"[scene support] raw relevant points: {len(xyz):,}")
    print(f"[scene support] full_resolution={args.full_resolution} disable_safety_filters={args.disable_safety_filters}")
    if args.full_resolution:
        print("[scene support] NO analysis voxelization, NO density-estimation voxelization, NO min-separation thinning, NO generated-point deduplication, NO point-count caps")
    if args.disable_safety_filters:
        print("[scene support] SAFETY FILTERS DISABLED: no anchor-outlier, grade, abnormal-gap, or semantic-boundary rejection")
    stage_started = time.perf_counter()
    centers, csem = voxel_centroids(xyz, sem, args.analysis_voxel); fam = semantic_family(csem)
    timing["analysis_representation_s"] = time.perf_counter() - stage_started
    analysis_mode = "FULL RAW POINT CLOUD (no analysis voxelization)" if args.analysis_voxel <= 0 else f"{args.analysis_voxel:g} m voxel centroids"
    print(f"[scene support] analysis representation: {analysis_mode}")
    print(f"[scene support] analysis points: {len(centers):,}; ground={int((fam>0).sum()):,}; curb={int((csem==17).sum()):,}")

    origins = load_sensor_origins(args.source_path, args.caseid); tstats = trajectory_stats(origins)
    ring_active = args.ring_mode == "on" or (args.ring_mode == "auto" and tstats["p95_xy_spread_m"] <= args.ring_origin_spread_max)
    print(f"[scene support] TOP trajectory p95 spread={tstats['p95_xy_spread_m']:.3f}m path={tstats['path_length_xy_m']:.3f}m ring_mode={args.ring_mode} active={ring_active}")

    all_xyz=[]; all_sem=[]; all_fam=[]; all_source=[]; all_normal=[]; all_rmse=[]; all_nearest=[]
    all_gap=[]; all_expected=[]; all_ratio=[]; all_r0=[]; all_z0=[]; all_r1=[]; all_z1=[]; all_t=[]; all_grade=[]; all_loo=[]; all_neigh=[]; all_spacing=[]; all_radial_spacing=[]
    family_meta={}
    family_timings = {}

    for family_id in (1,2,3):
        family_started = time.perf_counter()
        family_stage_timing = {}
        ids = np.flatnonzero(fam == family_id); pts = centers[ids]
        raw_pts = xyz[raw_fam == family_id]
        if len(pts) < args.plane_knn:
            family_meta[str(family_id)]={"points":int(len(pts)),"support":0}; continue
        stage_started = time.perf_counter()
        tree_xy = cKDTree(pts[:, :2])
        other_ground_xyz = centers[(fam > 0) & (fam != family_id)]; curb_xyz = centers[csem == CURB_SEMANTIC]
        boundary_xyz = np.concatenate([curb_xyz, other_ground_xyz], axis=0) if len(curb_xyz)+len(other_ground_xyz) else np.empty((0,3),np.float32)
        boundary_tree_xy = cKDTree(boundary_xyz[:, :2]) if len(boundary_xyz) else None
        if args.disable_safety_filters:
            boundary_tree_xy = None
        family_stage_timing["tree_build_s"] = time.perf_counter() - stage_started

        fam_xyz=[]; fam_sem=[]; fam_source=[]; fam_normal=[]; fam_rmse=[]; fam_nearest=[]
        fam_gap=[]; fam_expected=[]; fam_ratio=[]; fam_r0=[]; fam_z0=[]; fam_r1=[]; fam_z1=[]; fam_t=[]; fam_grade=[]; fam_loo=[]; fam_neigh=[]; fam_spacing=[]; fam_radial_spacing=[]
        ring_meta={"active":False}; ring_count=0; generic_count=0
        spacing_estimator = None
        stage_started = time.perf_counter()
        if args.fill_spacing_mode == "local_ring_density":
            spacing_estimator = build_local_ring_spacing_estimator(
                raw_pts[:, :2], args.adaptive_density_voxel,
                args.adaptive_density_neighbors, args.adaptive_query_neighbors
            )
        family_stage_timing["spacing_estimator_s"] = time.perf_counter() - stage_started

        if ring_active:
            gap_reference_spacing = args.fill_spacing if args.fill_spacing_mode == "fixed" else args.adaptive_fill_min

            if args.disable_safety_filters:
                ring_max_gap_factor = 1.0e12
                ring_min_points_per_run = 1
                ring_min_sector_points = 1
                ring_max_abs_grade_percent = 1.0e12
                ring_anchor_loo_floor_m = 1.0e12
                ring_anchor_loo_sigma = 1.0e12
                ring_neighbor_z_tol_m = 1.0e12
                ring_neighbor_min_matches = 0
            else:
                ring_max_gap_factor = args.ring_max_gap_factor
                ring_min_points_per_run = args.ring_min_points_per_run
                ring_min_sector_points = args.ring_min_sector_points
                ring_max_abs_grade_percent = args.ring_max_abs_grade_percent
                ring_anchor_loo_floor_m = args.ring_anchor_loo_floor_m
                ring_anchor_loo_sigma = args.ring_anchor_loo_sigma
                ring_neighbor_z_tol_m = args.ring_neighbor_z_tol_m
                ring_neighbor_min_matches = args.ring_neighbor_min_matches

            stage_started = time.perf_counter()
            intervals, ring_meta = build_ring_gap_intervals(
                raw_pts, tstats["center"][:2], args.ring_sector_deg, args.ring_split_gap, gap_reference_spacing,
                args.ring_profile_bin_m, ring_max_gap_factor, ring_min_points_per_run,
                ring_min_sector_points, ring_max_abs_grade_percent,
                ring_anchor_loo_floor_m, ring_anchor_loo_sigma,
                args.ring_neighbor_half_window, ring_neighbor_z_tol_m, ring_neighbor_min_matches,
                spacing_mode=args.fill_spacing_mode,
                adaptive_scale=args.adaptive_fill_scale,
                adaptive_min=args.adaptive_fill_min,
                adaptive_max=args.adaptive_fill_max,
                density_duplicate_threshold_m=args.ring_density_duplicate_threshold,
                density_min_unique_samples=args.ring_density_min_unique_samples)
            family_stage_timing["ring_interval_detection_s"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            rxyz, rn, rgap, rexp, rratio, rr0, rz0, rr1, rz1, rt, rgrade, rloo, rneigh, rspacing, rrspacing = ring_candidates_direct_z(
                intervals, tstats["center"][:2], args.ring_sector_deg, args.fill_spacing,
                args.max_candidates_per_family, spacing_mode=args.fill_spacing_mode,
                spacing_estimator=spacing_estimator, adaptive_scale=args.adaptive_fill_scale,
                adaptive_min=args.adaptive_fill_min, adaptive_max=args.adaptive_fill_max,
                radial_spacing_scale=args.radial_spacing_scale)
            family_stage_timing["ring_candidate_generation_s"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if args.disable_safety_filters:
                keep = np.ones(len(rxyz), dtype=bool)
                if len(rxyz):
                    rnear, _ = tree_xy.query(rxyz[:, :2], k=1, workers=-1)
                    rnear = np.asarray(rnear, dtype=np.float32)
                else:
                    rnear = np.empty(0, dtype=np.float32)
            else:
                keep, rnear = filter_ring_support(
                    rxyz, tree_xy, args.min_separation, boundary_tree_xy, args.boundary_radius
                )
            family_stage_timing["ring_filter_s"] = time.perf_counter() - stage_started
            rxyz,rn,rgap,rexp,rratio,rr0,rz0,rr1,rz1,rt,rgrade,rloo,rneigh,rspacing,rrspacing,rnear = [a[keep] for a in (rxyz,rn,rgap,rexp,rratio,rr0,rz0,rr1,rz1,rt,rgrade,rloo,rneigh,rspacing,rrspacing,rnear)]
            ring_count=len(rxyz); ring_meta["active"]=True; ring_meta["raw_candidates"]=int(len(keep)); ring_meta["accepted_support"]=int(ring_count)
            if ring_count:
                fam_xyz.append(rxyz); fam_sem.append(np.full(ring_count,FAMILY_DEFAULT_SEMANTIC[family_id],np.int16)); fam_source.append(np.full(ring_count,3,np.int8))
                fam_normal.append(rn); fam_rmse.append(np.zeros(ring_count,np.float32)); fam_nearest.append(rnear)
                fam_gap.append(rgap); fam_expected.append(rexp); fam_ratio.append(rratio); fam_r0.append(rr0); fam_z0.append(rz0); fam_r1.append(rr1); fam_z1.append(rz1); fam_t.append(rt); fam_grade.append(rgrade); fam_loo.append(rloo); fam_neigh.append(rneigh); fam_spacing.append(rspacing); fam_radial_spacing.append(rrspacing)

        if args.generic_coverage:
            coverage_min_quadrants = 0 if args.disable_safety_filters else args.coverage_min_quadrants
            stage_started = time.perf_counter()
            gxy = generic_coverage_candidates(
                pts[:, :2], args.fill_spacing, args.coverage_radius, args.coverage_knn,
                coverage_min_quadrants, args.max_candidates_per_family
            )
            family_stage_timing["generic_grid_query_s"] = time.perf_counter() - stage_started

            plane_rmse_max = float("inf") if args.disable_safety_filters else args.plane_rmse_max
            plane_support_radius = float("inf") if args.disable_safety_filters else args.plane_support_radius
            generic_min_separation = 0.0 if args.disable_safety_filters else args.min_separation
            generic_boundary_tree = None if args.disable_safety_filters else boundary_tree_xy

            stage_started = time.perf_counter()
            gxyz,gn,grmse,gnear = fit_generic_candidates(
                gxy, pts, tree_xy, args.plane_knn, plane_rmse_max,
                plane_support_radius, generic_min_separation,
                generic_boundary_tree, args.boundary_radius
            )
            family_stage_timing["generic_plane_fit_s"] = time.perf_counter() - stage_started
            generic_count=len(gxyz)
            if generic_count:
                fam_xyz.append(gxyz); fam_sem.append(np.full(generic_count,FAMILY_DEFAULT_SEMANTIC[family_id],np.int16)); fam_source.append(np.full(generic_count,4,np.int8))
                fam_normal.append(gn); fam_rmse.append(grmse); fam_nearest.append(gnear)
                zero=np.zeros(generic_count,np.float32)
                fam_gap.append(zero.copy()); fam_expected.append(zero.copy()); fam_ratio.append(zero.copy()); fam_r0.append(zero.copy()); fam_z0.append(zero.copy()); fam_r1.append(zero.copy()); fam_z1.append(zero.copy()); fam_t.append(zero.copy()); fam_grade.append(zero.copy()); fam_loo.append(zero.copy()); fam_neigh.append(zero.copy()); fam_spacing.append(np.full(generic_count,args.fill_spacing,np.float32)); fam_radial_spacing.append(np.full(generic_count,args.fill_spacing,np.float32))

        if fam_xyz:
            stage_started = time.perf_counter()
            fx=np.concatenate(fam_xyz); fs=np.concatenate(fam_sem); fst=np.concatenate(fam_source); fn=np.concatenate(fam_normal); frm=np.concatenate(fam_rmse); fnear=np.concatenate(fam_nearest)
            fg=np.concatenate(fam_gap); fe=np.concatenate(fam_expected); fr=np.concatenate(fam_ratio); fr0=np.concatenate(fam_r0); fz0=np.concatenate(fam_z0); fr1=np.concatenate(fam_r1); fz1=np.concatenate(fam_z1); ft=np.concatenate(fam_t); fgrade=np.concatenate(fam_grade); floo=np.concatenate(fam_loo); fneigh=np.concatenate(fam_neigh); fspacing=np.concatenate(fam_spacing); frspacing=np.concatenate(fam_radial_spacing)
            if args.deduplicate:
                dedup_spacing = args.dedup_spacing if args.dedup_spacing > 0 else 0.5 * (args.fill_spacing if args.fill_spacing_mode == "fixed" else args.adaptive_fill_min)
                keep=deduplicate_candidates(fx,fst,dedup_spacing,priority=fr)
                fx,fs,fst,fn,frm,fnear,fg,fe,fr,fr0,fz0,fr1,fz1,ft,fgrade,floo,fneigh,fspacing,frspacing=[a[keep] for a in (fx,fs,fst,fn,frm,fnear,fg,fe,fr,fr0,fz0,fr1,fz1,ft,fgrade,floo,fneigh,fspacing,frspacing)]
            family_stage_timing["family_concat_dedup_s"] = time.perf_counter() - stage_started
            all_xyz.append(fx); all_sem.append(fs); all_fam.append(np.full(len(fx),family_id,np.int8)); all_source.append(fst); all_normal.append(fn); all_rmse.append(frm); all_nearest.append(fnear)
            all_gap.append(fg); all_expected.append(fe); all_ratio.append(fr); all_r0.append(fr0); all_z0.append(fz0); all_r1.append(fr1); all_z1.append(fz1); all_t.append(ft); all_grade.append(fgrade); all_loo.append(floo); all_neigh.append(fneigh); all_spacing.append(fspacing); all_radial_spacing.append(frspacing)
            support_count=len(fx); ring_final=int((fst==3).sum()); generic_final=int((fst==4).sum())
        else:
            support_count=ring_final=generic_final=0
        family_stage_timing["total_s"] = time.perf_counter() - family_started
        family_meta[str(family_id)]={"analysis_points":int(len(pts)),"raw_points":int(len(raw_pts)),"support":support_count,"ring_support":ring_final,"generic_support":generic_final,"ring":ring_meta,"timing_s":family_stage_timing}
        ac = ring_meta.get("anchor_cleaning", {}) if isinstance(ring_meta, dict) else {}
        print(f"[scene support] family={family_id}: raw={len(raw_pts):,} centers={len(pts):,} accepted={support_count:,} ring-direct-Z={ring_final:,} generic={generic_final:,} | anchors={ac.get('total_anchors',0):,} loo_candidates={ac.get('loo_candidates',0):,} rejected={ac.get('total_rejected',0):,}")
        family_timings[str(family_id)] = time.perf_counter() - family_started

    stage_started = time.perf_counter()
    if all_xyz:
        arrays=[np.concatenate(x) for x in (all_xyz,all_sem,all_fam,all_source,all_normal,all_rmse,all_nearest,all_gap,all_expected,all_ratio,all_r0,all_z0,all_r1,all_z1,all_t,all_grade,all_loo,all_neigh,all_spacing,all_radial_spacing)]
        sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing=arrays
        if args.deduplicate:
            dedup_spacing = args.dedup_spacing if args.dedup_spacing > 0 else 0.5 * (args.fill_spacing if args.fill_spacing_mode == "fixed" else args.adaptive_fill_min)
            keep=deduplicate_candidates(sx,st,dedup_spacing,priority=sgr)
            sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing=[a[keep] for a in (sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing)]
        if args.max_support > 0 and len(sx)>args.max_support:
            score=(st==3).astype(np.float32)*10.0+sgr; order=np.argsort(score)[::-1][:args.max_support]
            sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing=[a[order] for a in (sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing)]
    else:
        sx=np.empty((0,3),np.float32); ss=np.empty(0,np.int16); sf=np.empty(0,np.int8); st=np.empty(0,np.int8); sn=np.empty((0,3),np.float32)
        srmse=snear=sg=se=sgr=sr0=sz0=sr1=sz1=sit=sgrade=sloo=sneigh=sspacing=sradial_spacing=np.empty(0,np.float32)
    timing["global_concat_dedup_s"] = time.perf_counter() - stage_started

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    stage_started = time.perf_counter()
    support_arrays = dict(
        xyz=sx.astype(np.float32), semantic_id=ss.astype(np.int16), ground_family=sf.astype(np.int8), source_type=st.astype(np.int8),
        plane_normal=sn.astype(np.float32), plane_rmse=srmse.astype(np.float32), nearest_observed_xy=snear.astype(np.float32),
        ring_gap_m=sg.astype(np.float32), ring_expected_gap_m=se.astype(np.float32), ring_gap_ratio=sgr.astype(np.float32),
        ring_inner_r_m=sr0.astype(np.float32), ring_inner_z_m=sz0.astype(np.float32), ring_outer_r_m=sr1.astype(np.float32), ring_outer_z_m=sz1.astype(np.float32),
        ring_interp_t=sit.astype(np.float32), ring_grade_percent=sgrade.astype(np.float32),
        ring_anchor_loo_max_m=sloo.astype(np.float32), ring_anchor_neighbor_max_m=sneigh.astype(np.float32),
        local_fill_spacing_m=sspacing.astype(np.float32),
        local_radial_spacing_m=sradial_spacing.astype(np.float32),
        sensor_origin_world=np.asarray(tstats["center"],dtype=np.float64), ring_mode_active=np.asarray([int(ring_active)],dtype=np.int8))
    save_npz(out, support_arrays, compression=args.npz_compression)
    timing["write_support_npz_s"] = time.perf_counter() - stage_started
    meta={"caseid":args.caseid,"support_npz":str(out),"num_support":int(len(sx)),"num_ring_support":int((st==3).sum()),"num_generic_support":int((st==4).sum()),
          "method":"Robust same-sector measured ring interpolation; isolated bad anchors rejected; neighbouring sectors validate but never overwrite Z",
          "analysis_voxel_m":args.analysis_voxel,"fill_spacing_m":args.fill_spacing,"fill_spacing_mode":args.fill_spacing_mode,
          "adaptive_density_voxel_m":args.adaptive_density_voxel,"adaptive_density_neighbors":args.adaptive_density_neighbors,
          "adaptive_query_neighbors":args.adaptive_query_neighbors,"adaptive_fill_scale":args.adaptive_fill_scale,
          "adaptive_fill_min_m":args.adaptive_fill_min,"adaptive_fill_max_m":args.adaptive_fill_max,
          "ring_density_duplicate_threshold_m":args.ring_density_duplicate_threshold,
          "ring_density_min_unique_samples":args.ring_density_min_unique_samples,
          "radial_spacing_scale":args.radial_spacing_scale,
          "deduplicate":bool(args.deduplicate),"dedup_spacing_m":args.dedup_spacing,
          "full_resolution":bool(args.full_resolution),
          "disable_safety_filters":bool(args.disable_safety_filters),
          "max_support":args.max_support,"max_candidates_per_family":args.max_candidates_per_family,
          "ring_mode_requested":args.ring_mode,"ring_mode_active":bool(ring_active),
          "ring_anchor_loo_floor_m":args.ring_anchor_loo_floor_m,"ring_anchor_loo_sigma":args.ring_anchor_loo_sigma,"ring_neighbor_half_window":args.ring_neighbor_half_window,"ring_neighbor_z_tol_m":args.ring_neighbor_z_tol_m,"trajectory":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in tstats.items()},"families":family_meta,
          "source_type":{"0":"observed","3":"scene_ring_direct_interpolation","4":"scene_generic_coverage"}}
    with open(str(out)+".json","w") as f: json.dump(meta,f,indent=2)
    if np.any(st==3):
        g=np.abs(sgrade[st==3]); print(f"[scene support] ring anchor grade |%|: p50={np.median(g):.3f} p95={np.percentile(g,95):.3f} max={g.max():.3f}")
        sp=sspacing[st==3]
        print(f"[scene support] tangential fill spacing used: p10={np.percentile(sp,10):.3f}m p50={np.median(sp):.3f}m p90={np.percentile(sp,90):.3f}m min={sp.min():.3f}m max={sp.max():.3f}m")
        rsp=sradial_spacing[st==3]
        print(f"[scene support] radial fill spacing used:     p10={np.percentile(rsp,10):.3f}m p50={np.median(rsp):.3f}m p90={np.percentile(rsp,90):.3f}m min={rsp.min():.3f}m max={rsp.max():.3f}m")
    print(f"[scene support] wrote {len(sx):,} support points: ring-direct-Z={(st==3).sum():,}, generic={(st==4).sum():,} -> {out}")


    # Optional one-step densified export.
    if args.densified_pcd is not None or args.densified_npz is not None:
        print("[densified export] mapping generated support to original metadata...")

        with np.load(strict_path, allow_pickle=False) as d:
            original_xyz_for_metadata = np.asarray(d["xyz"], dtype=np.float32)
            original_sem_for_metadata = np.asarray(d["semantic_id"], dtype=np.int16)

        stage_started = time.perf_counter()
        support_source_index = assign_support_source_indices(
            original_xyz_for_metadata,
            original_sem_for_metadata,
            sx.astype(np.float32),
            ss.astype(np.int16),
        )
        timing["metadata_source_mapping_s"] = time.perf_counter() - stage_started

        if args.densified_pcd is not None:
            stage_started = time.perf_counter()
            export_densified_pcd(
                strict_path,
                sx.astype(np.float32),
                ss.astype(np.int16),
                support_source_index,
                args.densified_pcd,
                chunk_size=args.pcd_chunk_size,
            )
            timing["write_densified_pcd_s"] = time.perf_counter() - stage_started

        if args.densified_npz is not None:
            stage_started = time.perf_counter()
            export_densified_npz(
                strict_path,
                sx.astype(np.float32),
                ss.astype(np.int16),
                st.astype(np.int8),
                sf.astype(np.int8),
                args.densified_npz,
                source_index=support_source_index,
                compression=args.npz_compression,
            )
            timing["write_densified_npz_s"] = time.perf_counter() - stage_started

    timing["family_total_s"] = float(sum(family_timings.values()))
    timing["families_s"] = family_timings
    timing["total_s"] = time.perf_counter() - total_started
    print("[timing] " + json.dumps(timing, sort_keys=True))
    if args.timing_json:
        timing_path = Path(args.timing_json)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w") as stream:
            json.dump(timing, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

# CASE=segment-17791493328130181905_1480_000_1500_000_with_camera_labels

# META_ROOT=/data/waymo/road_reconstruction_study
# DATASET_ROOT=/data/waymo/road_reconstruction_study

# STATIC_DIR=$DATASET_ROOT/recon_related/$CASE/static_filter


# python convert_static_npz_to_densified_pcd.py \
#   -s $META_ROOT \
#   --caseid $CASE \
#   --strict_path $STATIC_DIR/static_recon_labels_strict.npz \
#   -o $STATIC_DIR/scene_ground_support.npz \
#   --generic_coverage \
#   --densified_pcd $STATIC_DIR/static_recon_semantic_intensity_strict_densified.pcd \
#   --densified_npz $STATIC_DIR/static_recon_semantic_intensity_strict_densified.npz