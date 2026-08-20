#!/usr/bin/env python3
import argparse, csv, json, os, pickle
import numpy as np
from scipy.spatial import cKDTree

PRED_UNASSIGNED = -1

def parse_int_list(s):
    return sorted(set(int(x) for x in s.split(",") if x.strip()))

def parse_float_list(s):
    return sorted(set(float(x) for x in s.split(",") if x.strip()))

def find_labeled_frames(root):
    d = os.path.join(root, "01_labeled_frames")
    frames = []
    for name in os.listdir(d):
        if name.startswith("frame_") and name.endswith(".npz"):
            try: frames.append(int(name[6:-4]))
            except ValueError: pass
    frames.sort()
    return frames, d

def split_alternating(frames):
    return frames[0::2], frames[1::2]

def load_labeled_frame(frames_dir, idx):
    p = os.path.join(frames_dir, f"frame_{idx:03d}.npz")
    d = np.load(p)
    return (
        d["xyz_world"].astype(np.float32),
        d["semantic_id"].astype(np.int32),
        d["is_dynamic_track_point"].astype(bool),
    )

def build_reference(frames_dir, frames):
    xyzs, sems = [], []
    for idx in frames:
        xyz, sem, dyn = load_labeled_frame(frames_dir, idx)
        keep = ~dyn
        xyzs.append(xyz[keep])
        sems.append(sem[keep])
        print(f"  [{idx:03d}] {keep.sum():,} static labeled points")
    return np.concatenate(xyzs), np.concatenate(sems)

def vote_batch(neighbor_labels, neighbor_distances, k, threshold, min_neighbors):
    labels = neighbor_labels[:, :k]
    distances = neighbor_distances[:, :k]
    within = distances <= threshold
    counts_valid = within.sum(axis=1)

    pred = np.full(len(labels), PRED_UNASSIGNED, dtype=np.int32)
    agreement = np.zeros(len(labels), dtype=np.float32)

    for r in np.flatnonzero(counts_valid >= min_neighbors):
        labs = labels[r][within[r]]
        uniq, counts = np.unique(labs, return_counts=True)
        mx = counts.max()
        winners = uniq[counts == mx]
        if len(winners) == 1:
            pred[r] = int(winners[0])
            agreement[r] = float(mx / len(labs))

    return pred, pred != PRED_UNASSIGNED, agreement

def classwise_metrics(gt, pred, class_ids):
    rows = []
    for c in class_ids:
        c = int(c)
        tp = int(np.sum((gt == c) & (pred == c)))
        fp = int(np.sum((gt != c) & (pred == c)))
        fn = int(np.sum((gt == c) & (pred != c)))
        gt_count = int(np.sum(gt == c))
        pred_count = int(np.sum(pred == c))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        rows.append({
            "semantic_id": c, "gt_count": gt_count, "pred_count": pred_count,
            "tp": tp, "fp": fp, "fn": fn,
            "iou": iou, "precision": precision, "recall": recall
        })
    return rows

def aggregate_metrics(gt, pred, class_ids):
    valid = pred != PRED_UNASSIGNED
    total = len(gt)
    assigned = int(valid.sum())
    correct = int(np.sum(valid & (pred == gt)))
    cls = classwise_metrics(gt, pred, class_ids)
    iou_all = [r["iou"] for r in cls if r["iou"] is not None]
    iou_no0 = [r["iou"] for r in cls if r["semantic_id"] != 0 and r["iou"] is not None]
    return {
        "coverage": assigned/total if total else 0.0,
        "assigned_accuracy": correct/assigned if assigned else 0.0,
        "effective_accuracy": correct/total if total else 0.0,
        "miou_all": float(np.mean(iou_all)) if iou_all else 0.0,
        "miou_nonzero": float(np.mean(iou_no0)) if iou_no0 else 0.0,
        "assigned_points": assigned,
        "correct_points": correct,
        "total_points": total,
        "unassigned_points": int((~valid).sum()),
        "classwise": cls,
    }

def confusion_matrix(gt, pred, class_ids):
    cmap = {int(c): i for i, c in enumerate(class_ids)}
    M = np.zeros((len(class_ids), len(class_ids)+1), dtype=np.int64)
    ucol = len(class_ids)
    for g, p in zip(gt, pred):
        gi = cmap.get(int(g))
        if gi is None: continue
        pi = ucol if int(p) == PRED_UNASSIGNED else cmap.get(int(p), ucol)
        M[gi, pi] += 1
    return M

def load_meta(lidargs_root, case):
    p = os.path.join(lidargs_root, "meta_infos", case + ".pkl")
    with open(p, "rb") as f: return pickle.load(f)

def load_dynamic_track_ids(lidargs_root, case):
    p = os.path.join(lidargs_root, "temp", case, "track_classification.json")
    with open(p) as f: d = json.load(f)
    return {tid for tid, info in d.items() if bool(info.get("dynamic", False))}

def box_mask(points, box):
    cx,cy,cz,l,w,h,yaw = [float(x) for x in box]
    dx,dy,dz = points[:,0]-cx, points[:,1]-cy, points[:,2]-cz
    c,s = np.cos(yaw), np.sin(yaw)
    xl = c*dx + s*dy
    yl = -s*dx + c*dy
    return (np.abs(xl)<=l/2) & (np.abs(yl)<=w/2) & (np.abs(dz)<=h/2)

def static_mask_from_meta(xyz_vehicle, frame_meta, dynamic_track_ids):
    keep = np.ones(len(xyz_vehicle), dtype=bool)
    obj = frame_meta.get("obj_label", {})
    boxes = np.asarray(obj.get("gt_boxes", np.empty((0,7))))
    tokens = np.asarray(obj.get("gt_boxes_token", np.empty((0,), dtype=str)))
    for b, tok in zip(boxes, tokens):
        if str(tok) in dynamic_track_ids:
            keep &= ~box_mask(xyz_vehicle, b)
    return keep

def transform_points(points, T):
    ph = np.c_[points.astype(np.float64), np.ones(len(points))]
    return (ph @ T.T)[:, :3].astype(np.float32)

def run_heldout_grid(root, outdir, k_values, thresholds, min_neighbors, chunk_size, workers):
    labeled_frames, frames_dir = find_labeled_frames(root)
    ref_frames, eval_frames = split_alternating(labeled_frames)

    print("="*80)
    print("PART A — HELD-OUT VALIDATION")
    print("="*80)
    print("Reference:", ref_frames)
    print("Evaluation:", eval_frames)

    ref_xyz, ref_sem = build_reference(frames_dir, ref_frames)
    tree = cKDTree(ref_xyz.astype(np.float64))

    class_set = set(map(int, np.unique(ref_sem)))
    for idx in eval_frames:
        xyz, sem, dyn = load_labeled_frame(frames_dir, idx)
        class_set.update(map(int, np.unique(sem[~dyn])))
    class_ids = np.asarray(sorted(class_set), dtype=np.int32)

    configs = [(k,t) for k in k_values for t in thresholds]
    agg = {cfg: {"gt": [], "pred": [], "agree": 0.0, "nvalid": 0} for cfg in configs}
    per_frame_rows = []
    kmax = max(k_values)

    for n, idx in enumerate(eval_frames, 1):
        xyz, gt, dyn = load_labeled_frame(frames_dir, idx)
        xyz, gt = xyz[~dyn], gt[~dyn]
        print(f"[{idx:03d}] {len(xyz):,} static GT points ({n}/{len(eval_frames)})")
        fstore = {cfg: {"gt": [], "pred": [], "agree": 0.0, "nvalid": 0} for cfg in configs}

        for st in range(0, len(xyz), chunk_size):
            en = min(st+chunk_size, len(xyz))
            dist, ind = tree.query(xyz[st:en].astype(np.float64), k=kmax, workers=workers)
            if kmax == 1:
                dist, ind = dist[:,None], ind[:,None]
            neigh = ref_sem[ind]
            gt_chunk = gt[st:en]

            for cfg in configs:
                k,t = cfg
                pred, valid, agreement = vote_batch(neigh, dist, k, t, min_neighbors)
                for s in (agg[cfg], fstore[cfg]):
                    s["gt"].append(gt_chunk.copy())
                    s["pred"].append(pred.copy())
                    s["agree"] += float(agreement[valid].sum())
                    s["nvalid"] += int(valid.sum())

        for cfg in configs:
            k,t = cfg
            g = np.concatenate(fstore[cfg]["gt"])
            p = np.concatenate(fstore[cfg]["pred"])
            m = aggregate_metrics(g,p,class_ids)
            per_frame_rows.append({
                "frame_idx": idx, "k": k, "distance_threshold_m": t,
                "coverage": m["coverage"], "assigned_accuracy": m["assigned_accuracy"],
                "effective_accuracy": m["effective_accuracy"],
                "miou_all": m["miou_all"], "miou_nonzero": m["miou_nonzero"],
                "assigned_points": m["assigned_points"], "unassigned_points": m["unassigned_points"],
                "mean_vote_agreement": fstore[cfg]["agree"]/fstore[cfg]["nvalid"] if fstore[cfg]["nvalid"] else 0.0
            })

    summary_rows, class_rows, confs = [], [], {}
    for cfg in configs:
        k,t = cfg
        g = np.concatenate(agg[cfg]["gt"])
        p = np.concatenate(agg[cfg]["pred"])
        m = aggregate_metrics(g,p,class_ids)
        row = {
            "k": k, "distance_threshold_m": t, "min_neighbors": min_neighbors,
            "coverage": m["coverage"], "assigned_accuracy": m["assigned_accuracy"],
            "effective_accuracy": m["effective_accuracy"], "miou_all": m["miou_all"],
            "miou_nonzero": m["miou_nonzero"], "assigned_points": m["assigned_points"],
            "correct_points": m["correct_points"], "total_points": m["total_points"],
            "unassigned_points": m["unassigned_points"],
            "mean_vote_agreement": agg[cfg]["agree"]/agg[cfg]["nvalid"] if agg[cfg]["nvalid"] else 0.0
        }
        summary_rows.append(row)
        for r in m["classwise"]:
            class_rows.append({"k": k, "distance_threshold_m": t, **r})
        confs[f"k{k}_thr{t:.3f}"] = confusion_matrix(g,p,class_ids)

    summary_rows.sort(key=lambda r:(r["effective_accuracy"], r["miou_nonzero"], r["coverage"]), reverse=True)
    best = summary_rows[0]

    def save_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    save_csv(os.path.join(outdir,"knn_summary.csv"), summary_rows)
    save_csv(os.path.join(outdir,"knn_classwise_metrics.csv"), class_rows)
    save_csv(os.path.join(outdir,"knn_per_frame_metrics.csv"), per_frame_rows)

    payload = {"class_ids": class_ids}
    payload.update(confs)
    np.savez(os.path.join(outdir,"knn_confusion_matrices.npz"), **payload)

    with open(os.path.join(outdir,"split_manifest.json"),"w") as f:
        json.dump({
            "all_labeled_frames": labeled_frames, "reference_frames": ref_frames,
            "evaluation_frames": eval_frames, "k_values": k_values,
            "distance_thresholds_m": thresholds, "min_neighbors": min_neighbors,
            "tie_policy": "unassigned (-1)", "best_configuration": best
        }, f, indent=2)

    print("\nBEST:", best)
    return best, labeled_frames

def run_whole_sequence(root, lidargs_root, case, outdir, k, threshold, min_neighbors, chunk_size, workers):
    print("\n"+"="*80)
    print("PART B — WHOLE-SEQUENCE STATIC COVERAGE")
    print("="*80)

    labeled_frames, frames_dir = find_labeled_frames(root)
    labeled_set = set(labeled_frames)

    print("Building full labeled STATIC reference from all labeled frames...")
    ref_xyz, ref_sem = build_reference(frames_dir, labeled_frames)
    tree = cKDTree(ref_xyz.astype(np.float64))

    meta = load_meta(lidargs_root, case)
    dyn_ids = load_dynamic_track_ids(lidargs_root, case)
    pcd_root = os.path.join(lidargs_root, "pcds_new", case)

    rows = []
    for idx, fm in enumerate(meta["frames"]):
        arr = np.load(os.path.join(pcd_root, f"{idx:03d}.npz"))["data"]
        xyzv = arr[:,:3].astype(np.float32)
        keep = static_mask_from_meta(xyzv, fm, dyn_ids)
        xyzv = xyzv[keep]
        T = np.asarray(fm["lidar2world"], dtype=np.float64).reshape(4,4)
        xyzw = transform_points(xyzv, T)

        preds, valids, dists, agrees = [], [], [], []
        for st in range(0, len(xyzw), chunk_size):
            en = min(st+chunk_size, len(xyzw))
            dist, ind = tree.query(xyzw[st:en].astype(np.float64), k=k, workers=workers)
            if k == 1:
                dist, ind = dist[:,None], ind[:,None]
            neigh = ref_sem[ind]
            pred, valid, agreement = vote_batch(neigh, dist, k, threshold, min_neighbors)
            preds.append(pred); valids.append(valid); dists.append(dist[:,0].astype(np.float32)); agrees.append(agreement)

        pred = np.concatenate(preds)
        valid = np.concatenate(valids)
        nd = np.concatenate(dists)
        agree = np.concatenate(agrees)

        vd = nd[valid]
        va = agree[valid]
        row = {
            "frame_idx": idx,
            "has_waymo_segmentation_gt": idx in labeled_set,
            "num_static_points": len(xyzw),
            "assigned_points": int(valid.sum()),
            "unassigned_points": int((~valid).sum()),
            "coverage": float(valid.mean()),
            "nn_distance_median_assigned": float(np.median(vd)) if len(vd) else None,
            "nn_distance_p95_assigned": float(np.percentile(vd,95)) if len(vd) else None,
            "mean_vote_agreement_assigned": float(np.mean(va)) if len(va) else None,
        }
        ids, cnts = np.unique(pred[valid], return_counts=True)
        row["predicted_semantic_histogram_json"] = json.dumps({str(int(a)):int(b) for a,b in zip(ids,cnts)})
        rows.append(row)
        print(f"[{idx:03d}] GT={'yes' if idx in labeled_set else 'no ':3s} | coverage={100*row['coverage']:6.2f}% | unassigned={row['unassigned_points']:,}")

    with open(os.path.join(outdir,"whole_sequence_frame_coverage.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    first_label, last_label = min(labeled_frames), max(labeled_frames)
    groups = {
        "before_labeled_span":[r for r in rows if r["frame_idx"] < first_label],
        "within_labeled_span":[r for r in rows if first_label <= r["frame_idx"] <= last_label],
        "after_labeled_span":[r for r in rows if r["frame_idx"] > last_label],
    }
    summary = {}
    for name, subset in groups.items():
        total = sum(r["num_static_points"] for r in subset)
        assigned = sum(r["assigned_points"] for r in subset)
        summary[name] = {
            "num_frames": len(subset),
            "frame_indices": [r["frame_idx"] for r in subset],
            "total_static_points": int(total),
            "assigned_points": int(assigned),
            "unassigned_points": int(total-assigned),
            "coverage": float(assigned/total) if total else None,
            "mean_frame_coverage": float(np.mean([r["coverage"] for r in subset])) if subset else None
        }

    with open(os.path.join(outdir,"whole_sequence_region_summary.json"),"w") as f:
        json.dump({
            "k": k, "distance_threshold_m": threshold, "min_neighbors": min_neighbors,
            "first_labeled_frame": first_label, "last_labeled_frame": last_label,
            "regions": summary
        }, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--association-root", required=True)
    ap.add_argument("--lidargs-root", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--k-values", default="1,3,5")
    ap.add_argument("--distance-thresholds", default="0.8,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--min-neighbors", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--final-k", type=int, default=None)
    ap.add_argument("--final-threshold", type=float, default=None)
    ap.add_argument("--skip-whole-sequence", action="store_true")
    args = ap.parse_args()

    outdir = args.output_dir or os.path.join(args.association_root, "06_knn_classwise_and_sequence")
    os.makedirs(outdir, exist_ok=True)

    best, labeled_frames = run_heldout_grid(
        args.association_root, outdir,
        parse_int_list(args.k_values), parse_float_list(args.distance_thresholds),
        args.min_neighbors, args.chunk_size, args.workers
    )

    if not args.skip_whole_sequence:
        final_k = args.final_k if args.final_k is not None else int(best["k"])
        final_thr = args.final_threshold if args.final_threshold is not None else float(best["distance_threshold_m"])
        print(f"\nWhole-sequence config: K={final_k}, threshold={final_thr:.3f} m")
        run_whole_sequence(
            args.association_root, args.lidargs_root, args.case, outdir,
            final_k, final_thr, args.min_neighbors, args.chunk_size, args.workers
        )

if __name__ == "__main__":
    main()