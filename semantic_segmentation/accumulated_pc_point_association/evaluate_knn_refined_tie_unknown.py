#!/usr/bin/env python3

import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial import cKDTree

PRED_UNASSIGNED = -1


def parse_int_list(text):
    vals = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if not vals:
        raise ValueError("Empty K list.")
    return vals


def parse_float_list(text):
    vals = sorted(set(float(x.strip()) for x in text.split(",") if x.strip()))
    if not vals:
        raise ValueError("Empty threshold list.")
    return vals


def find_labeled_frames(association_root):
    frames_dir = os.path.join(association_root, "01_labeled_frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(frames_dir)

    frames = []
    for name in os.listdir(frames_dir):
        if name.startswith("frame_") and name.endswith(".npz"):
            try:
                frames.append(int(name[6:-4]))
            except ValueError:
                pass

    frames.sort()
    if len(frames) < 2:
        raise RuntimeError("Need at least two labeled frames.")
    return frames, frames_dir


def split_alternating(frames):
    return frames[0::2], frames[1::2]


def load_frame(frames_dir, frame_idx):
    path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.npz")
    d = np.load(path)
    return {
        "xyz_world": d["xyz_world"].astype(np.float32),
        "semantic_id": d["semantic_id"].astype(np.int32),
        "is_dynamic": d["is_dynamic_track_point"].astype(bool),
    }


def build_reference(frames_dir, reference_frames):
    xyz_parts = []
    sem_parts = []

    print("\nBuilding held-in STATIC reference cloud...")

    for frame_idx in reference_frames:
        d = load_frame(frames_dir, frame_idx)
        keep = ~d["is_dynamic"]

        xyz = d["xyz_world"][keep]
        sem = d["semantic_id"][keep]

        xyz_parts.append(xyz)
        sem_parts.append(sem)

        print(f"  [{frame_idx:03d}] {len(xyz):,} static labeled points")

    return (
        np.concatenate(xyz_parts, axis=0),
        np.concatenate(sem_parts, axis=0),
    )


def vote_batch(neighbor_labels, neighbor_distances, k, threshold, min_neighbors):
    """
    Equal-weight majority vote.

    Tie handling:
      if two or more classes share the largest vote count, prediction=-1.
    """
    labels = neighbor_labels[:, :k]
    distances = neighbor_distances[:, :k]

    within = distances <= threshold
    n_valid_neighbors = within.sum(axis=1)

    pred = np.full(len(labels), PRED_UNASSIGNED, dtype=np.int32)
    agreement = np.zeros(len(labels), dtype=np.float32)

    rows = np.flatnonzero(n_valid_neighbors >= min_neighbors)

    for row in rows:
        labs = labels[row][within[row]]

        uniq, counts = np.unique(labs, return_counts=True)
        max_count = counts.max()
        winners = uniq[counts == max_count]

        if len(winners) != 1:
            continue

        chosen = int(winners[0])
        pred[row] = chosen
        agreement[row] = float(max_count / len(labs))

    valid = pred != PRED_UNASSIGNED
    return pred, valid, agreement


def compute_metrics(gt, pred, class_ids):
    valid = pred != PRED_UNASSIGNED

    total = len(gt)
    assigned = int(np.count_nonzero(valid))
    correct = int(np.count_nonzero(valid & (pred == gt)))

    coverage = assigned / total if total else 0.0
    assigned_accuracy = correct / assigned if assigned else 0.0
    effective_accuracy = correct / total if total else 0.0

    ious = []
    ious_nonzero = []

    for c in class_ids:
        c = int(c)
        tp = np.count_nonzero((gt == c) & (pred == c))
        fp = np.count_nonzero((gt != c) & (pred == c))
        fn = np.count_nonzero((gt == c) & (pred != c))

        denom = tp + fp + fn
        if denom == 0:
            continue

        iou = tp / denom
        ious.append(iou)

        if c != 0:
            ious_nonzero.append(iou)

    return {
        "coverage": float(coverage),
        "assigned_accuracy": float(assigned_accuracy),
        "effective_accuracy": float(effective_accuracy),
        "miou_all": float(np.mean(ious)) if ious else 0.0,
        "miou_nonzero": float(np.mean(ious_nonzero)) if ious_nonzero else 0.0,
        "assigned_points": assigned,
        "correct_points": correct,
        "total_points": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--association-root", required=True)

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--k-values",
        default="1,3,5,6,7,10",
    )

    parser.add_argument(
        "--distance-thresholds",
        default="0.30,0.40,0.50,0.60,0.70,0.80,1.00",
    )

    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    association_root = os.path.abspath(args.association_root)

    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(
            association_root,
            "05_knn_refined_evaluation",
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    k_values = parse_int_list(args.k_values)
    thresholds = parse_float_list(args.distance_thresholds)
    k_max = max(k_values)

    labeled_frames, frames_dir = find_labeled_frames(association_root)
    reference_frames, evaluation_frames = split_alternating(labeled_frames)

    print("=" * 80)
    print("HELD-OUT SPLIT")
    print("=" * 80)
    print("All labeled frames :", len(labeled_frames))
    print("Reference frames   :", len(reference_frames))
    print("Evaluation frames  :", len(evaluation_frames))
    print()
    print("Reference:", reference_frames)
    print()
    print("Evaluation:", evaluation_frames)

    ref_xyz, ref_semantic = build_reference(
        frames_dir,
        reference_frames,
    )

    print()
    print("Reference points:", f"{len(ref_xyz):,}")
    print("Building KD-tree...")
    tree = cKDTree(ref_xyz.astype(np.float64))
    print("KD-tree ready.")

    class_ids = set(int(x) for x in np.unique(ref_semantic))
    for frame_idx in evaluation_frames:
        d = load_frame(frames_dir, frame_idx)
        keep = ~d["is_dynamic"]
        class_ids.update(int(x) for x in np.unique(d["semantic_id"][keep]))

    class_ids = np.asarray(sorted(class_ids), dtype=np.int32)

    configs = [(k, t) for k in k_values for t in thresholds]

    aggregate = {
        config: {
            "gt": [],
            "pred": [],
            "agreement_sum": 0.0,
            "valid_count": 0,
        }
        for config in configs
    }

    per_frame_rows = []

    print()
    print("=" * 80)
    print("REFINED KNN GRID")
    print("=" * 80)

    for n_eval, frame_idx in enumerate(evaluation_frames, start=1):
        d = load_frame(frames_dir, frame_idx)
        keep = ~d["is_dynamic"]

        xyz = d["xyz_world"][keep].astype(np.float32)
        gt = d["semantic_id"][keep].astype(np.int32)

        print(
            f"\n[{frame_idx:03d}] static GT points: {len(xyz):,} "
            f"({n_eval}/{len(evaluation_frames)})"
        )

        frame_parts = {
            config: {"gt": [], "pred": [], "agreement_sum": 0.0, "valid_count": 0}
            for config in configs
        }

        chunk_size = max(1, int(args.chunk_size))

        for start in range(0, len(xyz), chunk_size):
            end = min(start + chunk_size, len(xyz))

            q = xyz[start:end].astype(np.float64)
            gt_chunk = gt[start:end]

            distances, indices = tree.query(
                q,
                k=k_max,
                workers=args.workers,
            )

            if k_max == 1:
                distances = distances[:, None]
                indices = indices[:, None]

            neighbor_semantic = ref_semantic[indices]

            for config in configs:
                k, threshold = config

                pred, valid, agreement = vote_batch(
                    neighbor_semantic,
                    distances,
                    k=k,
                    threshold=threshold,
                    min_neighbors=args.min_neighbors,
                )

                aggregate[config]["gt"].append(gt_chunk.copy())
                aggregate[config]["pred"].append(pred.copy())
                aggregate[config]["agreement_sum"] += float(agreement[valid].sum())
                aggregate[config]["valid_count"] += int(np.count_nonzero(valid))

                frame_parts[config]["gt"].append(gt_chunk.copy())
                frame_parts[config]["pred"].append(pred.copy())
                frame_parts[config]["agreement_sum"] += float(agreement[valid].sum())
                frame_parts[config]["valid_count"] += int(np.count_nonzero(valid))

        for config in configs:
            k, threshold = config
            gt_frame = np.concatenate(frame_parts[config]["gt"], axis=0)
            pred_frame = np.concatenate(frame_parts[config]["pred"], axis=0)

            metrics = compute_metrics(gt_frame, pred_frame, class_ids)

            mean_agreement = (
                frame_parts[config]["agreement_sum"]
                / frame_parts[config]["valid_count"]
                if frame_parts[config]["valid_count"] > 0
                else 0.0
            )

            per_frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "k": k,
                    "distance_threshold_m": threshold,
                    "min_neighbors": args.min_neighbors,
                    **metrics,
                    "mean_vote_agreement": mean_agreement,
                    "unassigned_count": int(
                        np.count_nonzero(pred_frame == PRED_UNASSIGNED)
                    ),
                }
            )

    summary_rows = []

    for config in configs:
        k, threshold = config

        gt_all = np.concatenate(aggregate[config]["gt"], axis=0)
        pred_all = np.concatenate(aggregate[config]["pred"], axis=0)

        metrics = compute_metrics(gt_all, pred_all, class_ids)

        mean_agreement = (
            aggregate[config]["agreement_sum"]
            / aggregate[config]["valid_count"]
            if aggregate[config]["valid_count"] > 0
            else 0.0
        )

        summary_rows.append(
            {
                "k": k,
                "distance_threshold_m": threshold,
                "min_neighbors": args.min_neighbors,
                **metrics,
                "mean_vote_agreement": mean_agreement,
                "unassigned_count": int(
                    np.count_nonzero(pred_all == PRED_UNASSIGNED)
                ),
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (
            r["effective_accuracy"],
            r["miou_nonzero"],
            r["coverage"],
        ),
        reverse=True,
    )

    summary_csv = os.path.join(output_dir, "knn_refined_summary.csv")
    per_frame_csv = os.path.join(output_dir, "knn_refined_per_frame.csv")
    summary_json = os.path.join(output_dir, "knn_refined_summary.json")
    split_json = os.path.join(output_dir, "split_manifest.json")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(per_frame_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    with open(summary_json, "w") as f:
        json.dump(
            {
                "prediction_unassigned_value": PRED_UNASSIGNED,
                "note": (
                    "Waymo semantic ID 0 remains TYPE_UNDEFINED. "
                    "Ambiguous/tied/insufficient KNN predictions use -1."
                ),
                "class_ids": [int(x) for x in class_ids],
                "results": summary_rows,
            },
            f,
            indent=2,
        )

    with open(split_json, "w") as f:
        json.dump(
            {
                "all_labeled_frames": labeled_frames,
                "reference_frames": reference_frames,
                "evaluation_frames": evaluation_frames,
                "k_values": k_values,
                "distance_thresholds_m": thresholds,
                "min_neighbors": args.min_neighbors,
                "tie_policy": "unassigned (-1)",
            },
            f,
            indent=2,
        )

    print()
    print("=" * 92)
    print("TOP CONFIGURATIONS")
    print("=" * 92)
    print(
        f"{'K':>3} {'thr(m)':>8} {'coverage':>10} {'acc(valid)':>11} "
        f"{'eff.acc':>10} {'mIoU(no0)':>11} {'agreement':>11} {'unassigned':>11}"
    )
    print("-" * 92)

    for row in summary_rows[:20]:
        print(
            f"{row['k']:>3d} "
            f"{row['distance_threshold_m']:8.3f} "
            f"{100.0 * row['coverage']:9.2f}% "
            f"{100.0 * row['assigned_accuracy']:10.2f}% "
            f"{100.0 * row['effective_accuracy']:9.2f}% "
            f"{100.0 * row['miou_nonzero']:10.2f}% "
            f"{100.0 * row['mean_vote_agreement']:10.2f}% "
            f"{row['unassigned_count']:11,d}"
        )

    print()
    print("Saved:")
    print(summary_csv)
    print(per_frame_csv)
    print(summary_json)
    print(split_json)


if __name__ == "__main__":
    main()