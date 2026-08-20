#!/usr/bin/env python3

"""
FINAL controlled KNN semantic-label propagation experiment.

Controlled variables ONLY:
    K = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    distance threshold = 0.4, 0.5, ..., 1.4 m

Everything else is fixed:
    - alternating held-out split of dynamically detected labeled frames
    - STATIC points only
    - equal-weight majority vote
    - minimum valid neighbors = 1
    - ties -> UNASSIGNED (-1)
    - no voxelization
    - no downsampling
    - no statistical/radius filtering

Outputs:
    final_knn_summary.csv
    final_knn_per_frame.csv
    final_knn_classwise.csv
    final_knn_confusion_matrices.npz
    final_knn_best.json
    final_knn_report.txt
    final_knn_split.json

The script selects ONE final best configuration by:
    1. highest effective accuracy
    2. then highest mIoU excluding Waymo class 0
    3. then highest coverage
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial import cKDTree


UNASSIGNED = -1

CLASS_NAMES = {
    0: "UNDEFINED",
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "OTHER_VEHICLE",
    5: "MOTORCYCLIST",
    6: "BICYCLIST",
    7: "PEDESTRIAN",
    8: "SIGN",
    9: "TRAFFIC_LIGHT",
    10: "POLE",
    11: "CONSTRUCTION_CONE",
    12: "BICYCLE",
    13: "MOTORCYCLE",
    14: "BUILDING",
    15: "VEGETATION",
    16: "TREE_TRUNK",
    17: "CURB",
    18: "ROAD",
    19: "LANE_MARKER",
    20: "OTHER_GROUND",
    21: "WALKABLE",
    22: "SIDEWALK",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def find_labeled_frames(association_root):
    frames_dir = os.path.join(
        association_root,
        "01_labeled_frames",
    )

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"Missing labeled frame directory:\n{frames_dir}"
        )

    frames = []

    for name in os.listdir(frames_dir):
        if not (
            name.startswith("frame_")
            and name.endswith(".npz")
        ):
            continue

        try:
            frames.append(
                int(name[6:-4])
            )
        except ValueError:
            pass

    frames.sort()

    if len(frames) < 2:
        raise RuntimeError(
            "Need at least two labeled frames."
        )

    return frames, frames_dir


def split_alternating(labeled_frames):
    """
    Split by position in the sorted list, not by absolute frame number.

    This handles non-contiguous segmentation availability.
    """
    reference_frames = labeled_frames[0::2]
    evaluation_frames = labeled_frames[1::2]

    return reference_frames, evaluation_frames


def load_frame(frames_dir, frame_idx):
    path = os.path.join(
        frames_dir,
        f"frame_{frame_idx:03d}.npz",
    )

    d = np.load(path)

    required = [
        "xyz_world",
        "semantic_id",
        "is_dynamic_track_point",
    ]

    for key in required:
        if key not in d.files:
            raise KeyError(
                f"Missing '{key}' in {path}. "
                f"Available keys: {d.files}"
            )

    return {
        "xyz_world": d["xyz_world"].astype(
            np.float32
        ),
        "semantic_id": d["semantic_id"].astype(
            np.int32
        ),
        "is_dynamic": d[
            "is_dynamic_track_point"
        ].astype(bool),
    }


def build_static_reference(
    frames_dir,
    reference_frames,
):
    xyz_parts = []
    semantic_parts = []

    print()
    print("Building held-in STATIC reference cloud...")

    for frame_idx in reference_frames:
        d = load_frame(
            frames_dir,
            frame_idx,
        )

        keep = ~d["is_dynamic"]

        xyz = d["xyz_world"][keep]
        semantic = d["semantic_id"][keep]

        xyz_parts.append(xyz)
        semantic_parts.append(semantic)

        print(
            f"  [{frame_idx:03d}] "
            f"{len(xyz):,} static labeled points"
        )

    return (
        np.concatenate(
            xyz_parts,
            axis=0,
        ),
        np.concatenate(
            semantic_parts,
            axis=0,
        ),
    )


# ---------------------------------------------------------------------
# KNN voting
# ---------------------------------------------------------------------

def majority_vote_batch(
    neighbor_labels,
    neighbor_distances,
    k,
    threshold,
):
    """
    Equal-weight majority vote.

    Rules:
      - use first K nearest neighbors
      - reject individual neighbors farther than threshold
      - require at least 1 valid neighbor
      - unique majority -> predicted semantic class
      - tie -> UNASSIGNED (-1)
      - zero valid neighbors -> UNASSIGNED (-1)
    """

    labels = neighbor_labels[:, :k]
    distances = neighbor_distances[:, :k]

    within = (
        distances <= threshold
    )

    num_valid_neighbors = (
        within.sum(axis=1)
    )

    pred = np.full(
        len(labels),
        UNASSIGNED,
        dtype=np.int32,
    )

    agreement = np.zeros(
        len(labels),
        dtype=np.float32,
    )

    eligible_rows = np.flatnonzero(
        num_valid_neighbors >= 1
    )

    for row in eligible_rows:
        labs = labels[row][within[row]]

        unique_labels, counts = np.unique(
            labs,
            return_counts=True,
        )

        max_votes = counts.max()

        winners = unique_labels[
            counts == max_votes
        ]

        if len(winners) != 1:
            # Ambiguous tie.
            continue

        chosen = int(
            winners[0]
        )

        pred[row] = chosen

        agreement[row] = float(
            max_votes / len(labs)
        )

    valid = (
        pred != UNASSIGNED
    )

    return (
        pred,
        valid,
        agreement,
        num_valid_neighbors,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def classwise_metrics(
    gt,
    pred,
    class_ids,
):
    rows = []

    for class_id in class_ids:
        c = int(class_id)

        tp = int(
            np.count_nonzero(
                (gt == c)
                & (pred == c)
            )
        )

        fp = int(
            np.count_nonzero(
                (gt != c)
                & (pred == c)
            )
        )

        # Wrong class OR unassigned both count as FN.
        fn = int(
            np.count_nonzero(
                (gt == c)
                & (pred != c)
            )
        )

        gt_count = int(
            np.count_nonzero(
                gt == c
            )
        )

        pred_count = int(
            np.count_nonzero(
                pred == c
            )
        )

        iou_den = (
            tp + fp + fn
        )

        precision_den = (
            tp + fp
        )

        recall_den = (
            tp + fn
        )

        iou = (
            tp / iou_den
            if iou_den > 0
            else None
        )

        precision = (
            tp / precision_den
            if precision_den > 0
            else None
        )

        recall = (
            tp / recall_den
            if recall_den > 0
            else None
        )

        rows.append(
            {
                "semantic_id": c,
                "class_name": CLASS_NAMES.get(
                    c,
                    f"CLASS_{c}",
                ),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            }
        )

    return rows


def aggregate_metrics(
    gt,
    pred,
    class_ids,
):
    valid = (
        pred != UNASSIGNED
    )

    total = len(gt)

    assigned = int(
        np.count_nonzero(
            valid
        )
    )

    correct = int(
        np.count_nonzero(
            valid
            & (pred == gt)
        )
    )

    coverage = (
        assigned / total
        if total > 0
        else 0.0
    )

    assigned_accuracy = (
        correct / assigned
        if assigned > 0
        else 0.0
    )

    effective_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    per_class = classwise_metrics(
        gt,
        pred,
        class_ids,
    )

    ious_all = [
        row["iou"]
        for row in per_class
        if row["iou"] is not None
    ]

    ious_nonzero = [
        row["iou"]
        for row in per_class
        if (
            row["semantic_id"] != 0
            and row["iou"] is not None
        )
    ]

    return {
        "coverage": float(
            coverage
        ),
        "assigned_accuracy": float(
            assigned_accuracy
        ),
        "effective_accuracy": float(
            effective_accuracy
        ),
        "miou_all": float(
            np.mean(ious_all)
        ) if ious_all else 0.0,
        "miou_nonzero": float(
            np.mean(ious_nonzero)
        ) if ious_nonzero else 0.0,
        "assigned_points": assigned,
        "correct_points": correct,
        "total_points": int(
            total
        ),
        "unassigned_points": int(
            total - assigned
        ),
        "classwise": per_class,
    }


def make_confusion_matrix(
    gt,
    pred,
    class_ids,
):
    class_to_index = {
        int(c): i
        for i, c in enumerate(
            class_ids
        )
    }

    # Extra final column for UNASSIGNED.
    matrix = np.zeros(
        (
            len(class_ids),
            len(class_ids) + 1,
        ),
        dtype=np.int64,
    )

    unassigned_col = len(
        class_ids
    )

    for g, p in zip(
        gt,
        pred,
    ):
        gi = class_to_index.get(
            int(g)
        )

        if gi is None:
            continue

        if int(p) == UNASSIGNED:
            pi = unassigned_col
        else:
            pi = class_to_index.get(
                int(p),
                unassigned_col,
            )

        matrix[
            gi,
            pi
        ] += 1

    return matrix


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final controlled KNN semantic association experiment: "
            "K=1..10 and threshold=0.4..1.4 m."
        )
    )

    parser.add_argument(
        "--association-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help=(
            "Memory-management only. "
            "Does not change the KNN result."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    association_root = os.path.abspath(
        args.association_root
    )

    output_dir = (
        os.path.abspath(
            args.output_dir
        )
        if args.output_dir
        else os.path.join(
            association_root,
            "07_final_controlled_knn",
        )
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # FIXED CONTROLLED EXPERIMENT GRID
    # -------------------------------------------------------------

    k_values = list(
        range(
            1,
            11,
        )
    )

    thresholds = [
        round(
            x,
            1,
        )
        for x in np.arange(
            0.4,
            1.41,
            0.1,
        )
    ]

    print("=" * 88)
    print("FINAL CONTROLLED KNN EXPERIMENT")
    print("=" * 88)

    print(
        "K values:",
        k_values,
    )

    print(
        "Distance thresholds [m]:",
        thresholds,
    )

    print(
        "Tie policy: UNASSIGNED (-1)"
    )

    print(
        "Minimum valid neighbors: 1 (fixed)"
    )

    print()

    # -------------------------------------------------------------
    # Detect and split labeled frames
    # -------------------------------------------------------------

    (
        labeled_frames,
        frames_dir,
    ) = find_labeled_frames(
        association_root
    )

    (
        reference_frames,
        evaluation_frames,
    ) = split_alternating(
        labeled_frames
    )

    print(
        "All labeled frames:",
        labeled_frames,
    )

    print()

    print(
        "Reference frames:",
        reference_frames,
    )

    print()

    print(
        "Evaluation frames:",
        evaluation_frames,
    )

    # -------------------------------------------------------------
    # Build held-in static reference
    # -------------------------------------------------------------

    (
        reference_xyz,
        reference_semantic,
    ) = build_static_reference(
        frames_dir,
        reference_frames,
    )

    print()

    print(
        "Reference points:",
        f"{len(reference_xyz):,}",
    )

    print(
        "Building KD-tree..."
    )

    tree = cKDTree(
        reference_xyz.astype(
            np.float64
        )
    )

    print(
        "KD-tree ready."
    )

    # -------------------------------------------------------------
    # Determine class IDs
    # -------------------------------------------------------------

    class_id_set = set(
        int(x)
        for x in np.unique(
            reference_semantic
        )
    )

    for frame_idx in evaluation_frames:
        d = load_frame(
            frames_dir,
            frame_idx,
        )

        keep = ~d[
            "is_dynamic"
        ]

        class_id_set.update(
            int(x)
            for x in np.unique(
                d[
                    "semantic_id"
                ][keep]
            )
        )

    class_ids = np.asarray(
        sorted(
            class_id_set
        ),
        dtype=np.int32,
    )

    print()

    print(
        "Semantic classes evaluated:"
    )

    for c in class_ids:
        print(
            f"  {int(c):>2}: "
            f"{CLASS_NAMES.get(int(c), 'UNKNOWN')}"
        )

    # -------------------------------------------------------------
    # Evaluation state
    # -------------------------------------------------------------

    configs = [
        (
            k,
            threshold,
        )
        for k in k_values
        for threshold in thresholds
    ]

    aggregate = {
        config: {
            "gt": [],
            "pred": [],
            "agreement_sum": 0.0,
            "assigned_count": 0,
        }
        for config in configs
    }

    per_frame_rows = []

    k_max = 10

    print()

    print("=" * 88)
    print("HELD-OUT EVALUATION")
    print("=" * 88)

    # -------------------------------------------------------------
    # Evaluate frame by frame
    # -------------------------------------------------------------

    for eval_num, frame_idx in enumerate(
        evaluation_frames,
        start=1,
    ):

        d = load_frame(
            frames_dir,
            frame_idx,
        )

        keep = ~d[
            "is_dynamic"
        ]

        xyz = d[
            "xyz_world"
        ][keep]

        gt = d[
            "semantic_id"
        ][keep]

        print(
            f"\n[{frame_idx:03d}] "
            f"{len(xyz):,} static GT points "
            f"({eval_num}/{len(evaluation_frames)})"
        )

        frame_store = {
            config: {
                "gt": [],
                "pred": [],
                "agreement_sum": 0.0,
                "assigned_count": 0,
            }
            for config in configs
        }

        for start in range(
            0,
            len(xyz),
            args.chunk_size,
        ):

            end = min(
                start + args.chunk_size,
                len(xyz),
            )

            query_xyz = xyz[
                start:end
            ].astype(
                np.float64
            )

            gt_chunk = gt[
                start:end
            ]

            distances, indices = tree.query(
                query_xyz,
                k=k_max,
                workers=args.workers,
            )

            neighbor_labels = (
                reference_semantic[
                    indices
                ]
            )

            # Reuse the same 10-NN result for all 110 configs.
            for config in configs:

                k, threshold = config

                (
                    pred,
                    valid,
                    agreement,
                    _,
                ) = majority_vote_batch(
                    neighbor_labels,
                    distances,
                    k,
                    threshold,
                )

                for store in (
                    aggregate[
                        config
                    ],
                    frame_store[
                        config
                    ],
                ):

                    store[
                        "gt"
                    ].append(
                        gt_chunk.copy()
                    )

                    store[
                        "pred"
                    ].append(
                        pred.copy()
                    )

                    store[
                        "agreement_sum"
                    ] += float(
                        agreement[
                            valid
                        ].sum()
                    )

                    store[
                        "assigned_count"
                    ] += int(
                        np.count_nonzero(
                            valid
                        )
                    )

        # Per-frame metrics for every configuration.
        for config in configs:

            k, threshold = config

            gt_frame = np.concatenate(
                frame_store[
                    config
                ][
                    "gt"
                ]
            )

            pred_frame = np.concatenate(
                frame_store[
                    config
                ][
                    "pred"
                ]
            )

            metrics = aggregate_metrics(
                gt_frame,
                pred_frame,
                class_ids,
            )

            mean_agreement = (
                frame_store[
                    config
                ][
                    "agreement_sum"
                ]
                / frame_store[
                    config
                ][
                    "assigned_count"
                ]
                if frame_store[
                    config
                ][
                    "assigned_count"
                ] > 0
                else 0.0
            )

            per_frame_rows.append(
                {
                    "frame_idx": frame_idx,
                    "k": k,
                    "distance_threshold_m": threshold,
                    "coverage": metrics[
                        "coverage"
                    ],
                    "assigned_accuracy": metrics[
                        "assigned_accuracy"
                    ],
                    "effective_accuracy": metrics[
                        "effective_accuracy"
                    ],
                    "miou_all": metrics[
                        "miou_all"
                    ],
                    "miou_nonzero": metrics[
                        "miou_nonzero"
                    ],
                    "assigned_points": metrics[
                        "assigned_points"
                    ],
                    "unassigned_points": metrics[
                        "unassigned_points"
                    ],
                    "mean_vote_agreement": mean_agreement,
                }
            )

    # -------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------

    summary_rows = []
    classwise_rows = []
    confusion_payload = {
        "class_ids": class_ids,
    }

    for config in configs:

        k, threshold = config

        gt_all = np.concatenate(
            aggregate[
                config
            ][
                "gt"
            ]
        )

        pred_all = np.concatenate(
            aggregate[
                config
            ][
                "pred"
            ]
        )

        metrics = aggregate_metrics(
            gt_all,
            pred_all,
            class_ids,
        )

        mean_agreement = (
            aggregate[
                config
            ][
                "agreement_sum"
            ]
            / aggregate[
                config
            ][
                "assigned_count"
            ]
            if aggregate[
                config
            ][
                "assigned_count"
            ] > 0
            else 0.0
        )

        summary_rows.append(
            {
                "k": k,
                "distance_threshold_m": threshold,
                "coverage": metrics[
                    "coverage"
                ],
                "assigned_accuracy": metrics[
                    "assigned_accuracy"
                ],
                "effective_accuracy": metrics[
                    "effective_accuracy"
                ],
                "miou_all": metrics[
                    "miou_all"
                ],
                "miou_nonzero": metrics[
                    "miou_nonzero"
                ],
                "assigned_points": metrics[
                    "assigned_points"
                ],
                "correct_points": metrics[
                    "correct_points"
                ],
                "total_points": metrics[
                    "total_points"
                ],
                "unassigned_points": metrics[
                    "unassigned_points"
                ],
                "mean_vote_agreement": mean_agreement,
            }
        )

        for row in metrics[
            "classwise"
        ]:

            classwise_rows.append(
                {
                    "k": k,
                    "distance_threshold_m": threshold,
                    **row,
                }
            )

        confusion_payload[
            f"k{k}_thr{threshold:.1f}"
        ] = make_confusion_matrix(
            gt_all,
            pred_all,
            class_ids,
        )

    # Final ranking.
    summary_rows.sort(
        key=lambda r: (
            r[
                "effective_accuracy"
            ],
            r[
                "miou_nonzero"
            ],
            r[
                "coverage"
            ],
        ),
        reverse=True,
    )

    best = summary_rows[
        0
    ]

    # -------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------

    def save_csv(
        path,
        rows,
    ):

        with open(
            path,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    summary_path = os.path.join(
        output_dir,
        "final_knn_summary.csv",
    )

    per_frame_path = os.path.join(
        output_dir,
        "final_knn_per_frame.csv",
    )

    classwise_path = os.path.join(
        output_dir,
        "final_knn_classwise.csv",
    )

    confusion_path = os.path.join(
        output_dir,
        "final_knn_confusion_matrices.npz",
    )

    best_path = os.path.join(
        output_dir,
        "final_knn_best.json",
    )

    split_path = os.path.join(
        output_dir,
        "final_knn_split.json",
    )

    report_path = os.path.join(
        output_dir,
        "final_knn_report.txt",
    )

    save_csv(
        summary_path,
        summary_rows,
    )

    save_csv(
        per_frame_path,
        per_frame_rows,
    )

    save_csv(
        classwise_path,
        classwise_rows,
    )

    np.savez(
        confusion_path,
        **confusion_payload,
    )

    # -------------------------------------------------------------
    # Best config detailed class-wise metrics
    # -------------------------------------------------------------

    best_classwise = [
        row
        for row in classwise_rows
        if (
            row[
                "k"
            ] == best[
                "k"
            ]
            and abs(
                row[
                    "distance_threshold_m"
                ]
                - best[
                    "distance_threshold_m"
                ]
            ) < 1e-9
        )
    ]

    best_json = {
        "selection_rule": [
            "highest effective_accuracy",
            "then highest miou_nonzero",
            "then highest coverage",
        ],
        "best_configuration": best,
        "classwise_metrics": best_classwise,
    }

    with open(
        best_path,
        "w",
    ) as f:

        json.dump(
            best_json,
            f,
            indent=2,
        )

    with open(
        split_path,
        "w",
    ) as f:

        json.dump(
            {
                "all_labeled_frames": labeled_frames,
                "reference_frames": reference_frames,
                "evaluation_frames": evaluation_frames,
                "k_values": k_values,
                "distance_thresholds_m": thresholds,
                "minimum_valid_neighbors": 1,
                "vote_type": "equal-weight majority",
                "tie_policy": "UNASSIGNED (-1)",
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Human-readable report
    # -------------------------------------------------------------

    lines = []

    lines.append(
        "FINAL CONTROLLED KNN SEMANTIC ASSOCIATION EXPERIMENT"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        ""
    )

    lines.append(
        "Controlled grid:"
    )

    lines.append(
        "  K = 1,2,3,4,5,6,7,8,9,10"
    )

    lines.append(
        "  threshold = 0.4 to 1.4 m in 0.1 m increments"
    )

    lines.append(
        "  tie -> UNASSIGNED (-1)"
    )

    lines.append(
        "  minimum valid neighbors = 1"
    )

    lines.append(
        ""
    )

    lines.append(
        "BEST CONFIGURATION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"K                    : {best['k']}"
    )

    lines.append(
        f"distance threshold   : "
        f"{best['distance_threshold_m']:.1f} m"
    )

    lines.append(
        f"coverage             : "
        f"{100*best['coverage']:.3f}%"
    )

    lines.append(
        f"assigned accuracy    : "
        f"{100*best['assigned_accuracy']:.3f}%"
    )

    lines.append(
        f"effective accuracy   : "
        f"{100*best['effective_accuracy']:.3f}%"
    )

    lines.append(
        f"mIoU including ID 0  : "
        f"{100*best['miou_all']:.3f}%"
    )

    lines.append(
        f"mIoU excluding ID 0  : "
        f"{100*best['miou_nonzero']:.3f}%"
    )

    lines.append(
        f"vote agreement       : "
        f"{100*best['mean_vote_agreement']:.3f}%"
    )

    lines.append(
        f"unassigned points    : "
        f"{best['unassigned_points']:,}"
    )

    lines.append(
        ""
    )

    lines.append(
        "CLASS-WISE RESULTS FOR BEST CONFIGURATION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"{'ID':>3}  {'CLASS':<20} "
        f"{'IoU':>9} {'Precision':>10} {'Recall':>10} "
        f"{'GT points':>12}"
    )

    for row in best_classwise:

        iou = (
            "-"
            if row["iou"] is None
            else f"{100*row['iou']:.2f}%"
        )

        precision = (
            "-"
            if row["precision"] is None
            else f"{100*row['precision']:.2f}%"
        )

        recall = (
            "-"
            if row["recall"] is None
            else f"{100*row['recall']:.2f}%"
        )

        lines.append(
            f"{row['semantic_id']:>3}  "
            f"{row['class_name']:<20} "
            f"{iou:>9} "
            f"{precision:>10} "
            f"{recall:>10} "
            f"{row['gt_count']:>12,}"
        )

    lines.append(
        ""
    )

    lines.append(
        "TOP 15 CONFIGURATIONS"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"{'K':>3} {'thr':>5} {'coverage':>10} "
        f"{'acc(valid)':>11} {'eff.acc':>10} {'mIoU(no0)':>11}"
    )

    for row in summary_rows[
        :15
    ]:

        lines.append(
            f"{row['k']:>3d} "
            f"{row['distance_threshold_m']:>5.1f} "
            f"{100*row['coverage']:>9.2f}% "
            f"{100*row['assigned_accuracy']:>10.2f}% "
            f"{100*row['effective_accuracy']:>9.2f}% "
            f"{100*row['miou_nonzero']:>10.2f}%"
        )

    report = "\n".join(
        lines
    )

    with open(
        report_path,
        "w",
    ) as f:

        f.write(
            report
        )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    print()
    print("=" * 88)
    print("FINAL BEST CONFIGURATION")
    print("=" * 88)

    print(
        f"K                  : {best['k']}"
    )

    print(
        f"Threshold          : "
        f"{best['distance_threshold_m']:.1f} m"
    )

    print(
        f"Coverage           : "
        f"{100*best['coverage']:.3f}%"
    )

    print(
        f"Assigned accuracy  : "
        f"{100*best['assigned_accuracy']:.3f}%"
    )

    print(
        f"Effective accuracy : "
        f"{100*best['effective_accuracy']:.3f}%"
    )

    print(
        f"mIoU (no ID 0)     : "
        f"{100*best['miou_nonzero']:.3f}%"
    )

    print()

    print(
        "Class-wise IoU:"
    )

    for row in best_classwise:

        if row["iou"] is None:
            iou_text = "N/A"
        else:
            iou_text = (
                f"{100*row['iou']:.2f}%"
            )

        print(
            f"  {row['semantic_id']:>2} "
            f"{row['class_name']:<20} "
            f"{iou_text:>8}"
        )

    print()

    print(
        "Saved:"
    )

    print(
        summary_path
    )

    print(
        per_frame_path
    )

    print(
        classwise_path
    )

    print(
        confusion_path
    )

    print(
        best_path
    )

    print(
        report_path
    )

    print(
        split_path
    )


if __name__ == "__main__":
    main()
