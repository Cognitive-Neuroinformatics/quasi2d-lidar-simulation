#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_sparse_reference(path):
    data = np.load(path)

    required = [
        "xyz_world",
        "semantic_id",
        "instance_id",
    ]

    for key in required:
        if key not in data.files:
            raise KeyError(
                f"Missing '{key}' in sparse reference. "
                f"Available keys: {data.files}"
            )

    xyz = data["xyz_world"].astype(np.float32)
    semantic = data["semantic_id"].astype(np.int32)
    instance = data["instance_id"].astype(np.int32)

    if not (
        len(xyz) == len(semantic) == len(instance)
    ):
        raise RuntimeError(
            "Sparse reference arrays have inconsistent lengths."
        )

    return xyz, semantic, instance


def load_dense_static_pcd(path):
    pcd = o3d.io.read_point_cloud(path)

    xyz = np.asarray(
        pcd.points,
        dtype=np.float32,
    )

    if len(xyz) == 0:
        raise RuntimeError(
            f"Dense point cloud is empty: {path}"
        )

    return xyz


def percentile_report(distances, percentiles):
    finite = distances[
        np.isfinite(distances)
    ]

    if len(finite) == 0:
        return {}

    report = {}

    for p in percentiles:
        report[str(p)] = float(
            np.percentile(
                finite,
                p,
            )
        )

    return report


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Nearest-neighbor label propagation from a sparse labeled "
            "static Waymo world to the dense accumulated static scene."
        )
    )

    parser.add_argument(
        "--dense-pcd",
        required=True,
        help="Dense static accumulated PCD.",
    )

    parser.add_argument(
        "--sparse-labeled",
        required=True,
        help=(
            "labeled_static_accumulated.npz produced by the "
            "semantic-reference extraction step."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help=(
            "Optional maximum NN distance in meters. "
            "If omitted, labels are still transferred for all points, "
            "but nn_valid remains True only if a later threshold is applied "
            "during analysis. Recommended first run: omit this and inspect "
            "distance statistics."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500000,
        help="Dense query chunk size. Default: 500000.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help=(
            "Number of workers for scipy cKDTree.query. "
            "-1 uses all available CPU cores."
        ),
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        args.output_dir,
        "static_labeled_dense.npz",
    )

    stats_path = os.path.join(
        args.output_dir,
        "nn_association_stats.json",
    )

    print("Loading sparse labeled reference:")
    print(args.sparse_labeled)

    sparse_xyz, sparse_semantic, sparse_instance = (
        load_sparse_reference(
            args.sparse_labeled
        )
    )

    print(
        "Sparse labeled points:",
        f"{len(sparse_xyz):,}",
    )

    print()
    print("Building KD-tree...")

    tree = cKDTree(
        sparse_xyz.astype(np.float64)
    )

    print("KD-tree ready.")

    print()
    print("Loading dense static scene:")
    print(args.dense_pcd)

    dense_xyz = load_dense_static_pcd(
        args.dense_pcd
    )

    num_dense = len(dense_xyz)

    print(
        "Dense target points:",
        f"{num_dense:,}",
    )

    nn_distance = np.empty(
        num_dense,
        dtype=np.float32,
    )

    nn_index = np.empty(
        num_dense,
        dtype=np.int64,
    )

    chunk_size = max(
        1,
        int(args.chunk_size),
    )

    print()
    print(
        f"Querying NN in chunks of {chunk_size:,}..."
    )

    for start in range(
        0,
        num_dense,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            num_dense,
        )

        query_xyz = dense_xyz[
            start:end
        ].astype(np.float64)

        distances, indices = tree.query(
            query_xyz,
            k=1,
            workers=args.workers,
        )

        nn_distance[
            start:end
        ] = distances.astype(
            np.float32
        )

        nn_index[
            start:end
        ] = indices.astype(
            np.int64
        )

        print(
            f"  {start:>10,} -> {end:>10,} "
            f"({100.0 * end / num_dense:6.2f}%)"
        )

    print()
    print("Transferring labels...")

    semantic_id = sparse_semantic[
        nn_index
    ].astype(
        np.int32
    )

    instance_id = sparse_instance[
        nn_index
    ].astype(
        np.int32
    )

    if args.max_distance is None:

        nn_valid = np.ones(
            num_dense,
            dtype=bool,
        )

        # Do not overwrite semantic_id=0 because 0 is a real Waymo
        # semantic class (TYPE_UNDEFINED). There is deliberately no
        # artificial unknown semantic value in this first diagnostic run.

    else:

        nn_valid = (
            nn_distance
            <= args.max_distance
        )

        # Keep a separate invalid sentinel so Waymo semantic ID 0 retains
        # its real meaning.
        semantic_id = semantic_id.copy()
        instance_id = instance_id.copy()

        semantic_id[
            ~nn_valid
        ] = -1

        instance_id[
            ~nn_valid
        ] = -2

    percentile_values = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        97,
        99,
        99.5,
        99.9,
        100,
    ]

    distance_percentiles = percentile_report(
        nn_distance,
        percentile_values,
    )

    print()
    print("=" * 72)
    print("NN DISTANCE STATISTICS")
    print("=" * 72)

    print(
        "Mean   :",
        float(
            np.mean(
                nn_distance
            )
        ),
        "m",
    )

    print(
        "Median :",
        float(
            np.median(
                nn_distance
            )
        ),
        "m",
    )

    for p in percentile_values:
        print(
            f"{p:>5}% : "
            f"{distance_percentiles[str(p)]:.6f} m"
        )

    if args.max_distance is not None:
        print()
        print(
            "Max-distance threshold:",
            args.max_distance,
            "m",
        )

        print(
            "Associated points:",
            f"{np.count_nonzero(nn_valid):,}",
        )

        print(
            "Unassociated points:",
            f"{np.count_nonzero(~nn_valid):,}",
        )

        print(
            "Coverage:",
            f"{100.0 * np.mean(nn_valid):.2f}%",
        )

    print()
    print("Saving dense association:")
    print(output_path)

    np.savez(
        output_path,
        xyz=dense_xyz.astype(
            np.float32
        ),
        semantic_id=semantic_id,
        instance_id=instance_id,
        nn_distance=nn_distance,
        nn_index=nn_index,
        nn_valid=nn_valid,
    )

    semantic_ids, semantic_counts = np.unique(
        semantic_id,
        return_counts=True,
    )

    stats = {
        "dense_pcd": os.path.abspath(
            args.dense_pcd
        ),
        "sparse_labeled": os.path.abspath(
            args.sparse_labeled
        ),
        "num_sparse_points": int(
            len(sparse_xyz)
        ),
        "num_dense_points": int(
            num_dense
        ),
        "max_distance": (
            None
            if args.max_distance is None
            else float(args.max_distance)
        ),
        "associated_points": int(
            np.count_nonzero(nn_valid)
        ),
        "unassociated_points": int(
            np.count_nonzero(~nn_valid)
        ),
        "coverage_fraction": float(
            np.mean(nn_valid)
        ),
        "distance_mean_m": float(
            np.mean(nn_distance)
        ),
        "distance_median_m": float(
            np.median(nn_distance)
        ),
        "distance_percentiles_m": (
            distance_percentiles
        ),
        "semantic_id_counts": {
            str(int(sid)): int(count)
            for sid, count in zip(
                semantic_ids,
                semantic_counts,
            )
        },
    }

    with open(
        stats_path,
        "w",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print()
    print("Saved stats:")
    print(stats_path)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()