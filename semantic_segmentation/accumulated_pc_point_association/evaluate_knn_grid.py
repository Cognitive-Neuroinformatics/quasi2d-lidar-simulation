#!/usr/bin/env python3

import argparse
import csv
import json
import os

import numpy as np
from scipy.spatial import cKDTree

UNKNOWN = -1


def parse_ints(text):
    vals = sorted(set(int(x.strip()) for x in text.split(',') if x.strip()))
    if not vals:
        raise ValueError('No K values provided.')
    return vals


def parse_floats(text):
    vals = sorted(set(float(x.strip()) for x in text.split(',') if x.strip()))
    if not vals:
        raise ValueError('No distance thresholds provided.')
    return vals


def find_labeled_frames(root):
    frames_dir = os.path.join(root, '01_labeled_frames')
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(frames_dir)

    frames = []
    for name in os.listdir(frames_dir):
        if name.startswith('frame_') and name.endswith('.npz'):
            try:
                frames.append(int(name[6:-4]))
            except ValueError:
                pass

    frames.sort()
    if len(frames) < 2:
        raise RuntimeError('Need at least two labeled frames.')

    return frames, frames_dir


def split_alternating(frames):
    # Split by position in the sorted labeled-frame list, not absolute frame ID.
    # This works even if labeled frames are non-contiguous.
    return frames[0::2], frames[1::2]


def load_frame(frames_dir, frame_idx):
    path = os.path.join(frames_dir, f'frame_{frame_idx:03d}.npz')
    d = np.load(path)

    required = ['xyz_world', 'semantic_id', 'instance_id', 'is_dynamic_track_point']
    for key in required:
        if key not in d.files:
            raise KeyError(f"Missing {key} in {path}; keys={d.files}")

    return {
        'xyz': d['xyz_world'].astype(np.float32),
        'semantic': d['semantic_id'].astype(np.int32),
        'instance': d['instance_id'].astype(np.int32),
        'is_dynamic': d['is_dynamic_track_point'].astype(bool),
    }


def build_reference(frames_dir, reference_frames):
    xyz_parts = []
    sem_parts = []
    inst_parts = []

    print('\nBuilding held-in STATIC reference cloud...')

    for frame_idx in reference_frames:
        d = load_frame(frames_dir, frame_idx)
        keep = ~d['is_dynamic']

        xyz_parts.append(d['xyz'][keep])
        sem_parts.append(d['semantic'][keep])
        inst_parts.append(d['instance'][keep])

        print(f'  [{frame_idx:03d}] {np.count_nonzero(keep):,} static labeled points')

    xyz = np.concatenate(xyz_parts, axis=0)
    sem = np.concatenate(sem_parts, axis=0)
    inst = np.concatenate(inst_parts, axis=0)

    return xyz, sem, inst


def vote_rows(neighbor_labels, neighbor_distances, k, threshold, min_neighbors):
    """
    Equal-weight majority vote among the first K neighbors that are <= threshold.

    Ties are broken by the closest neighbor belonging to one of the tied labels.
    """
    labs = neighbor_labels[:, :k]
    dists = neighbor_distances[:, :k]
    eligible = dists <= threshold

    counts = eligible.sum(axis=1)
    valid = counts >= min_neighbors

    pred = np.full(len(labs), UNKNOWN, dtype=np.int32)
    agreement = np.zeros(len(labs), dtype=np.float32)

    for row in np.flatnonzero(valid):
        mask = eligible[row]
        row_labs = labs[row][mask]
        row_dists = dists[row][mask]

        unique, cnt = np.unique(row_labs, return_counts=True)
        max_count = cnt.max()
        tied = unique[cnt == max_count]

        if len(tied) == 1:
            chosen = int(tied[0])
        else:
            chosen = None
            best_d = np.inf
            for label in tied:
                d = row_dists[row_labs == label].min()
                if d < best_d:
                    best_d = d
                    chosen = int(label)

        pred[row] = chosen
        agreement[row] = np.count_nonzero(row_labs == chosen) / len(row_labs)

    return pred, valid, counts.astype(np.int16), agreement


def init_state(class_ids):
    n = len(class_ids)
    return {
        'total': 0,
        'assigned': 0,
        'correct': 0,
        'confusion': np.zeros((n, n), dtype=np.int64),
        'gt_counts': np.zeros(n, dtype=np.int64),
        'nearest_distance_sum': 0.0,
        'agreement_sum': 0.0,
    }


def update_state(state, gt, pred, valid, nearest_distance, agreement, class_to_idx):
    state['total'] += len(gt)
    state['assigned'] += int(np.count_nonzero(valid))
    state['correct'] += int(np.count_nonzero(valid & (pred == gt)))

    if np.any(valid):
        state['nearest_distance_sum'] += float(nearest_distance[valid].sum())
        state['agreement_sum'] += float(agreement[valid].sum())

    ids, counts = np.unique(gt, return_counts=True)
    for cid, count in zip(ids, counts):
        state['gt_counts'][class_to_idx[int(cid)]] += int(count)

    assigned_idx = np.flatnonzero(valid)
    for i in assigned_idx:
        gi = class_to_idx[int(gt[i])]
        pi = class_to_idx.get(int(pred[i]))
        if pi is not None:
            state['confusion'][gi, pi] += 1


def compute_metrics(state, class_ids):
    total = state['total']
    assigned = state['assigned']
    correct = state['correct']

    coverage = assigned / total if total else 0.0
    acc_assigned = correct / assigned if assigned else 0.0
    effective_acc = correct / total if total else 0.0

    ious_all = []
    ious_nonzero = []

    cm = state['confusion']
    gt_counts = state['gt_counts']

    for i, cid in enumerate(class_ids):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(gt_counts[i] - tp)  # includes unassigned GT points
        denom = tp + fp + fn

        if denom == 0:
            continue

        iou = tp / denom
        ious_all.append(iou)
        if int(cid) != 0:  # report mIoU excluding Waymo UNDEFINED separately
            ious_nonzero.append(iou)

    mean_dist = state['nearest_distance_sum'] / assigned if assigned else 0.0
    mean_agreement = state['agreement_sum'] / assigned if assigned else 0.0

    return {
        'coverage': coverage,
        'assigned_accuracy': acc_assigned,
        'effective_accuracy': effective_acc,
        'miou_all': float(np.mean(ious_all)) if ious_all else 0.0,
        'miou_nonzero': float(np.mean(ious_nonzero)) if ious_nonzero else 0.0,
        'mean_nearest_distance_m': mean_dist,
        'mean_vote_agreement': mean_agreement,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--association-root', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--k-values', default='1,3,5,6,10')
    parser.add_argument('--distance-thresholds', default='0.05,0.10,0.15,0.20,0.30,0.50')
    parser.add_argument('--min-neighbors', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=-1)

    args = parser.parse_args()

    association_root = os.path.abspath(args.association_root)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(association_root, '04_knn_evaluation')
    )
    os.makedirs(output_dir, exist_ok=True)

    k_values = parse_ints(args.k_values)
    thresholds = parse_floats(args.distance_thresholds)
    k_max = max(k_values)

    labeled_frames, frames_dir = find_labeled_frames(association_root)
    reference_frames, evaluation_frames = split_alternating(labeled_frames)

    print('=' * 72)
    print('HELD-OUT SPLIT')
    print('=' * 72)
    print('All labeled frames :', len(labeled_frames))
    print('Reference frames   :', len(reference_frames))
    print('Evaluation frames  :', len(evaluation_frames))
    print('\nReference:', reference_frames)
    print('\nEvaluation:', evaluation_frames)

    ref_xyz, ref_semantic, ref_instance = build_reference(frames_dir, reference_frames)

    print('\nReference points:', f'{len(ref_xyz):,}')
    print('Building KD-tree...')
    tree = cKDTree(ref_xyz.astype(np.float64))
    print('KD-tree ready.')

    # Collect semantic class universe from held-in and held-out static points.
    class_set = set(int(x) for x in np.unique(ref_semantic))
    for frame_idx in evaluation_frames:
        d = load_frame(frames_dir, frame_idx)
        keep = ~d['is_dynamic']
        class_set.update(int(x) for x in np.unique(d['semantic'][keep]))

    class_ids = np.asarray(sorted(class_set), dtype=np.int32)
    class_to_idx = {int(c): i for i, c in enumerate(class_ids)}

    configs = [(k, t) for k in k_values for t in thresholds]
    global_state = {cfg: init_state(class_ids) for cfg in configs}
    per_frame_rows = []

    print('\n' + '=' * 72)
    print('KNN GRID EVALUATION')
    print('=' * 72)

    chunk_size = max(1, args.chunk_size)

    for n_frame, frame_idx in enumerate(evaluation_frames, start=1):
        d = load_frame(frames_dir, frame_idx)
        keep = ~d['is_dynamic']

        xyz = d['xyz'][keep]
        gt = d['semantic'][keep]

        print(f'\n[{frame_idx:03d}] static GT points: {len(xyz):,} '
              f'({n_frame}/{len(evaluation_frames)})')

        frame_state = {cfg: init_state(class_ids) for cfg in configs}

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
            nearest_distance = distances[:, 0]

            for cfg in configs:
                k, threshold = cfg

                pred, valid, vote_count, agreement = vote_rows(
                    neighbor_semantic,
                    distances,
                    k=k,
                    threshold=threshold,
                    min_neighbors=args.min_neighbors,
                )

                update_state(
                    global_state[cfg],
                    gt_chunk,
                    pred,
                    valid,
                    nearest_distance,
                    agreement,
                    class_to_idx,
                )

                update_state(
                    frame_state[cfg],
                    gt_chunk,
                    pred,
                    valid,
                    nearest_distance,
                    agreement,
                    class_to_idx,
                )

        for cfg in configs:
            k, threshold = cfg
            m = compute_metrics(frame_state[cfg], class_ids)

            per_frame_rows.append({
                'frame_idx': frame_idx,
                'k': k,
                'distance_threshold_m': threshold,
                'min_neighbors': args.min_neighbors,
                'total_points': frame_state[cfg]['total'],
                'assigned_points': frame_state[cfg]['assigned'],
                **m,
            })

    summary_rows = []

    for cfg in configs:
        k, threshold = cfg
        m = compute_metrics(global_state[cfg], class_ids)

        summary_rows.append({
            'k': k,
            'distance_threshold_m': threshold,
            'min_neighbors': args.min_neighbors,
            'total_eval_points': global_state[cfg]['total'],
            'assigned_points': global_state[cfg]['assigned'],
            **m,
        })

    # Useful ranking: effective accuracy rewards both correctness and coverage.
    summary_rows.sort(
        key=lambda r: (
            r['effective_accuracy'],
            r['miou_nonzero'],
            r['coverage'],
        ),
        reverse=True,
    )

    summary_csv = os.path.join(output_dir, 'knn_grid_summary.csv')
    frame_csv = os.path.join(output_dir, 'knn_per_frame_metrics.csv')
    summary_json = os.path.join(output_dir, 'knn_grid_summary.json')
    split_json = os.path.join(output_dir, 'split_manifest.json')

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(frame_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    with open(summary_json, 'w') as f:
        json.dump({
            'semantic_class_ids': [int(x) for x in class_ids],
            'results': summary_rows,
        }, f, indent=2)

    with open(split_json, 'w') as f:
        json.dump({
            'split': 'alternating positions in sorted labeled-frame list',
            'all_labeled_frames': labeled_frames,
            'reference_frames': reference_frames,
            'evaluation_frames': evaluation_frames,
            'k_values': k_values,
            'distance_thresholds_m': thresholds,
            'min_neighbors': args.min_neighbors,
        }, f, indent=2)

    print('\n' + '=' * 88)
    print('TOP CONFIGURATIONS')
    print('=' * 88)
    print(f"{'K':>3} {'thr(m)':>8} {'coverage':>10} {'acc(valid)':>11} "
          f"{'eff.acc':>10} {'mIoU(no0)':>11} {'agreement':>11}")
    print('-' * 88)

    for row in summary_rows[:15]:
        print(
            f"{row['k']:>3d} "
            f"{row['distance_threshold_m']:8.3f} "
            f"{100*row['coverage']:9.2f}% "
            f"{100*row['assigned_accuracy']:10.2f}% "
            f"{100*row['effective_accuracy']:9.2f}% "
            f"{100*row['miou_nonzero']:10.2f}% "
            f"{100*row['mean_vote_agreement']:10.2f}%"
        )

    print('\nSaved:')
    print(summary_csv)
    print(frame_csv)
    print(summary_json)
    print(split_json)


if __name__ == '__main__':
    main()