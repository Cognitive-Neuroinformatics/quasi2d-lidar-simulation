#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import subprocess
import sys

# -----------------------------------------------------------------------------
# CPU THREAD CONFIGURATION
# -----------------------------------------------------------------------------
# Read --cpu-workers before importing NumPy/TensorFlow so BLAS/OpenMP libraries
# see the requested thread count during initialization. 0 means all CPUs visible
# to this process. Scenes are intentionally processed sequentially; expensive
# cKDTree operations inside each scene use all configured workers.
def _early_cpu_worker_count(argv):
    requested = 0
    for i, token in enumerate(argv):
        if token == '--cpu-workers' and i + 1 < len(argv):
            requested = int(argv[i + 1])
            break
        if token.startswith('--cpu-workers='):
            requested = int(token.split('=', 1)[1])
            break
    detected = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return detected if requested <= 0 else min(requested, detected)

CPU_WORKERS = _early_cpu_worker_count(sys.argv[1:])
TF_INTEROP_WORKERS = max(1, min(4, CPU_WORKERS // 4 if CPU_WORKERS >= 4 else 1))
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = str(CPU_WORKERS)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(CPU_WORKERS)
os.environ['TF_NUM_INTEROP_THREADS'] = str(TF_INTEROP_WORKERS)
from collections import Counter, defaultdict
import numpy as np
import open3d as o3d
import tensorflow as tf
import yaml
from scipy.spatial import cKDTree
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils

TYPE_NAMES = {0: 'UNKNOWN', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'SIGN', 4: 'CYCLIST'}

NN_UNASSIGNED = -1
UNDEFINED = 0
CAR = 1
TRUCK = 2
BUS = 3
OTHER_VEHICLE = 4
MOTORCYCLIST = 5
BICYCLIST = 6
PEDESTRIAN = 7
SIGN = 8
TRAFFIC_LIGHT = 9
POLE = 10
CONSTRUCTION_CONE = 11
BICYCLE = 12
MOTORCYCLE = 13
BUILDING = 14
VEGETATION = 15
TREE_TRUNK = 16
CURB = 17
ROAD = 18
LANE_MARKER = 19
OTHER_GROUND = 20
WALKABLE = 21
SIDEWALK = 22

GROUND_CLASSES = {CURB, ROAD, LANE_MARKER, OTHER_GROUND, WALKABLE, SIDEWALK}

DYNAMIC_BOX_TYPES = {label_pb2.Label.TYPE_VEHICLE, label_pb2.Label.TYPE_PEDESTRIAN, label_pb2.Label.TYPE_CYCLIST}

BOX_COMPATIBLE_SEMANTICS = {label_pb2.Label.TYPE_VEHICLE: {CAR, TRUCK, BUS, OTHER_VEHICLE}, label_pb2.Label.TYPE_PEDESTRIAN: {PEDESTRIAN}, label_pb2.Label.TYPE_CYCLIST: {MOTORCYCLIST, BICYCLIST}}

BOX_TYPE_FALLBACK_SEMANTIC = {label_pb2.Label.TYPE_VEHICLE: CAR, label_pb2.Label.TYPE_PEDESTRIAN: PEDESTRIAN, label_pb2.Label.TYPE_CYCLIST: BICYCLIST}

FOREGROUND_SEMANTIC_CLASSES = set().union(*BOX_COMPATIBLE_SEMANTICS.values())

# Integrated static-cloud post-filter support groups. These merged IDs are used
# ONLY internally while testing geometric support. Saved semantic_id values keep
# the original Waymo classes unchanged.
FILTER_VEHICLE_SOURCE_IDS = (CAR, TRUCK, BUS, OTHER_VEHICLE)
FILTER_CYCLIST_SOURCE_IDS = (MOTORCYCLIST, BICYCLIST)
SEMANTIC_NAMES = {
    NN_UNASSIGNED: 'NN_UNASSIGNED', UNDEFINED: 'UNDEFINED', CAR: 'CAR', TRUCK: 'TRUCK', BUS: 'BUS',
    OTHER_VEHICLE: 'OTHER_VEHICLE', MOTORCYCLIST: 'MOTORCYCLIST', BICYCLIST: 'BICYCLIST',
    PEDESTRIAN: 'PEDESTRIAN', SIGN: 'SIGN', TRAFFIC_LIGHT: 'TRAFFIC_LIGHT', POLE: 'POLE',
    CONSTRUCTION_CONE: 'CONSTRUCTION_CONE', BICYCLE: 'BICYCLE', MOTORCYCLE: 'MOTORCYCLE',
    BUILDING: 'BUILDING', VEGETATION: 'VEGETATION', TREE_TRUNK: 'TREE_TRUNK', CURB: 'CURB',
    ROAD: 'ROAD', LANE_MARKER: 'LANE_MARKER', OTHER_GROUND: 'OTHER_GROUND', WALKABLE: 'WALKABLE', SIDEWALK: 'SIDEWALK'
}

class TrackReference:
    """Accumulated clean foreground/background samples for one Waymo track."""

    def __init__(self):
        self.positive_parts = []
        self.negative_parts = []
        self.semantic_votes = Counter()
        self.positive_tree = None
        self.negative_tree = None
        self.box_extensions = {}

def flatten_config(mapping):
    """Flatten human-readable YAML sections to argparse destination names."""
    flattened = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested = flatten_config(value)
            for nested_key, nested_value in nested.items():
                if nested_key in flattened:
                    raise ValueError(f'Duplicate YAML configuration key: {nested_key}')
                flattened[nested_key] = nested_value
        else:
            normalized_key = str(key).replace('-', '_')
            if normalized_key in flattened:
                raise ValueError(f'Duplicate YAML configuration key: {normalized_key}')
            flattened[normalized_key] = value
    return flattened

def case_name_from_tfrecord(tfrecord_path):
    case = os.path.basename(tfrecord_path)
    if case.endswith('.tfrecord'):
        case = case[:-len('.tfrecord')]
    return case

def read_split_entries(split_files):
    """Read ordered, unique scene entries from one or more ImageSets files."""
    entries = []
    seen = set()
    for split_file in split_files:
        split_file = os.path.abspath(split_file)
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f'Split file does not exist: {split_file}')
        with open(split_file, 'r') as f:
            for line_number, raw_line in enumerate(f, start=1):
                entry = raw_line.split('#', 1)[0].strip()
                if not entry:
                    continue
                if entry not in seen:
                    entries.append({'entry': entry, 'split_file': split_file, 'line_number': line_number})
                    seen.add(entry)
    return entries

def resolve_tfrecord_entry(entry, tfrecord_dir):
    """Resolve an absolute path, relative path, or bare Waymo case name."""
    raw_entry = os.path.expanduser(os.path.expandvars(entry))
    candidates = []
    if os.path.isabs(raw_entry):
        candidates.append(raw_entry)
    else:
        candidates.append(os.path.join(tfrecord_dir, raw_entry))
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        if not candidate.endswith('.tfrecord'):
            expanded.append(candidate + '.tfrecord')
    for candidate in expanded:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None

def child_arguments_for_scene(tfrecord_path):
    """Reuse processing options while removing parent-only batch options."""
    options_with_values = {'--split-file', '--tfrecord-dir'}
    parent_only_flags = {'--continue-on-error', '--no-continue-on-error', '--skip-existing-scenes', '--no-skip-existing-scenes'}
    original = sys.argv[1:]
    child = []
    index = 0
    while index < len(original):
        token = original[index]
        if token in options_with_values:
            index += 2
            continue
        if any((token.startswith(option + '=') for option in options_with_values)):
            index += 1
            continue
        if token in parent_only_flags:
            index += 1
            continue
        child.append(token)
        index += 1
    child.extend(['--tfrecord', tfrecord_path])
    return child

def run_batch(args):
    """Process a scene list sequentially, using the full CPU worker pool per scene.

    Sequential scene execution is deliberate: one scene can hold large point arrays and
    KD-trees in memory, and its cKDTree queries already use CPU_WORKERS threads. Running
    multiple scenes concurrently would oversubscribe a 24-thread CPU and multiply RAM use.
    """
    if args.case is not None:
        raise ValueError('--case cannot be used with --split-file because every TFRecord must keep its own case name.')
    tfrecord_dir = os.path.abspath(args.tfrecord_dir)
    if not os.path.isdir(tfrecord_dir):
        raise NotADirectoryError(f'TFRecord directory does not exist: {tfrecord_dir}')
    entries = read_split_entries(args.split_file)
    if not entries:
        raise RuntimeError('The selected split file(s) contain no scene entries.')

    output_root = os.path.abspath(args.output_root)
    manifest_dir = os.path.join(output_root, 'batch_manifests')
    log_dir = os.path.join(output_root, 'logs', 'preprocessing_batch')
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    split_names = '__'.join((os.path.splitext(os.path.basename(path))[0] for path in args.split_file))
    manifest_path = os.path.join(manifest_dir, f'{split_names}_manifest.json')

    manifest = {
        'split_files': [os.path.abspath(path) for path in args.split_file],
        'tfrecord_dir': tfrecord_dir,
        'output_root': output_root,
        'frame_selection': args.frame_selection,
        'cpu_workers_per_scene': int(CPU_WORKERS),
        'execution_mode': 'sequential_scenes_full_cpu_per_scene',
        'scenes': []
    }
    resolved_jobs = []
    for item in entries:
        tfrecord_path = resolve_tfrecord_entry(item['entry'], tfrecord_dir)
        case = case_name_from_tfrecord(tfrecord_path) if tfrecord_path is not None else case_name_from_tfrecord(item['entry'])
        job = {**item, 'case': case, 'tfrecord': tfrecord_path, 'status': 'pending'}
        manifest['scenes'].append(job)
        if tfrecord_path is None:
            job['status'] = 'missing'
            job['error'] = 'Could not resolve entry under tfrecord_dir, with or without the .tfrecord suffix.'
        else:
            resolved_jobs.append(job)

    def save_manifest():
        tmp_path = manifest_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)

    missing = [job for job in manifest['scenes'] if job['status'] == 'missing']
    save_manifest()
    if missing and (not args.continue_on_error):
        formatted = '\n'.join((f"  - {job['entry']} ({job['split_file']}:{job['line_number']})" for job in missing))
        raise FileNotFoundError('Some split entries could not be resolved:\n' + formatted)

    print('\n' + '=' * 72)
    print('LIDAR-GS WAYMO OVERNIGHT BATCH PREPROCESSING')
    print('=' * 72)
    print(f'Split files       : {len(args.split_file)}')
    print(f'Unique entries    : {len(entries)}')
    print(f'Resolved TFRecords: {len(resolved_jobs)}')
    print(f'Missing TFRecords : {len(missing)}')
    print(f'CPU workers/scene : {CPU_WORKERS}')
    print('Parallel scenes   : NO (avoids RAM/CPU oversubscription)')
    print(f'Output root       : {output_root}')
    print(f'Per-scene logs    : {log_dir}')
    print(f'Manifest          : {manifest_path}')

    child_env = os.environ.copy()
    child_env['OMP_NUM_THREADS'] = str(CPU_WORKERS)
    child_env['OPENBLAS_NUM_THREADS'] = str(CPU_WORKERS)
    child_env['MKL_NUM_THREADS'] = str(CPU_WORKERS)
    child_env['NUMEXPR_NUM_THREADS'] = str(CPU_WORKERS)
    child_env['TF_NUM_INTRAOP_THREADS'] = str(CPU_WORKERS)
    child_env['TF_NUM_INTEROP_THREADS'] = str(TF_INTEROP_WORKERS)

    for job_number, job in enumerate(resolved_jobs, start=1):
        completion_marker = os.path.join(output_root, 'temp', job['case'], 'PREPROCESSING_COMPLETE.json')
        existing_complete = os.path.isfile(completion_marker)
        if existing_complete and args.run_static_filter:
            strict_expected = os.path.join(output_root, 'recon_related', job['case'], 'static_filter', 'static_recon_labels_strict.npz')
            try:
                with open(completion_marker, 'r') as f:
                    marker_data = json.load(f)
                existing_complete = bool(marker_data.get('static_filter_completed', False)) and os.path.isfile(strict_expected)
            except Exception:
                existing_complete = False
        if args.skip_existing_scenes and existing_complete:
            job['status'] = 'skipped_existing'
            print(f"\n[{job_number}/{len(resolved_jobs)}] SKIP {job['case']} (matching completion marker exists)")
            save_manifest()
            continue

        print('\n' + '-' * 72)
        print(f"[{job_number}/{len(resolved_jobs)}] Processing {job['case']}")
        print(f"TFRecord: {job['tfrecord']}")
        print('-' * 72)
        scene_log = os.path.join(log_dir, f"{job['case']}.log")
        job['log'] = scene_log
        job['status'] = 'running'
        save_manifest()

        command = [sys.executable, '-u', os.path.abspath(__file__), *child_arguments_for_scene(job['tfrecord'])]
        with open(scene_log, 'a', buffering=1) as log_handle:
            log_handle.write('\n' + '=' * 80 + '\n')
            log_handle.write('COMMAND: ' + ' '.join(command) + '\n')
            log_handle.write('=' * 80 + '\n')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=child_env)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='')
                log_handle.write(line)
            return_code = process.wait()

        job['return_code'] = int(return_code)
        if return_code == 0:
            job['status'] = 'completed'
        else:
            job['status'] = 'failed'
            job['error'] = f'Scene subprocess exited with code {return_code}.'
        save_manifest()

        if return_code != 0 and not args.continue_on_error:
            break

    counts = Counter((job['status'] for job in manifest['scenes']))
    print('\n' + '=' * 72)
    print('BATCH COMPLETE')
    print('=' * 72)
    for status in ['completed', 'skipped_existing', 'missing', 'failed', 'pending', 'running']:
        print(f'{status:20s}: {counts.get(status, 0)}')
    print(f'Manifest:\n{manifest_path}')

    if counts.get('failed', 0) or counts.get('missing', 0):
        if not args.continue_on_error:
            raise RuntimeError('Batch stopped because at least one scene failed or was missing.')

def transform_points(points_xyz, T):
    """Transform Nx3 points with a 4x4 homogeneous transform."""
    if len(points_xyz) == 0:
        return points_xyz.copy()
    points_h = np.concatenate([points_xyz.astype(np.float64), np.ones((len(points_xyz), 1), dtype=np.float64)], axis=1)
    return (points_h @ T.T)[:, :3]

def save_pcd_xyz(points, path):
    """Save XYZ only. No voxelization, no filtering, no downsampling."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f'Failed writing {path}')

def get_top_calibration(frame):
    for calib in frame.context.laser_calibrations:
        if calib.name == open_dataset.LaserName.TOP:
            return calib
    raise RuntimeError('TOP LiDAR calibration not found.')

def load_frames(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    frames = []
    for idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
        if idx % 20 == 0:
            print(f'Loaded frame {idx}')
    print(f'Total frames: {len(frames)}')
    return frames

def extract_top_both_returns(frame):
    """
    Extract the Waymo TOP LiDAR using the current official conversion.

    - TOP LiDAR only
    - return 0 + return 1
    - TOP per-pixel motion compensation enabled
    - XYZ in vehicle frame
    - raw Waymo intensity preserved

    Output:
        pcd: Nx4 [x_vehicle, y_vehicle, z_vehicle, intensity]
        segmentation: optional Nx2 [instance_id, semantic_class]
    """
    range_images, camera_projections, segmentation_labels, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    top_name = open_dataset.LaserName.TOP
    output = []
    label_output = []
    labels_complete = True
    for ri_index in [0, 1]:
        points, _ = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_index)
        xyz = np.asarray(points[0], dtype=np.float64)
        ri_proto = range_images[top_name][ri_index]
        ri = tf.reshape(tf.convert_to_tensor(ri_proto.data), ri_proto.shape.dims).numpy()
        valid = ri[..., 0] > 0
        intensity = ri[..., 1][valid].astype(np.float64)
        if len(xyz) != len(intensity):
            raise RuntimeError(f'XYZ/intensity mismatch for return {ri_index}: {len(xyz)} vs {len(intensity)}')
        output.append(np.concatenate([xyz, intensity[:, None]], axis=1))
        available = top_name in segmentation_labels and len(segmentation_labels[top_name]) > ri_index and bool(segmentation_labels[top_name][ri_index].data)
        if available:
            label_proto = segmentation_labels[top_name][ri_index]
            label_ri = tf.reshape(tf.convert_to_tensor(label_proto.data), label_proto.shape.dims).numpy()
            aligned_labels = np.asarray(label_ri[valid, :2], dtype=np.int32)
            if len(aligned_labels) != len(xyz):
                raise RuntimeError(f'XYZ/segmentation mismatch for return {ri_index}: {len(xyz)} vs {len(aligned_labels)}')
            label_output.append(aligned_labels)
        else:
            labels_complete = False
    pcd = np.concatenate(output, axis=0)
    segmentation = None
    if labels_complete and len(label_output) == 2:
        segmentation = np.concatenate(label_output, axis=0)
    return (pcd, segmentation)

def discover_segmentation_frames(frames):
    """Discover frames containing complete TOP labels for both returns."""
    labelled = []
    for frame_idx, frame in enumerate(frames):
        _, _, segmentation_labels, _ = frame_utils.parse_range_image_and_camera_projection(frame)
        top_labels = segmentation_labels.get(open_dataset.LaserName.TOP, [])
        if len(top_labels) >= 2 and bool(top_labels[0].data) and bool(top_labels[1].data):
            labelled.append(frame_idx)
    return labelled

def box_mask(points_vehicle, box, margin=0.0):
    """Boolean mask for vehicle-frame points inside a Waymo 3D label box."""
    dx = points_vehicle[:, 0] - box.center_x
    dy = points_vehicle[:, 1] - box.center_y
    dz = points_vehicle[:, 2] - box.center_z
    c = np.cos(box.heading)
    s = np.sin(box.heading)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return (np.abs(x_local) <= box.length / 2.0 + margin) & (np.abs(y_local) <= box.width / 2.0 + margin) & (np.abs(dz) <= box.height / 2.0 + margin)

def dynamic_component_parameters(box_type, args):
    """Class-specific local growth limits for completing truncated boxes."""
    if box_type == label_pb2.Label.TYPE_VEHICLE:
        return (args.vehicle_component_margin, args.vehicle_component_link_radius)
    if box_type == label_pb2.Label.TYPE_PEDESTRIAN:
        return (0.0, args.pedestrian_component_link_radius)
    return (0.0, args.cyclist_component_link_radius)

def extension_bounds(local_points, dimensions):
    """Union of original box bounds and ALL already-accepted object points.

    These are box-local coordinates at the original heading. Each face can
    only move outward. Point filtering must happen before calling this helper.
    """
    half = np.asarray(dimensions, dtype=np.float64) / 2.0
    lower, upper = (-half.copy(), half.copy())
    if len(local_points):
        lower = np.minimum(lower, np.min(local_points, axis=0))
        upper = np.maximum(upper, np.max(local_points, axis=0))
    if not (np.all(lower <= -half) and np.all(upper >= half)):
        raise RuntimeError('Box extension failed original-box containment')
    return (lower, upper)

def interpolate_extension(samples, source_frame):
    """Interpolate outward face offsets in source-frame time, never world poses."""
    if not samples:
        return (np.zeros(6), 'no_reference', [])
    frames = sorted(samples)
    if source_frame in samples:
        return (np.asarray(samples[source_frame]).copy(), 'observed', [source_frame])
    right = int(np.searchsorted(frames, source_frame))
    if right == 0 or right == len(frames):
        nearest = frames[0] if right == 0 else frames[-1]
        return (np.asarray(samples[nearest]).copy(), 'nearest', [nearest])
    before, after = (frames[right - 1], frames[right])
    alpha = (source_frame - before) / (after - before)
    offsets = (1 - alpha) * np.asarray(samples[before]) + alpha * np.asarray(samples[after])
    return (np.maximum(offsets, 0), 'interpolated', [before, after])

def extended_track_box(label, local_points, offsets):
    if label.type != label_pb2.Label.TYPE_VEHICLE:
        b = label.box
        return ([float(b.center_x), float(b.center_y), float(b.center_z), float(b.length), float(b.width), float(b.height), float(b.heading)], np.zeros(6))
    half = np.array([label.box.length, label.box.width, label.box.height]) / 2
    lo, hi = extension_bounds(local_points, 2 * half)
    lo = np.minimum(lo, -half - offsets[:3])
    hi = np.maximum(hi, half + offsets[3:])
    centre = transform_points(((lo + hi) / 2)[None], make_T_b2l(label.box))[0]
    return ([*centre.tolist(), *(hi - lo).tolist(), float(label.box.heading)], np.r_[-half - lo, hi - half])

def connected_semantic_component(xyz_vehicle, semantic_id, label, args):
    """Return the semantic component attached to a dynamic-box seed.

    A component must contain semantic-compatible points inside the original box; a remote
    same-class cluster cannot be attached merely because it has the same class.
    """
    compatible = BOX_COMPATIBLE_SEMANTICS.get(int(label.type), set())
    if not compatible:
        return (np.empty(0, dtype=np.int64), None)
    strict_box = box_mask(xyz_vehicle, label.box, margin=0.0)
    seed = strict_box & np.isin(semantic_id, list(compatible))
    if not np.any(seed):
        return (np.empty(0, dtype=np.int64), None)
    margin, link_radius = dynamic_component_parameters(label.type, args)
    expanded_box = box_mask(xyz_vehicle, label.box, margin=margin)
    candidate = expanded_box & np.isin(semantic_id, list(compatible))
    candidate_indices = np.flatnonzero(candidate)
    candidate_xyz = xyz_vehicle[candidate_indices]
    parent = np.arange(len(candidate_indices), dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = (find(left), find(right))
        if left_root != right_root:
            parent[right_root] = left_root
    tree = cKDTree(candidate_xyz)
    for index, neighbours in enumerate(tree.query_ball_point(candidate_xyz, r=link_radius, workers=CPU_WORKERS)):
        for neighbour in neighbours:
            if neighbour > index:
                union(index, neighbour)
    local_seed = np.flatnonzero(seed[candidate_indices])
    seed_roots = {find(index) for index in local_seed}
    component_local = np.asarray([i for i in range(len(candidate_indices)) if find(i) in seed_roots], dtype=np.int64)
    component_indices = candidate_indices[component_local]
    T_l2b = np.linalg.inv(make_T_b2l(label.box))
    local = transform_points(xyz_vehicle[component_indices], T_l2b)
    lower, upper = extension_bounds(local, [label.box.length, label.box.width, label.box.height])
    center_local = 0.5 * (lower + upper)
    center_vehicle = transform_points(center_local[None, :], make_T_b2l(label.box))[0]
    corrected_box = [float(center_vehicle[0]), float(center_vehicle[1]), float(center_vehicle[2]), float(upper[0] - lower[0]), float(upper[1] - lower[1]), float(upper[2] - lower[2]), float(label.box.heading)]
    return (component_indices, corrected_box)

def make_T_b2l(box):
    """
    Construct box-local/object-local -> vehicle-frame transform.

    This matches the T_b2l convention used by the existing LiDAR-GS
    Waymo dynamic-object preprocessing.
    """
    c = np.cos(box.heading)
    s = np.sin(box.heading)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    T[:3, 3] = np.array([box.center_x, box.center_y, box.center_z], dtype=np.float64)
    return T

def collect_track_motion(frames):
    """
    Collect velocity information over the COMPLETE lifetime of every track.

    Static/dynamic classification is done once per track, not once per frame.
    If a track is classified as dynamic, it is treated as dynamic in every
    frame in which it appears, including frames where it temporarily stops.
    """
    tracks = defaultdict(list)
    for frame_idx, frame in enumerate(frames):
        for label in frame.laser_labels:
            if not label.id:
                continue
            vx = float(label.metadata.speed_x)
            vy = float(label.metadata.speed_y)
            speed_xy = float(np.hypot(vx, vy))
            tracks[label.id].append({'frame_idx': frame_idx, 'speed_xy': speed_xy, 'vx': vx, 'vy': vy, 'type': int(label.type)})
    return tracks

def classify_tracks(tracks, speed_threshold):
    """
    Whole-track classification using velocity only.

    A track is DYNAMIC if it exceeds speed_threshold in ANY frame.
    Otherwise the entire track is STATIC.

    This intentionally does NOT use displacement.
    """
    dynamic_tracks = set()
    stats = {}
    for track_id, observations in tracks.items():
        speeds = np.asarray([obs['speed_xy'] for obs in observations], dtype=np.float64)
        max_speed = float(np.max(speeds))
        mean_speed = float(np.mean(speeds))
        median_speed = float(np.median(speeds))
        moving_frame_count = int(np.count_nonzero(speeds >= speed_threshold))
        is_dynamic = max_speed >= speed_threshold
        if is_dynamic:
            dynamic_tracks.add(track_id)
        stats[track_id] = {'num_frames': int(len(observations)), 'max_speed_mps': max_speed, 'mean_speed_mps': mean_speed, 'median_speed_mps': median_speed, 'frames_at_or_above_threshold': moving_frame_count, 'speed_threshold_mps': float(speed_threshold), 'dynamic': bool(is_dynamic), 'type': TYPE_NAMES.get(observations[0]['type'], 'UNKNOWN')}
    return (dynamic_tracks, stats)

def assign_numeric_ids(dynamic_tracks, tracks):
    """
    Assign LiDAR-GS object folder IDs 1,2,3,... by first appearance.
    """
    ordering = []
    for track_id in dynamic_tracks:
        first_frame = min((obs['frame_idx'] for obs in tracks[track_id]))
        ordering.append((first_frame, track_id))
    ordering.sort()

    return {track_id: str(i + 1) for i, (_, track_id) in enumerate(ordering)}

def assign_point_instance_ids(instance_tracks, dynamic_id_map, tracks):
    """Stable point-instance IDs, keeping dynamic folder IDs unchanged."""
    point_ids = {track_id: int(object_id) for track_id, object_id in dynamic_id_map.items()}
    next_id = max(point_ids.values(), default=0) + 1
    remaining = sorted(instance_tracks - set(point_ids), key=lambda track_id: min((obs['frame_idx'] for obs in tracks[track_id])))
    for track_id in remaining:
        point_ids[track_id] = next_id
        next_id += 1

    return point_ids

def majority_nonzero(values):
    values = np.asarray(values)
    values = values[values > 0]
    if len(values) == 0:
        return (0, 0, 0.0)
    labels, counts = np.unique(values, return_counts=True)
    winner_index = int(np.argmax(counts))
    winner = int(labels[winner_index])
    count = int(counts[winner_index])
    purity = float(count / len(values))

    return (winner, count, purity)

def subsample_rows(array, maximum, rng):
    if maximum <= 0 or len(array) <= maximum:
        return array
    selected = rng.choice(len(array), size=maximum, replace=False)

    return array[selected]

def query_mean_distance(tree, query, k):
    actual_k = min(int(k), int(tree.n))
    distances, _ = tree.query(query, k=actual_k, workers=CPU_WORKERS)
    distances = np.asarray(distances, dtype=np.float64)
    if actual_k == 1:
        return distances
    
    return distances.mean(axis=1)

def semantic_to_ground_id(semantic_id):
    """Return -1 unknown, 0 non-ground, 1 ground for every point."""
    semantic_id = np.asarray(semantic_id)
    ground_id = np.full(len(semantic_id), -1, dtype=np.int8)
    defined = semantic_id > 0
    is_ground = np.isin(semantic_id, list(GROUND_CLASSES))
    ground_id[defined & ~is_ground] = 0
    ground_id[is_ground] = 1

    return ground_id

def local_ground_ring(points_vehicle, box, ring_margin):
    """Points around, but not inside, a box near its expected bottom."""
    outer = box_mask(points_vehicle, box, margin=ring_margin)
    inner = box_mask(points_vehicle, box, margin=0.15)
    expected_base = box.center_z - box.height / 2.0
    vertical = np.abs(points_vehicle[:, 2] - expected_base) < 0.45
    return points_vehicle[outer & ~inner & vertical]

def fit_ground_plane_ransac(points, iterations, rng):
    if len(points) < 3:
        return None
    best_plane = None
    best_count = 0

    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-08:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < np.cos(np.deg2rad(25.0)):
            continue
        d = -float(normal @ sample[0])
        distances = np.abs(points @ normal + d)
        count = int(np.count_nonzero(distances < 0.06))
        if count > best_count:
            best_count = count
            best_plane = np.concatenate([normal, [d]])

    return best_plane

def build_label_references(frames, labelled_frame_indices, dynamic_tracks, args, rng):
    """Build accumulated static-world and per-track object references."""
    track_references = defaultdict(TrackReference)
    static_xyz_parts = []
    static_semantic_parts = []

    print('\nBuilding semantic and instance references...')
    for source_frame_idx in labelled_frame_indices:
        frame = frames[source_frame_idx]
        pcd_vehicle, segmentation = extract_top_both_returns(frame)
        if segmentation is None:
            continue
        xyz_vehicle = pcd_vehicle[:, :3]
        gt_semantic = segmentation[:, 1]
        dynamic_candidate_union = np.zeros(len(pcd_vehicle), dtype=bool)
        for label in frame.laser_labels:
            if label.id not in dynamic_tracks or label.type not in DYNAMIC_BOX_TYPES:
                continue
            compatible = BOX_COMPATIBLE_SEMANTICS.get(int(label.type), set())
            positive, _ = connected_semantic_component(xyz_vehicle, gt_semantic, label, args)
            if len(positive) < args.association_min_points:
                continue
            dynamic_candidate_union[positive] = True
            expansion = args.box_margin
            if label.id in dynamic_tracks:
                expansion, _ = dynamic_component_parameters(label.type, args)
            candidate = box_mask(xyz_vehicle, label.box, margin=expansion)
            negative = candidate.copy()
            negative[positive] = False
            T_l2b = np.linalg.inv(make_T_b2l(label.box))
            reference = track_references[label.id]
            reference_local = transform_points(xyz_vehicle[positive], T_l2b)
            _, offsets = extended_track_box(label, reference_local, np.zeros(6))
            reference.box_extensions[int(source_frame_idx)] = offsets
            if len(positive):
                reference.positive_parts.append(transform_points(xyz_vehicle[positive], T_l2b))
                for semantic in gt_semantic[positive]:
                    if int(semantic) in compatible:
                        reference.semantic_votes[int(semantic)] += 1
            if np.any(negative):
                reference.negative_parts.append(transform_points(xyz_vehicle[negative], T_l2b))

        static_mask = ~dynamic_candidate_union & (gt_semantic != UNDEFINED)

        if np.any(static_mask):
            T_world_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
            static_xyz_parts.append(transform_points(xyz_vehicle[static_mask], T_world_vehicle).astype(np.float32))
            static_semantic_parts.append(gt_semantic[static_mask].astype(np.int16))
        print(f'  Labelled frame {source_frame_idx:03d}: static references={np.count_nonzero(static_mask):,}')

    for reference in track_references.values():
        if reference.positive_parts:
            positive = np.concatenate(reference.positive_parts, axis=0)
            positive = subsample_rows(positive, args.max_object_reference_points, rng)
            reference.positive_tree = cKDTree(positive)
        if reference.negative_parts:
            negative = np.concatenate(reference.negative_parts, axis=0)
            negative = subsample_rows(negative, args.max_object_reference_points, rng)
            reference.negative_tree = cKDTree(negative)

        reference.positive_parts.clear()
        reference.negative_parts.clear()

    if not static_xyz_parts:
        raise RuntimeError('No static semantic reference points could be constructed.')
    
    static_xyz = np.concatenate(static_xyz_parts, axis=0)
    static_semantic = np.concatenate(static_semantic_parts, axis=0)
    if args.max_static_reference_points > 0 and len(static_xyz) > args.max_static_reference_points:
        selected = rng.choice(len(static_xyz), size=args.max_static_reference_points, replace=False)
        static_xyz = static_xyz[selected]
        static_semantic = static_semantic[selected]

    return (track_references, static_xyz, static_semantic, cKDTree(static_xyz))

def propagate_static_semantics(query_world, reference_tree, reference_semantic, k, max_distance, min_vote_fraction, return_debug=False):
    if len(query_world) == 0:
        result = np.empty(0, dtype=np.int16)
        confidence = np.empty(0, dtype=np.float32)
        if not return_debug:
            return (result, confidence)
        return (result, confidence, {'distances': np.empty((0, 0), dtype=np.float32), 'indices': np.empty((0, 0), dtype=np.int64), 'neighbour_semantic': np.empty((0, 0), dtype=np.int16), 'within': np.empty((0, 0), dtype=bool), 'valid_count': np.empty(0, dtype=np.uint8), 'winning_votes': np.empty(0, dtype=np.uint8), 'raw_winner': np.empty(0, dtype=np.int16), 'vote_fraction': np.empty(0, dtype=np.float32), 'accepted': np.empty(0, dtype=bool), 'nearest_distance': np.empty(0, dtype=np.float32), 'mean_valid_distance': np.empty(0, dtype=np.float32)})

    actual_k = min(int(k), int(reference_tree.n))
    distances, indices = reference_tree.query(query_world, k=actual_k, workers=CPU_WORKERS)
    if actual_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    neighbour_labels = reference_semantic[indices]
    within = distances <= max_distance
    class_counts = np.zeros((len(query_world), 23), dtype=np.uint8)
    rows = np.arange(len(query_world))

    for neighbour in range(actual_k):
        mask = within[:, neighbour]
        valid_rows = rows[mask]
        valid_classes = neighbour_labels[mask, neighbour].astype(np.int64)
        np.add.at(class_counts, (valid_rows, valid_classes), 1)

    raw_winner = np.argmax(class_counts, axis=1).astype(np.int16)
    winning_votes = np.max(class_counts, axis=1)
    valid_count = np.sum(within, axis=1)
    vote_fraction = np.divide(winning_votes, valid_count, out=np.zeros(len(query_world), dtype=np.float32), where=valid_count > 0)
    accepted = (valid_count > 0) & (vote_fraction >= min_vote_fraction)
    result = raw_winner.copy()
    result[~accepted] = NN_UNASSIGNED
    confidence = vote_fraction.copy()
    confidence[~accepted] = 0.0

    if not return_debug:
        return (result, confidence)

    valid_distances = np.where(within, distances, np.nan)
    nearest_distance = np.min(distances, axis=1).astype(np.float32)
    mean_valid_distance = np.full(len(query_world), np.nan, dtype=np.float32)
    has_valid = valid_count > 0
    mean_valid_distance[has_valid] = np.nanmean(valid_distances[has_valid], axis=1).astype(np.float32)
    debug = {'distances': distances.astype(np.float32), 'indices': indices.astype(np.int64), 'neighbour_semantic': neighbour_labels.astype(np.int16), 'within': within, 'valid_count': valid_count.astype(np.uint8), 'winning_votes': winning_votes.astype(np.uint8), 'raw_winner': raw_winner, 'vote_fraction': vote_fraction.astype(np.float32), 'accepted': accepted, 'nearest_distance': nearest_distance, 'mean_valid_distance': mean_valid_distance}

    return (result, confidence, debug)

def save_static_propagation_debug(output_dir, source_frame_idx, output_frame_idx, pcd_vehicle, query_indices, query_world, propagated_semantic, propagated_confidence, debug, reference_tree, reference_semantic, max_distance, min_vote_fraction):
    os.makedirs(output_dir, exist_ok=True)
    stem = f'source_{source_frame_idx:03d}_output_{output_frame_idx:03d}'
    np.savez(os.path.join(output_dir, stem + '_propagation_debug.npz'), point_index=query_indices.astype(np.int32), xyz_vehicle=pcd_vehicle[query_indices, :3].astype(np.float32), xyz_world=query_world.astype(np.float32), intensity=pcd_vehicle[query_indices, 3].astype(np.float64), propagated_semantic=propagated_semantic.astype(np.int16), label_confidence=propagated_confidence.astype(np.float32), neighbour_distances_m=debug['distances'], neighbour_semantic=debug['neighbour_semantic'], neighbour_reference_index=debug['indices'], neighbour_reference_xyz_world=np.asarray(reference_tree.data)[debug['indices']].astype(np.float32), neighbour_within_radius=debug['within'], valid_neighbor_count=debug['valid_count'], winning_votes=debug['winning_votes'], raw_winner=debug['raw_winner'], vote_fraction=debug['vote_fraction'], accepted=debug['accepted'], nearest_distance_m=debug['nearest_distance'], mean_valid_distance_m=debug['mean_valid_distance'])
    assigned = propagated_semantic > 0
    counts = {int(k): int(v) for k, v in zip(*np.unique(propagated_semantic, return_counts=True))}
    valid_hist = {str(i): int(np.count_nonzero(debug['valid_count'] == i)) for i in range(debug['distances'].shape[1] + 1)}
    one_neighbor = assigned & (debug['valid_count'] == 1)
    summary = {'source_frame_index': int(source_frame_idx), 'output_frame_index': int(output_frame_idx), 'query_points': int(len(query_indices)), 'assigned_points': int(np.count_nonzero(assigned)), 'nn_unassigned': int(np.count_nonzero(propagated_semantic == NN_UNASSIGNED)), 'semantic_counts': counts, 'valid_neighbor_count_histogram': valid_hist, 'accepted_with_one_valid_neighbor': int(np.count_nonzero(one_neighbor)), 'accepted_with_one_valid_neighbor_fraction': float(np.count_nonzero(one_neighbor) / max(np.count_nonzero(assigned), 1)), 'nearest_distance_m': {'median': float(np.median(debug['nearest_distance'])) if len(query_indices) else None, 'p95': float(np.percentile(debug['nearest_distance'], 95)) if len(query_indices) else None, 'max': float(np.max(debug['nearest_distance'])) if len(query_indices) else None}, 'parameters': {'k': int(debug['distances'].shape[1]), 'max_distance_m': float(max_distance), 'min_vote_fraction': float(min_vote_fraction), 'reference_points': int(len(reference_semantic))}}
    sidewalk = propagated_semantic == SIDEWALK

    if np.any(sidewalk):
        summary['sidewalk'] = {'predicted_points': int(np.count_nonzero(sidewalk)), 'one_valid_neighbor': int(np.count_nonzero(sidewalk & (debug['valid_count'] == 1))), 'median_nearest_distance_m': float(np.median(debug['nearest_distance'][sidewalk])), 'p95_nearest_distance_m': float(np.percentile(debug['nearest_distance'][sidewalk], 95)), 'median_vote_fraction': float(np.median(debug['vote_fraction'][sidewalk]))}
    with open(os.path.join(output_dir, stem + '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n  STATIC PROPAGATION DEBUG')
    print(f'  source/output frame     : {source_frame_idx}/{output_frame_idx}')
    print(f'  query background points : {len(query_indices):,}')
    print(f'  assigned                : {np.count_nonzero(assigned):,}')
    print(f'  NN-unassigned           : {np.count_nonzero(propagated_semantic == NN_UNASSIGNED):,}')
    print(f'  valid-neighbour counts  : {valid_hist}')
    print(f'  accepted with 1 neighbour: {np.count_nonzero(one_neighbor):,}')

    if np.any(sidewalk):
        print(f'  SIDEWALK predictions    : {np.count_nonzero(sidewalk):,}')
        print(f"  SIDEWALK one-neighbour  : {np.count_nonzero(sidewalk & (debug['valid_count'] == 1)):,}")
        print(f"  SIDEWALK nearest d p50/p95: {np.median(debug['nearest_distance'][sidewalk]):.3f}/{np.percentile(debug['nearest_distance'][sidewalk], 95):.3f} m")

    print(f"  debug NPZ               : {os.path.join(output_dir, stem + '_propagation_debug.npz')}")
    print(f"  summary JSON            : {os.path.join(output_dir, stem + '_summary.json')}")

def track_semantic_id(track_id, box_type, track_references):
    reference = track_references.get(track_id)
    if reference is not None and reference.semantic_votes:
        return int(reference.semantic_votes.most_common(1)[0][0])
    return int(BOX_TYPE_FALLBACK_SEMANTIC.get(int(box_type), UNDEFINED))

def classify_dynamic_candidates(xyz_vehicle, candidate_indices, label, reference, args, rng):
    """Filter box candidates using object references and local ground."""
    if len(candidate_indices) == 0:
        return (candidate_indices, np.empty(0, dtype=np.float32))
    candidate_xyz = xyz_vehicle[candidate_indices]
    T_l2b = np.linalg.inv(make_T_b2l(label.box))
    candidate_local = transform_points(candidate_xyz, T_l2b)
    positive_distance = np.full(len(candidate_indices), np.inf, dtype=np.float64)
    negative_distance = np.full(len(candidate_indices), np.inf, dtype=np.float64)

    if reference is not None and reference.positive_tree is not None:
        positive_distance = query_mean_distance(reference.positive_tree, candidate_local, args.object_k)

    if reference is not None and reference.negative_tree is not None:
        negative_distance = query_mean_distance(reference.negative_tree, candidate_local, args.object_k)

    has_positive = np.isfinite(positive_distance).any()

    if has_positive:
        accepted = positive_distance <= args.object_max_distance
        if np.isfinite(negative_distance).any():
            accepted &= positive_distance + args.object_distance_margin < negative_distance
    else:
        accepted = np.ones(len(candidate_indices), dtype=bool)

    ring = local_ground_ring(xyz_vehicle, label.box, args.ground_ring_margin)
    plane = None

    if len(ring) >= args.ground_min_ring_points:
        plane = fit_ground_plane_ransac(ring, args.ground_ransac_iterations, rng)
    if plane is None:
        near_ground = np.zeros(len(candidate_indices), dtype=bool)
    else:
        near_ground = np.abs(candidate_xyz @ plane[:3] + plane[3]) <= args.ground_plane_tolerance

    strong_object = positive_distance <= args.strong_object_distance
    accepted &= ~(near_ground & ~strong_object)

    if has_positive:
        confidence = (args.object_max_distance - positive_distance).astype(np.float32)
        finite_negative = np.isfinite(negative_distance)
        confidence[finite_negative] += np.maximum(0.0, negative_distance[finite_negative] - positive_distance[finite_negative]).astype(np.float32)
    else:
        confidence = np.full(len(candidate_indices), 0.001, dtype=np.float32)

    confidence[near_ground] -= 0.5

    return (candidate_indices[accepted], confidence[accepted])

def choose_output_frame_indices(num_frames, labelled_frame_indices, selection):
    if selection == 'segmentation_bounds':
        first = int(labelled_frame_indices[0])
        last = int(labelled_frame_indices[-1])
        return list(range(first, last + 1))
    if selection == 'all':
        return list(range(num_frames))
    
    raise ValueError(f'Unknown frame selection: {selection}')

def build_meta_frame(frame, frame_idx, case, point_instance_id_map, dynamic_tracks, dynamic_id_map, track_references, top_extrinsic):
    """Build a frame record with complete Stage-A box/track metadata."""
    boxes = []
    names = []
    velocities = []
    tokens = []
    semantic_ids = []
    instance_ids = []
    is_dynamic = []
    lidargs_object_ids = []
    waymo_types = []
    num_lidar_points = []
    box_poses_vehicle = []
    box_poses_world = []
    box_poses_top_lidar = []
    T_world_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    T_vehicle_top = np.asarray(top_extrinsic, dtype=np.float64).reshape(4, 4)
    T_top_vehicle = np.linalg.inv(T_vehicle_top)

    for label in frame.laser_labels:
        box = label.box
        boxes.append([box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading])
        names.append(TYPE_NAMES.get(int(label.type), 'UNKNOWN'))
        velocities.append([float(label.metadata.speed_x), float(label.metadata.speed_y)])
        tokens.append(label.id)
        semantic_ids.append(track_semantic_id(label.id, int(label.type), track_references))
        instance_ids.append(int(point_instance_id_map.get(label.id, 0)))
        is_dynamic.append(bool(label.id in dynamic_tracks))
        lidargs_object_ids.append(int(dynamic_id_map[label.id]) if label.id in dynamic_id_map else -1)
        waymo_types.append(int(label.type))
        num_lidar_points.append(int(getattr(label, 'num_lidar_points_in_box', 0)))
        T_vehicle_box = make_T_b2l(box).astype(np.float64)
        box_poses_vehicle.append(T_vehicle_box)
        box_poses_world.append(T_world_vehicle @ T_vehicle_box)
        box_poses_top_lidar.append(T_top_vehicle @ T_vehicle_box)

    return {'path': {'pcd': f'pcds/{case}/{frame_idx:03d}.npz'}, 'lidar2world': np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4), 'log_time_stamp': int(frame.timestamp_micros), 'obj_label': {'gt_boxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 7), 'gt_names': np.asarray(names), 'gt_boxes_velocity': np.asarray(velocities, dtype=np.float32).reshape(-1, 2), 'gt_boxes_token': np.asarray(tokens), 'gt_box_semantic_ids': np.asarray(semantic_ids, dtype=np.int16), 'gt_box_instance_ids': np.asarray(instance_ids, dtype=np.int32), 'gt_box_is_dynamic': np.asarray(is_dynamic, dtype=bool), 'gt_box_lidargs_object_ids': np.asarray(lidargs_object_ids, dtype=np.int32), 'gt_box_waymo_types': np.asarray(waymo_types, dtype=np.int16), 'gt_box_num_lidar_points': np.asarray(num_lidar_points, dtype=np.int32), 'gt_box_pose_vehicle': np.asarray(box_poses_vehicle, dtype=np.float64).reshape(-1, 4, 4), 'gt_box_pose_world': np.asarray(box_poses_world, dtype=np.float64).reshape(-1, 4, 4), 'gt_box_pose_top_lidar': np.asarray(box_poses_top_lidar, dtype=np.float64).reshape(-1, 4, 4)}}

def semantic_name(value):
    return SEMANTIC_NAMES.get(int(value), f'UNKNOWN_{int(value)}')


def static_filter_semantic_groups(semantic_id):
    """Return semantic IDs used only for static-filter support tests.

    Vehicle subtypes 1/2/3/4 share one support group and Waymo cyclist rider
    classes 5/6 share one support group. The original semantic IDs are preserved
    in every saved point cloud.
    """
    grouped = np.asarray(semantic_id, dtype=np.int16).copy()
    grouped[np.isin(grouped, FILTER_VEHICLE_SOURCE_IDS)] = CAR
    grouped[np.isin(grouped, FILTER_CYCLIST_SOURCE_IDS)] = MOTORCYCLIST
    return grouped


def query_other_frame_distance_filter(xyz, frames, tree, sample_indices, k):
    query_xyz = xyz[sample_indices]
    query_frames = frames[sample_indices]
    actual_k = min(int(k), int(tree.n))
    distances, neighbours = tree.query(query_xyz, k=actual_k, workers=CPU_WORKERS)
    if actual_k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    result = np.full(len(sample_indices), np.inf, dtype=np.float64)
    for rank in range(actual_k):
        neighbour_idx = neighbours[:, rank]
        different_frame = frames[neighbour_idx] != query_frames
        choose = ~np.isfinite(result) & different_frame
        result[choose] = distances[choose, rank]
    return result


def estimate_static_filter_threshold(class_xyz, class_frames, rng, args):
    if len(class_xyz) < 2:
        return args.static_filter_max_class_threshold, {'status': 'too_few_reference_points', 'reference_points': int(len(class_xyz))}
    tree = cKDTree(class_xyz)
    if len(class_xyz) > args.static_filter_threshold_sample_max:
        selected = rng.choice(len(class_xyz), size=args.static_filter_threshold_sample_max, replace=False)
    else:
        selected = np.arange(len(class_xyz), dtype=np.int64)
    distances = query_other_frame_distance_filter(class_xyz, class_frames, tree, selected, args.static_filter_cross_frame_k)
    valid = np.isfinite(distances)
    if np.count_nonzero(valid) < 100:
        return args.static_filter_max_class_threshold, {
            'status': 'insufficient_cross_frame_support', 'reference_points': int(len(class_xyz)),
            'valid_cross_frame_samples': int(np.count_nonzero(valid))
        }
    d = distances[valid]
    raw = float(np.percentile(d, args.static_filter_threshold_percentile))
    threshold = float(np.clip(raw * args.static_filter_threshold_scale, args.static_filter_min_class_threshold, args.static_filter_max_class_threshold))
    return threshold, {
        'status': 'estimated', 'reference_points': int(len(class_xyz)), 'sampled_points': int(len(selected)),
        'valid_cross_frame_samples': int(len(d)), 'median_cross_frame_distance_m': float(np.median(d)),
        'p95_cross_frame_distance_m': float(np.percentile(d, 95)), 'p99_cross_frame_distance_m': float(np.percentile(d, 99)),
        'raw_percentile_distance_m': raw, 'final_threshold_m': threshold
    }


def run_integrated_static_filter(static_npz_path, output_dir, output_source_indices, labelled_frame_indices, args):
    """Create a strict static cloud while leaving the raw accumulation untouched.

    The filter reproduces the geometric-support / vertical-height post-processing
    logic previously run as a separate script:
      1. remove NN-unassigned (-1) and Waymo UNDEFINED (0),
      2. test propagated points against native same-class geometry,
      3. remove clear competing-class conflicts / unsupported propagated points,
      4. for ground classes, remove propagated points whose local Z differs from
         trusted same-class ground by more than static_filter_ground_max_dz.

    Native Waymo-labelled points are trusted by default and are not marked
    suspicious unless --static-filter-save-native-suspicious is enabled.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    print('\n' + '=' * 88)
    print('INTEGRATED STATIC CLOUD FILTER')
    print('=' * 88)
    print(f'Raw input (preserved): {static_npz_path}')
    print(f'Filtered outputs     : {output_dir}')

    with np.load(static_npz_path, allow_pickle=False) as d:
        xyz = d['xyz'].astype(np.float32)
        intensity = d['intensity']
        original_semantic_id = d['semantic_id'].astype(np.int16)
        ground_id = d['ground_id'].astype(np.int8)
        instance_id = d['instance_id'].astype(np.int32)
        label_confidence = d['label_confidence'].astype(np.float32)
        observation_frame = d['observation_frame_index'].astype(np.int32)

    n = len(xyz)
    semantic_support_id = static_filter_semantic_groups(original_semantic_id)
    output_source_indices = np.asarray(output_source_indices, dtype=np.int32)
    if len(observation_frame) and (observation_frame.min() < 0 or observation_frame.max() >= len(output_source_indices)):
        raise RuntimeError('Static observation_frame_index is outside output_source_indices.')
    source_frame = output_source_indices[observation_frame]
    native = np.isin(source_frame, np.asarray(sorted(set(int(x) for x in labelled_frame_indices)), dtype=np.int32))
    propagated = ~native

    trusted = native & (semantic_support_id > 0)
    trusted_indices = np.flatnonzero(trusted)
    trusted_xyz = xyz[trusted_indices].astype(np.float64)
    trusted_semantic = semantic_support_id[trusted_indices]
    trusted_source_frame = source_frame[trusted_indices]
    if len(trusted_xyz) == 0:
        raise RuntimeError('Integrated static filter found no trusted native semantic points.')

    print(f'Input points                 : {n:,}')
    print(f'Native-labelled observations : {np.count_nonzero(native):,}')
    print(f'Propagated observations      : {np.count_nonzero(propagated):,}')
    print(f'Trusted native references    : {len(trusted_xyz):,}')

    classes = sorted(int(v) for v in np.unique(semantic_support_id[semantic_support_id > 0]))
    class_thresholds = {}
    threshold_diagnostics = {}
    print('\nEstimating class-specific cross-frame support thresholds...')
    for semantic in classes:
        class_mask = trusted_semantic == semantic
        threshold, diagnostics = estimate_static_filter_threshold(trusted_xyz[class_mask], trusted_source_frame[class_mask], rng, args)
        class_thresholds[semantic] = threshold
        threshold_diagnostics[str(semantic)] = {'semantic_name': semantic_name(semantic), **diagnostics, 'threshold_m': float(threshold)}
        print(f"  {semantic:2d} {semantic_name(semantic):22s} refs={np.count_nonzero(class_mask):9,d} threshold={threshold:6.3f} m {diagnostics['status']}")

    valid_query = semantic_support_id > 0
    if not args.static_filter_save_native_suspicious:
        valid_query &= propagated
    query_indices = np.flatnonzero(valid_query)
    query_xyz = xyz[query_indices].astype(np.float64)
    query_semantic = semantic_support_id[query_indices]
    print(f'Points tested geometrically  : {len(query_indices):,}')

    global_tree = cKDTree(trusted_xyz)
    nearest_any_distance, nearest_any_local = global_tree.query(query_xyz, k=1, workers=CPU_WORKERS)
    nearest_any_semantic = trusted_semantic[nearest_any_local]
    same_distance = np.full(len(query_indices), np.inf, dtype=np.float64)

    for semantic in classes:
        qmask = query_semantic == semantic
        if not np.any(qmask):
            continue
        rmask = trusted_semantic == semantic
        if not np.any(rmask):
            continue
        class_tree = cKDTree(trusted_xyz[rmask])
        same_distance[qmask], _ = class_tree.query(query_xyz[qmask], k=1, workers=CPU_WORKERS)
        del class_tree

    class_threshold_array = np.asarray([class_thresholds.get(int(s), args.static_filter_max_class_threshold) for s in query_semantic], dtype=np.float64)
    unsupported = (same_distance > class_threshold_array) | ~np.isfinite(same_distance)
    competing_class = nearest_any_semantic != query_semantic
    competing_close = nearest_any_distance <= args.static_filter_conflict_max_distance
    competing_clearly_better = nearest_any_distance + args.static_filter_conflict_margin < same_distance
    conflict = competing_class & competing_close & competing_clearly_better

    print('\nGROUND XY / HEIGHT CONSISTENCY')
    ground_height_bad = np.zeros(len(query_indices), dtype=bool)
    ground_no_local_support = np.zeros(len(query_indices), dtype=bool)
    ground_local_z = np.full(len(query_indices), np.nan, dtype=np.float64)
    ground_local_dz = np.full(len(query_indices), np.nan, dtype=np.float64)
    ground_local_valid_count = np.zeros(len(query_indices), dtype=np.int16)

    for semantic in sorted(GROUND_CLASSES):
        qmask = query_semantic == semantic
        if not np.any(qmask):
            continue
        rmask = trusted_semantic == semantic
        reference_xyz = trusted_xyz[rmask]
        qidx = np.flatnonzero(qmask)
        qxyz = query_xyz[qmask]
        print(f'  {semantic:2d} {semantic_name(semantic):22s} queries={len(qidx):9,d} refs={len(reference_xyz):9,d}')
        if len(reference_xyz) == 0:
            ground_no_local_support[qidx] = True
            continue
        xy_tree = cKDTree(reference_xyz[:, :2])
        actual_k = min(args.static_filter_ground_k, len(reference_xyz))
        dxy, nn = xy_tree.query(qxyz[:, :2], k=actual_k, workers=CPU_WORKERS)
        if actual_k == 1:
            dxy = dxy[:, None]
            nn = nn[:, None]
        valid = dxy <= args.static_filter_ground_xy_radius
        valid_count = np.sum(valid, axis=1)
        ground_local_valid_count[qidx] = valid_count.astype(np.int16)
        neighbour_z = reference_xyz[nn, 2]
        masked_z = np.where(valid, neighbour_z, np.nan)
        with np.errstate(all='ignore'):
            local_z = np.nanmedian(masked_z, axis=1)
        dz = qxyz[:, 2] - local_z
        has_support = valid_count >= args.static_filter_ground_min_neighbours
        no_support = ~has_support
        bad_height = has_support & np.isfinite(dz) & (np.abs(dz) > args.static_filter_ground_max_dz)
        ground_local_z[qidx] = local_z
        ground_local_dz[qidx] = dz
        ground_height_bad[qidx] = bad_height
        ground_no_local_support[qidx] = no_support
        valid_dz = np.abs(dz[has_support & np.isfinite(dz)])
        print(f'       no local support={np.count_nonzero(no_support):,}, |dz|>{args.static_filter_ground_max_dz:.2f} m={np.count_nonzero(bad_height):,}')
        if len(valid_dz):
            print(f'       |dz| median/p95/p99={np.median(valid_dz):.4f}/{np.percentile(valid_dz,95):.4f}/{np.percentile(valid_dz,99):.4f} m')
        del xy_tree

    if args.static_filter_mode == 'conflict':
        suspicious_query = conflict | ground_height_bad
    elif args.static_filter_mode == 'unsupported':
        suspicious_query = unsupported | ground_height_bad
    else:
        suspicious_query = conflict | unsupported | ground_height_bad

    suspicious = np.zeros(n, dtype=bool)
    suspicious[query_indices] = suspicious_query
    if not args.static_filter_save_native_suspicious:
        suspicious[native] = False

    full_same_distance = np.full(n, np.nan, dtype=np.float32)
    full_nearest_any_distance = np.full(n, np.nan, dtype=np.float32)
    full_nearest_any_semantic = np.full(n, -1, dtype=np.int16)
    full_conflict = np.zeros(n, dtype=bool)
    full_unsupported = np.zeros(n, dtype=bool)
    full_ground_height_bad = np.zeros(n, dtype=bool)
    full_ground_no_local_support = np.zeros(n, dtype=bool)
    full_ground_local_dz = np.full(n, np.nan, dtype=np.float32)
    full_ground_local_z = np.full(n, np.nan, dtype=np.float32)
    full_ground_valid_count = np.zeros(n, dtype=np.int16)

    full_same_distance[query_indices] = same_distance.astype(np.float32)
    full_nearest_any_distance[query_indices] = nearest_any_distance.astype(np.float32)
    full_nearest_any_semantic[query_indices] = nearest_any_semantic.astype(np.int16)
    full_conflict[query_indices] = conflict
    full_unsupported[query_indices] = unsupported
    full_ground_height_bad[query_indices] = ground_height_bad
    full_ground_no_local_support[query_indices] = ground_no_local_support
    full_ground_local_dz[query_indices] = ground_local_dz.astype(np.float32)
    full_ground_local_z[query_indices] = ground_local_z.astype(np.float32)
    full_ground_valid_count[query_indices] = ground_local_valid_count

    invalid_semantic = original_semantic_id <= 0
    remove = invalid_semantic | suspicious
    keep = ~remove

    print('\n' + '=' * 88)
    print('STATIC FILTER SUMMARY')
    print('=' * 88)
    print(f'Input points               : {n:,}')
    print(f'Remove NN-unassigned       : {np.count_nonzero(original_semantic_id == NN_UNASSIGNED):,}')
    print(f'Remove UNDEFINED           : {np.count_nonzero(original_semantic_id == UNDEFINED):,}')
    print(f'Suspicious geometry        : {np.count_nonzero(suspicious):,}')
    print(f'  competing-class conflict : {np.count_nonzero(full_conflict):,}')
    print(f'  same-class unsupported   : {np.count_nonzero(full_unsupported):,}')
    print(f'  bad ground height        : {np.count_nonzero(full_ground_height_bad):,}')
    print(f'  ground no local support  : {np.count_nonzero(full_ground_no_local_support):,} [not removed by this alone]')
    print(f'Output strict points       : {np.count_nonzero(keep):,}')
    print(f'Retained fraction          : {100.0*np.mean(keep):.3f}%')

    suspicious_path = os.path.join(output_dir, 'all_suspicious_points.npz')
    s = suspicious
    np.savez(suspicious_path,
        xyz=xyz[s], intensity=intensity[s], semantic_id=original_semantic_id[s], filter_semantic_id=semantic_support_id[s],
        ground_id=ground_id[s], instance_id=instance_id[s], label_confidence=label_confidence[s],
        observation_frame_index=observation_frame[s], source_frame_index=source_frame[s], same_class_distance_m=full_same_distance[s],
        nearest_any_distance_m=full_nearest_any_distance[s], nearest_any_semantic_id=full_nearest_any_semantic[s],
        competing_class_conflict=full_conflict[s], same_class_unsupported=full_unsupported[s], ground_height_bad=full_ground_height_bad[s],
        ground_no_local_support=full_ground_no_local_support[s], ground_local_z=full_ground_local_z[s],
        ground_local_dz_m=full_ground_local_dz[s], ground_local_valid_count=full_ground_valid_count[s])

    strict_npz_path = os.path.join(output_dir, 'static_recon_labels_strict.npz')
    strict_arrays = {
        'xyz': xyz[keep], 'intensity': intensity[keep], 'semantic_id': original_semantic_id[keep], 'ground_id': ground_id[keep],
        'instance_id': instance_id[keep], 'label_confidence': label_confidence[keep],
        'observation_frame_index': observation_frame[keep], 'coordinate_frame': np.asarray('world')
    }
    if args.save_combined_data_labeled:
        strict_arrays['data_labeled'] = np.column_stack([
            xyz[keep], intensity[keep], original_semantic_id[keep], ground_id[keep], instance_id[keep]
        ]).astype(np.float64)
        strict_arrays['data_labeled_columns'] = np.asarray(['x_world','y_world','z_world','intensity','semantic_id','ground_id','instance_id'])
    np.savez(strict_npz_path, **strict_arrays)

    strict_pcd_path = os.path.join(output_dir, 'static_recon_voxels_strict.pcd')
    save_pcd_xyz(xyz[keep], strict_pcd_path)

    mask_path = os.path.join(output_dir, 'static_filter_masks.npz')
    np.savez(mask_path,
        keep=keep, remove=remove, invalid_semantic=invalid_semantic, suspicious=suspicious,
        competing_class_conflict=full_conflict, same_class_unsupported=full_unsupported,
        ground_height_bad=full_ground_height_bad, ground_no_local_support=full_ground_no_local_support,
        ground_local_dz_m=full_ground_local_dz, ground_local_valid_count=full_ground_valid_count,
        native=native, propagated=propagated, source_frame_index=source_frame)

    class_report = {}
    for semantic in classes:
        cmask = semantic_support_id == semantic
        propagated_count = int(np.count_nonzero(cmask & propagated))
        suspicious_count = int(np.count_nonzero(cmask & suspicious))
        class_report[str(semantic)] = {
            'name': semantic_name(semantic), 'total': int(np.count_nonzero(cmask)), 'native': int(np.count_nonzero(cmask & native)),
            'propagated': propagated_count, 'suspicious': suspicious_count,
            'suspicious_fraction_of_propagated': float(suspicious_count / propagated_count) if propagated_count else 0.0,
            'support_threshold_m': float(class_thresholds[semantic]),
            'bad_ground_height': int(np.count_nonzero(cmask & full_ground_height_bad)),
            'ground_no_local_support': int(np.count_nonzero(cmask & full_ground_no_local_support))
        }

    report = {
        'input_static_raw_preserved': os.path.abspath(static_npz_path), 'filter_mode': args.static_filter_mode,
        'semantic_support_grouping': {
            'note': 'Grouping is used only for geometric support tests. Saved strict semantic_id preserves original Waymo IDs.',
            'vehicle_group': list(FILTER_VEHICLE_SOURCE_IDS), 'cyclist_group': list(FILTER_CYCLIST_SOURCE_IDS)
        },
        'parameters': {
            'threshold_percentile': args.static_filter_threshold_percentile, 'threshold_scale': args.static_filter_threshold_scale,
            'minimum_class_threshold_m': args.static_filter_min_class_threshold, 'maximum_class_threshold_m': args.static_filter_max_class_threshold,
            'conflict_max_distance_m': args.static_filter_conflict_max_distance, 'conflict_margin_m': args.static_filter_conflict_margin,
            'cross_frame_k': args.static_filter_cross_frame_k, 'ground_xy_radius_m': args.static_filter_ground_xy_radius,
            'ground_max_dz_m': args.static_filter_ground_max_dz, 'ground_k': args.static_filter_ground_k,
            'ground_min_neighbours': args.static_filter_ground_min_neighbours, 'cpu_workers': int(CPU_WORKERS)
        },
        'counts': {
            'input': int(n), 'nn_unassigned': int(np.count_nonzero(original_semantic_id == NN_UNASSIGNED)),
            'undefined': int(np.count_nonzero(original_semantic_id == UNDEFINED)), 'native': int(np.count_nonzero(native)),
            'propagated': int(np.count_nonzero(propagated)), 'suspicious': int(np.count_nonzero(suspicious)),
            'competing_class_conflict': int(np.count_nonzero(full_conflict)), 'same_class_unsupported': int(np.count_nonzero(full_unsupported)),
            'bad_ground_height': int(np.count_nonzero(full_ground_height_bad)),
            'ground_no_local_support': int(np.count_nonzero(full_ground_no_local_support)),
            'removed_total': int(np.count_nonzero(remove)), 'retained': int(np.count_nonzero(keep))
        },
        'class_thresholds': threshold_diagnostics, 'per_class': class_report,
        'outputs': {'strict_npz': strict_npz_path, 'strict_pcd': strict_pcd_path, 'masks': mask_path, 'suspicious': suspicious_path}
    }
    report_path = os.path.join(output_dir, 'static_filter_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Raw cloud kept unchanged    : {static_npz_path}')
    print(f'Strict labelled cloud       : {strict_npz_path}')
    print(f'Strict XYZ cloud            : {strict_pcd_path}')
    print(f'Filter masks                : {mask_path}')
    print(f'Filter report               : {report_path}')
    return report


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=None, help='YAML configuration file. Explicit command-line options override the corresponding YAML defaults.')
    config_args, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description='Preprocess one or more Waymo TFRecords into the directory structure expected by the current LiDAR-GS Waymo dataloader.', parents=[config_parser])
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--tfrecord', default=None, help='Path to one Waymo TFRecord segment.')
    input_group.add_argument('--split-file', action='append', default=None, help='ImageSets text file containing Waymo case names, TFRecord filenames, or absolute TFRecord paths. Repeat this option to combine multiple split files while removing duplicate scenes.')
    parser.add_argument('--tfrecord-dir', default=None, help='Directory containing TFRecords referenced by --split-file. Not needed when every split entry is an absolute path.')
    parser.add_argument('--output-root', default=None, help='LiDAR-GS dataset root to generate.')
    parser.add_argument('--case', default=None, help='Optional case name. By default the TFRecord filename without the .tfrecord suffix is used.')
    parser.add_argument('--speed-threshold', type=float, default=0.5, help='Whole-track dynamic threshold in m/s. If a track reaches this XY speed in any frame, the ENTIRE track is classified as dynamic. Default: 0.5 m/s.')
    parser.add_argument('--box-margin', type=float, default=0.0, help='Optional enlargement of dynamic-object boxes when cropping points. Default: 0.0 m.')
    parser.add_argument('--frame-selection', choices=['segmentation_bounds', 'all'], default='segmentation_bounds', help='Frames written to LiDAR-GS: segmentation_bounds writes every frame from the first TOP-segmentation-labelled frame through the last, including all unlabelled intermediate frames; all writes the complete TFRecord. Missing frames receive KNN/track propagation. Selected source frames are reindexed contiguously from zero for LiDAR-GS.')
    parser.add_argument('--static-k', type=int, default=3, help='Validated static world-coordinate KNN neighbour count.')
    parser.add_argument('--static-max-distance', type=float, default=1.4, help='Validated static world-coordinate KNN threshold in metres.')
    parser.add_argument('--static-min-vote-fraction', type=float, default=0.5)
    parser.add_argument('--max-static-reference-points', type=int, default=5000000)
    parser.add_argument('--debug-propagation-source-frame', type=int, action='append', default=[], help='Waymo source frame to audit. Repeat for multiple frames. Unlabelled frames save the exact static-KNN neighbours, distances, votes and predictions without changing preprocessing behaviour.')
    parser.add_argument('--object-k', type=int, default=3)
    parser.add_argument('--object-max-distance', type=float, default=0.2)
    parser.add_argument('--object-distance-margin', type=float, default=0.03)
    parser.add_argument('--max-object-reference-points', type=int, default=100000)
    parser.add_argument('--association-min-points', type=int, default=3)
    parser.add_argument('--association-min-purity', type=float, default=0.5)
    parser.add_argument('--vehicle-component-margin', type=float, default=4.0)
    parser.add_argument('--vehicle-component-link-radius', type=float, default=0.75)
    parser.add_argument('--pedestrian-component-margin', type=float, default=0.75)
    parser.add_argument('--pedestrian-component-link-radius', type=float, default=0.35)
    parser.add_argument('--cyclist-component-margin', type=float, default=1.25)
    parser.add_argument('--cyclist-component-link-radius', type=float, default=0.5)
    parser.add_argument('--ground-ring-margin', type=float, default=1.0)
    parser.add_argument('--ground-plane-tolerance', type=float, default=0.08)
    parser.add_argument('--ground-ransac-iterations', type=int, default=80)
    parser.add_argument('--ground-min-ring-points', type=int, default=30)
    parser.add_argument('--strong-object-distance', type=float, default=0.08)
    parser.add_argument('--random-seed', type=int, default=13)
    parser.add_argument('--cpu-workers', type=int, default=0, help='CPU workers used by cKDTree/BLAS/TensorFlow. 0 = all CPUs visible to the process. Recommended: 24 on a 24-thread machine.')
    parser.add_argument('--continue-on-error', action=argparse.BooleanOptionalAction, default=False, help='Batch mode only: continue processing later scenes after a missing TFRecord or failed scene.')
    parser.add_argument('--skip-existing-scenes', action=argparse.BooleanOptionalAction, default=False, help='Batch mode only: skip a case only when its final PREPROCESSING_COMPLETE.json marker exists.')
    parser.add_argument('--save-combined-data-labeled', action=argparse.BooleanOptionalAction, default=True, help='Also save the convenience N x 7 float64 data_labeled matrix. Disable for large batches because it duplicates the named arrays.')
    parser.add_argument('--debug-foreground-source-frame',type=int,action='append',default=[])

    # Integrated strict static-cloud filtering. Raw static_recon_* files are ALWAYS kept.
    parser.add_argument('--run-static-filter', action=argparse.BooleanOptionalAction, default=True, help='After raw static accumulation, create a separate strict filtered static cloud. Raw static_recon_labels.npz/static_recon_voxels.pcd remain untouched.')
    parser.add_argument('--static-filter-mode', choices=['conflict','unsupported','both'], default='both')
    parser.add_argument('--static-filter-threshold-percentile', type=float, default=99.0)
    parser.add_argument('--static-filter-threshold-scale', type=float, default=2.0)
    parser.add_argument('--static-filter-min-class-threshold', type=float, default=0.15)
    parser.add_argument('--static-filter-max-class-threshold', type=float, default=1.0)
    parser.add_argument('--static-filter-conflict-max-distance', type=float, default=0.25)
    parser.add_argument('--static-filter-conflict-margin', type=float, default=0.10)
    parser.add_argument('--static-filter-threshold-sample-max', type=int, default=100000)
    parser.add_argument('--static-filter-cross-frame-k', type=int, default=32)
    parser.add_argument('--static-filter-save-native-suspicious', action='store_true', help='Also allow native Waymo-labelled points to be marked suspicious. Default: native points remain trusted.')
    parser.add_argument('--static-filter-ground-xy-radius', type=float, default=0.50)
    parser.add_argument('--static-filter-ground-max-dz', type=float, default=0.20)
    parser.add_argument('--static-filter-ground-k', type=int, default=8)
    parser.add_argument('--static-filter-ground-min-neighbours', type=int, default=3)

    if config_args.config is not None:
        config_path = os.path.abspath(config_args.config)
        if not os.path.isfile(config_path):
            parser.error(f'YAML configuration does not exist: {config_path}')
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        if loaded_config is None:
            loaded_config = {}
        if not isinstance(loaded_config, dict):
            parser.error('The YAML root must be a mapping/dictionary.')
        try:
            config_values = flatten_config(loaded_config)
        except ValueError as error:
            parser.error(str(error))
        valid_destinations = {action.dest for action in parser._actions if action.dest not in {'help', 'config'}}
        unknown = sorted(set(config_values) - valid_destinations)
        if unknown:
            parser.error('Unknown YAML configuration keys: ' + ', '.join(unknown))
        parser.set_defaults(**config_values)

    args = parser.parse_args()
    cli_has_tfrecord = any((token == '--tfrecord' or token.startswith('--tfrecord=') for token in sys.argv[1:]))
    cli_has_split_file = any((token == '--split-file' or token.startswith('--split-file=') for token in sys.argv[1:]))

    if cli_has_split_file:
        explicit_split_files = []
        raw_cli = sys.argv[1:]
        cli_index = 0
        while cli_index < len(raw_cli):
            token = raw_cli[cli_index]
            if token == '--split-file':
                if cli_index + 1 >= len(raw_cli):
                    parser.error('--split-file requires a path value.')
                explicit_split_files.append(raw_cli[cli_index + 1])
                cli_index += 2
                continue
            if token.startswith('--split-file='):
                explicit_split_files.append(token.split('=', 1)[1])
            cli_index += 1
        args.split_file = explicit_split_files

    if cli_has_tfrecord:
        args.split_file = None
    elif cli_has_split_file:
        args.tfrecord = None

    if (args.tfrecord is None) == (args.split_file is None):
        parser.error('Provide exactly one of tfrecord or split_file, in YAML or on the command line.')

    if args.output_root is None:
        parser.error('output_root must be provided in YAML or via --output-root.')

    print(f'CPU workers              : {CPU_WORKERS}')
    print(f'TensorFlow inter-op      : {TF_INTEROP_WORKERS}')

    if args.split_file is not None:
        if args.tfrecord_dir is None:
            args.tfrecord_dir = os.path.dirname(os.path.abspath(args.split_file[0]))
        run_batch(args)
        return

    if args.case is None:
        case = os.path.basename(args.tfrecord)
        if case.endswith('.tfrecord'):
            case = case[:-len('.tfrecord')]
    else:
        case = args.case

    root = os.path.abspath(args.output_root)
    meta_dir = os.path.join(root, 'meta_infos')
    pcd_dir = os.path.join(root, 'pcds_new', case)
    calib_dir = os.path.join(root, 'laser_calibrations', case, 'laser_calibrations')
    beam_dir = os.path.join(root, 'temp', case, 'beam_inclinations')
    static_dir = os.path.join(root, 'recon_related', case)
    dynamic_objects_root = os.path.join(root, 'temp', case, 'occ', 'preproc', 'dynamic', 'objects')
    temp_case_dir = os.path.join(root, 'temp', case)

    for directory in [meta_dir, pcd_dir, calib_dir, beam_dir, static_dir, dynamic_objects_root, temp_case_dir]:
        os.makedirs(directory, exist_ok=True)
    print('\nLoading Waymo TFRecord...')

    frames = load_frames(args.tfrecord)
    num_frames = len(frames)

    if num_frames == 0:
        raise RuntimeError('No frames found in TFRecord.')
    labelled_frame_indices = discover_segmentation_frames(frames)

    if not labelled_frame_indices:
        raise RuntimeError('No frames with complete TOP-LiDAR segmentation labels found.')

    output_source_indices = choose_output_frame_indices(num_frames, labelled_frame_indices, args.frame_selection)
    source_to_output_index = {source_idx: output_idx for output_idx, source_idx in enumerate(output_source_indices)}

    print(f'TOP segmentation-labelled frames: {labelled_frame_indices}')
    print(f'Frame selection mode       : {args.frame_selection}')
    print(f'Source frames written      : {output_source_indices[0]}..{output_source_indices[-1]} ({len(output_source_indices)} frames)')
    print('\nAnalysing complete object tracks...')

    tracks = collect_track_motion(frames)
    dynamic_tracks, track_stats = classify_tracks(tracks, speed_threshold=args.speed_threshold)
    id_map = assign_numeric_ids(dynamic_tracks, tracks)
    instance_tracks = {track_id for track_id, observations in tracks.items() if observations and observations[0]['type'] in DYNAMIC_BOX_TYPES}
    point_instance_id_map = assign_point_instance_ids(instance_tracks, id_map, tracks)

    print(f'Total labeled tracks      : {len(tracks)}')
    print(f'Dynamic complete tracks   : {len(dynamic_tracks)}')
    print(f'Static complete tracks    : {len(tracks) - len(dynamic_tracks)}')
    print('\nTrack classification:')
    print(f"{'folder':>6s}  {'type':>10s}  {'frames':>6s}  {'max m/s':>9s}  {'dynamic':>8s}")
    print('-' * 55)

    ordered_track_ids = sorted(tracks.keys(), key=lambda tid: min((obs['frame_idx'] for obs in tracks[tid])))

    for track_id in ordered_track_ids:
        s = track_stats[track_id]
        folder_id = id_map.get(track_id, '-')
        print(f"{folder_id:>6s}  {s['type']:>10s}  {s['num_frames']:6d}  {s['max_speed_mps']:9.3f}  {str(s['dynamic']):>8s}")

    rng = np.random.default_rng(args.random_seed)
    track_references, static_reference_xyz, static_reference_semantic, static_reference_tree = build_label_references(frames, labelled_frame_indices, instance_tracks, args, rng)

    print(f'Static semantic references : {len(static_reference_xyz):,}')
    print(f'Tracked instance references: {sum((ref.positive_tree is not None for ref in track_references.values()))}')

    top_calib = get_top_calibration(frames[0])
    top_extrinsic = np.asarray(top_calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
    top_beams = np.asarray(top_calib.beam_inclinations, dtype=np.float64)

    if top_beams.shape != (64,):
        raise RuntimeError(f'Expected 64 TOP beam inclinations; got {top_beams.shape}.')
    np.save(os.path.join(beam_dir, 'beam_inclinations.npy'), top_beams.astype(np.float32))

    meta_frames = []
    frame_poses = []
    calibration_beams = []
    calibration_extrinsics = []
    static_world_parts = []
    static_semantic_parts = []
    static_ground_parts = []
    static_instance_parts = []
    static_intensity_parts = []
    static_label_confidence_parts = []
    static_observation_frame_parts = []
    object_local_parts = defaultdict(list)
    object_labeled_parts = defaultdict(list)
    object_info = defaultdict(lambda: {'tracks': {}})
    total_top_points = 0
    total_static_points = 0
    total_dynamic_removed = 0
    total_quarantined_foreground = 0

    for output_frame_idx, source_frame_idx in enumerate(output_source_indices):
        frame = frames[source_frame_idx]
        print(f'\nOutput frame {output_frame_idx:03d}/{len(output_source_indices) - 1:03d} (Waymo source frame {source_frame_idx:03d})')
        pcd_vehicle, gt_segmentation = extract_top_both_returns(frame)
        xyz_vehicle = pcd_vehicle[:, :3]
        point_count = len(pcd_vehicle)
        total_top_points += point_count
        print(f'  TOP points, both returns : {point_count:,}')
        semantic_id = np.full(point_count, NN_UNASSIGNED, dtype=np.int16)
        instance_id = np.zeros(point_count, dtype=np.int32)
        is_dynamic = np.zeros(point_count, dtype=bool)
        label_confidence = np.zeros(point_count, dtype=np.float32)
        corrected_boxes_this_frame = {}
        debug_fg=source_frame_idx in args.debug_foreground_source_frame
        debug_track_hits=[] if debug_fg else None

        for label in frame.laser_labels:
            if label.id not in instance_tracks or label.type not in DYNAMIC_BOX_TYPES:
                continue
            expansion = args.box_margin
            if label.id in dynamic_tracks:
                expansion, _ = dynamic_component_parameters(label.type, args)
            candidate = box_mask(xyz_vehicle, label.box, margin=expansion)
            candidate_indices = np.flatnonzero(candidate)

            if debug_fg and len(candidate_indices):
                debug_track_hits.append({
                    "track_id":label.id,
                    "track_type":int(label.type),
                    "is_dynamic_track":label.id in dynamic_tracks,
                    "candidate_indices":candidate_indices.copy(),
                    "inside_original_box":np.flatnonzero(box_mask(xyz_vehicle,label.box,margin=0.0))
                })

            reference = track_references.get(label.id)
            offsets, _, _ = interpolate_extension(reference.box_extensions if reference is not None else {}, source_frame_idx)
            local_candidates = transform_points(xyz_vehicle, np.linalg.inv(make_T_b2l(label.box)))
            half = np.array([label.box.length, label.box.width, label.box.height]) / 2
            candidate |= np.all((local_candidates >= -half - offsets[:3]) & (local_candidates <= half + offsets[3:]), axis=1)

            if label.type != label_pb2.Label.TYPE_VEHICLE:
                candidate &= box_mask(xyz_vehicle, label.box, margin=0.0)
                if gt_segmentation is not None:
                    candidate &= np.isin(gt_segmentation[:, 1], list(BOX_COMPATIBLE_SEMANTICS.get(int(label.type), set())))
            candidate_indices = np.flatnonzero(candidate)

            if len(candidate_indices) == 0:
                continue
            accepted_indices = np.empty(0, dtype=np.int64)
            accepted_confidence = np.empty(0, dtype=np.float32)
            used_exact_segmentation = False

            if gt_segmentation is not None:
                gt_semantic = gt_segmentation[:, 1]
                accepted_indices, corrected_box = connected_semantic_component(xyz_vehicle, gt_semantic, label, args)
                if len(accepted_indices) >= args.association_min_points:
                    accepted_confidence = np.ones(len(accepted_indices), dtype=np.float32)
                    used_exact_segmentation = True
                    corrected_boxes_this_frame[label.id] = corrected_box
                else:
                    accepted_indices = np.empty(0, dtype=np.int64)
            if debug_fg:
                debug_track_hits[-1]["accepted_indices"]=accepted_indices.copy()
                debug_track_hits[-1]["accepted_confidence"]=accepted_confidence.copy()
                debug_track_hits[-1]["used_exact_segmentation"]=used_exact_segmentation      

            if len(accepted_indices) == 0:
                accepted_indices, accepted_confidence = classify_dynamic_candidates(xyz_vehicle, candidate_indices, label, track_references.get(label.id), args, rng)

            replace = accepted_confidence > label_confidence[accepted_indices]
            accepted_indices = accepted_indices[replace]
            accepted_confidence = accepted_confidence[replace]
            if len(accepted_indices) == 0:
                continue
            stable_instance = point_instance_id_map[label.id]
            instance_id[accepted_indices] = stable_instance
            is_dynamic[accepted_indices] = label.id in dynamic_tracks
            label_confidence[accepted_indices] = accepted_confidence
            if used_exact_segmentation:
                semantic_id[accepted_indices] = gt_segmentation[accepted_indices, 1].astype(np.int16)
            else:
                semantic_id[accepted_indices] = track_semantic_id(label.id, int(label.type), track_references)

        if debug_fg:
            debug_dir=os.path.join(temp_case_dir,"foreground_association_debug")
            os.makedirs(debug_dir,exist_ok=True)
            np.savez_compressed(
                os.path.join(debug_dir,f"source_{source_frame_idx:03d}_foreground_association_debug.npz"),
                xyz_vehicle=xyz_vehicle.astype(np.float32),
                instance_id=instance_id,
                is_dynamic=is_dynamic,
                label_confidence=label_confidence,
                track_hits=np.asarray(debug_track_hits,dtype=object)
            )
        T_world_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        background_mask = instance_id == 0

        if gt_segmentation is not None:
            semantic_id[background_mask] = gt_segmentation[background_mask, 1].astype(np.int16)
            label_confidence[background_mask] = (semantic_id[background_mask] != UNDEFINED).astype(np.float32)
            if source_frame_idx in set(args.debug_propagation_source_frame):
                print(f'  DEBUG requested for source frame {source_frame_idx}, but this frame has native Waymo segmentation; static KNN propagation is not executed.')
        else:
            background_indices = np.flatnonzero(background_mask)
            background_world = transform_points(xyz_vehicle[background_indices], T_world_vehicle)
            debug_this_frame = source_frame_idx in set(args.debug_propagation_source_frame)
            if debug_this_frame:
                propagated_semantic, propagated_confidence, propagation_debug = propagate_static_semantics(background_world, static_reference_tree, static_reference_semantic, args.static_k, args.static_max_distance, args.static_min_vote_fraction, return_debug=True)
                save_static_propagation_debug(os.path.join(temp_case_dir, 'semantic_propagation_debug'), source_frame_idx, output_frame_idx, pcd_vehicle, background_indices, background_world, propagated_semantic, propagated_confidence, propagation_debug, static_reference_tree, static_reference_semantic, args.static_max_distance, args.static_min_vote_fraction)
            else:
                propagated_semantic, propagated_confidence = propagate_static_semantics(background_world, static_reference_tree, static_reference_semantic, args.static_k, args.static_max_distance, args.static_min_vote_fraction)
            semantic_id[background_indices] = propagated_semantic
            label_confidence[background_indices] = propagated_confidence

        ground_id = semantic_to_ground_id(semantic_id)
        unowned_foreground = np.isin(semantic_id, list(FOREGROUND_SEMANTIC_CLASSES)) & (instance_id == 0)
        frame_arrays = {'data': pcd_vehicle, 'semantic_id': semantic_id, 'ground_id': ground_id, 'instance_id': instance_id, 'is_dynamic': is_dynamic, 'quarantined_unowned_foreground': unowned_foreground, 'label_confidence': label_confidence, 'source_frame_index': np.asarray([source_frame_idx], dtype=np.int32), 'output_frame_index': np.asarray([output_frame_idx], dtype=np.int32), 'has_ground_truth_segmentation': np.asarray([gt_segmentation is not None], dtype=bool)}

        if args.save_combined_data_labeled:
            frame_arrays['data_labeled'] = np.column_stack([pcd_vehicle, semantic_id, ground_id, instance_id]).astype(np.float64)
            frame_arrays['data_labeled_columns'] = np.asarray(['x_vehicle', 'y_vehicle', 'z_vehicle', 'intensity', 'semantic_id', 'ground_id', 'instance_id'])

        np.savez(os.path.join(pcd_dir, f'{output_frame_idx:03d}.npz'), **frame_arrays)

        meta_frame = build_meta_frame(frame, output_frame_idx, case, point_instance_id_map, dynamic_tracks, id_map, track_references, top_extrinsic)
        meta_frame['source_frame_index'] = int(source_frame_idx)
        meta_frames.append(meta_frame)
        frame_poses.append(T_world_vehicle)
        calibration_beams.append(top_beams.copy())
        calibration_extrinsics.append(top_extrinsic.copy())
        static_keep = ~is_dynamic & ~unowned_foreground
        dynamic_points_this_frame = int(np.count_nonzero(is_dynamic))
        quarantined_this_frame = int(np.count_nonzero(unowned_foreground))
        dynamic_objects_this_frame = 0
        frame_box_diagnostics = {}

        for label in frame.laser_labels:
            if label.id not in dynamic_tracks:
                continue
            stable_instance = point_instance_id_map[label.id]
            object_mask = is_dynamic & (instance_id == stable_instance)
            count_object = int(np.count_nonzero(object_mask))
            if label.type != label_pb2.Label.TYPE_VEHICLE:
                if np.any(object_mask & ~box_mask(xyz_vehicle, label.box, margin=0.0)):
                    raise RuntimeError(f'Small-object containment failure: {label.id}, frame {source_frame_idx}')
            reference = track_references.get(label.id)
            offsets, extension_source, support_frames = interpolate_extension(reference.box_extensions if reference is not None else {}, source_frame_idx)
            if label.type != label_pb2.Label.TYPE_VEHICLE:
                offsets = np.zeros(6)
                extension_source = 'original_box_small_object'
                support_frames = []
            owned_local = transform_points(xyz_vehicle[object_mask], np.linalg.inv(make_T_b2l(label.box)))
            final_box, final_offsets = extended_track_box(label, owned_local, offsets)
            box_record = {'box_vehicle_original': [float(label.box.center_x), float(label.box.center_y), float(label.box.center_z), float(label.box.length), float(label.box.width), float(label.box.height), float(label.box.heading)], 'box_vehicle_corrected': final_box, 'extension_source': extension_source, 'extension_support_source_frames': support_frames, 'propagated_face_extensions_m': offsets.tolist(), 'final_face_extensions_m': final_offsets.tolist(), 'face_order': ['minus_x', 'minus_y', 'minus_z', 'plus_x', 'plus_y', 'plus_z'], 'expanded_by_current_points': bool(np.any(final_offsets > offsets + 1e-06)), 'num_associated_points': count_object, 'source_frame_index': int(source_frame_idx), 'output_frame_index': int(output_frame_idx)}
            frame_box_diagnostics[str(id_map[label.id])] = box_record
            if count_object == 0:
                continue
            dynamic_objects_this_frame += 1
            object_points_vehicle = xyz_vehicle[object_mask]
            object_id = id_map[label.id]
            T_b2l = make_T_b2l(label.box)
            T_l2b = np.linalg.inv(T_b2l)
            object_points_local = transform_points(object_points_vehicle, T_l2b)
            object_local_parts[object_id].append(object_points_local)
            object_ground = ground_id[object_mask]
            object_semantic = semantic_id[object_mask]
            object_intensity = pcd_vehicle[object_mask, 3]
            object_instance = instance_id[object_mask]
            object_labeled_parts[object_id].append(np.column_stack([object_points_local, object_intensity, object_semantic, object_ground, object_instance]))
            object_info[object_id]['tracks'][str(output_frame_idx)] = {'name': f'{output_frame_idx:03d}.pcd', 'source_frame_index': int(source_frame_idx), 'type': TYPE_NAMES.get(int(label.type), 'UNKNOWN'), 'T_b2l': T_b2l.reshape(-1).tolist(), 'box_vehicle': [float(label.box.center_x), float(label.box.center_y), float(label.box.center_z), float(label.box.length), float(label.box.width), float(label.box.height), float(label.box.heading)], **box_record, 'association_method': 'semantic_connected_component' if label.id in corrected_boxes_this_frame else 'object_reference_knn_fallback', 'num_associated_points': count_object, 'box_pose_vehicle': T_b2l.reshape(-1).tolist(), 'box_pose_world': (T_world_vehicle @ T_b2l).reshape(-1).tolist(), 'box_pose_top_lidar': (np.linalg.inv(top_extrinsic) @ T_b2l).reshape(-1).tolist(), 'velocity_vehicle_xy': [float(label.metadata.speed_x), float(label.metadata.speed_y)], 'num_lidar_points_in_box': int(getattr(label, 'num_lidar_points_in_box', 0)), 'semantic_id': int(track_semantic_id(label.id, int(label.type), track_references)), 'instance_id': int(stable_instance), 'is_dynamic': True}

        static_vehicle = xyz_vehicle[static_keep]
        diagnostics_dir = os.path.join(temp_case_dir, 'box_extension_diagnostics')
        os.makedirs(diagnostics_dir, exist_ok=True)

        with open(os.path.join(diagnostics_dir, f'{output_frame_idx:03d}.json'), 'w') as handle:
            json.dump(frame_box_diagnostics, handle, indent=2)

        static_world = transform_points(static_vehicle, T_world_vehicle)
        static_world_parts.append(static_world)
        static_semantic_parts.append(semantic_id[static_keep])
        static_ground_parts.append(ground_id[static_keep])
        static_instance_parts.append(instance_id[static_keep])
        static_intensity_parts.append(pcd_vehicle[static_keep, 3])
        static_label_confidence_parts.append(label_confidence[static_keep])
        static_observation_frame_parts.append(np.full(len(static_world), output_frame_idx, dtype=np.int32))
        total_static_points += len(static_world)
        total_dynamic_removed += dynamic_points_this_frame
        total_quarantined_foreground += quarantined_this_frame

        print(f'  Dynamic objects removed : {dynamic_objects_this_frame}')
        print(f'  Dynamic points removed  : {dynamic_points_this_frame:,}')
        print(f'  Static points retained  : {len(static_world):,}')
        print(f'  Semantic NN-unassigned : {np.count_nonzero(semantic_id == NN_UNASSIGNED):,}')
        print(f'  Waymo undefined        : {np.count_nonzero(semantic_id == UNDEFINED):,}')

    meta = {'car_type': 'WAYMO', 'seq_name': case, 'frames': meta_frames, 'stuff_start_index': 0, 'obj_id_dict': {waymo_id: int(object_id) for waymo_id, object_id in id_map.items()}, 'id_obj_dict': {int(object_id): waymo_id for waymo_id, object_id in id_map.items()}, 'point_instance_id_dict': {waymo_id: int(instance_id_value) for waymo_id, instance_id_value in point_instance_id_map.items()}, 'point_instance_waymo_id_dict': {int(instance_id_value): waymo_id for waymo_id, instance_id_value in point_instance_id_map.items()}, 'frame_selection': args.frame_selection, 'source_frame_indices': np.asarray(output_source_indices, dtype=np.int32), 'segmentation_labelled_source_frames': np.asarray(labelled_frame_indices, dtype=np.int32), 'point_attribute_schema': {'legacy_data': ['x_vehicle', 'y_vehicle', 'z_vehicle', 'intensity'], 'data_labeled': ['x_vehicle', 'y_vehicle', 'z_vehicle', 'intensity', 'semantic_id', 'ground_id', 'instance_id'], 'semantic_id': {'-1': 'NN-unassigned', '0': 'Waymo UNDEFINED', '1-22': 'Waymo semantic class'}, 'ground_id': {'-1': 'undefined or NN-unassigned', '0': 'non-ground', '1': 'ground'}, 'instance_id': {'0': 'undefined, background, or no instance', 'positive': 'stable tracked instance'}, 'static_recon_labels.npz': {'coordinate_frame': 'world', 'label_confidence': 'semantic/instance association confidence in [0, 1]', 'observation_frame_index': 'LiDAR-GS output-frame index; use it to select frame_pose from laser_calibrations.npz'}}}
    meta_path = os.path.join(meta_dir, f'{case}.pkl')

    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'\nSaved meta info:\n{meta_path}')
    calibration_path = os.path.join(calib_dir, 'laser_calibrations.npz')
    np.savez(calibration_path, beam_inclinations=np.stack(calibration_beams, axis=0), extrinsic=np.stack(calibration_extrinsics, axis=0), frame_pose=np.stack(frame_poses, axis=0))
    print(f'Saved laser calibration:\n{calibration_path}')

    static_scene = np.concatenate(static_world_parts, axis=0)
    static_path = os.path.join(static_dir, 'static_recon_voxels.pcd')

    print('\nWriting RAW static accumulated scene (kept unfiltered)...')
    print(f'Static points: {len(static_scene):,}')
    print('No voxel downsampling.')
    print('No statistical filtering.')
    print('No radius filtering.')

    save_pcd_xyz(static_scene, static_path)

    static_semantic = np.concatenate(static_semantic_parts, axis=0).astype(np.int16)
    static_ground = np.concatenate(static_ground_parts, axis=0).astype(np.int8)
    static_instance = np.concatenate(static_instance_parts, axis=0).astype(np.int32)
    static_intensity = np.concatenate(static_intensity_parts, axis=0).astype(np.float64)
    static_label_confidence = np.concatenate(static_label_confidence_parts, axis=0).astype(np.float32)
    static_observation_frame = np.concatenate(static_observation_frame_parts, axis=0).astype(np.int32)
    static_labeled_path = os.path.join(static_dir, 'static_recon_labels.npz')
    static_arrays = {'xyz': static_scene.astype(np.float32), 'intensity': static_intensity, 'semantic_id': static_semantic, 'ground_id': static_ground, 'instance_id': static_instance, 'label_confidence': static_label_confidence, 'observation_frame_index': static_observation_frame, 'coordinate_frame': np.asarray('world')}

    if args.save_combined_data_labeled:
        static_arrays['data_labeled'] = np.column_stack([static_scene, static_intensity, static_semantic, static_ground, static_instance]).astype(np.float64)
        static_arrays['data_labeled_columns'] = np.asarray(['x_world', 'y_world', 'z_world', 'intensity', 'semantic_id', 'ground_id', 'instance_id'])
    np.savez(static_labeled_path, **static_arrays)

    print(f'Saved static scene:\n{static_path}')
    print(f'Saved labelled static scene:\n{static_labeled_path}')
    print('\nWriting complete dynamic tracks...')

    for object_id in sorted(object_local_parts.keys(), key=lambda x: int(x)):
        object_dir = os.path.join(dynamic_objects_root, object_id)
        os.makedirs(object_dir, exist_ok=True)
        stitch = np.concatenate(object_local_parts[object_id], axis=0)
        stitch_path = os.path.join(object_dir, 'stitch.pcd')
        save_pcd_xyz(stitch, stitch_path)
        stitch_labeled = np.concatenate(object_labeled_parts[object_id], axis=0)
        stitch_labeled_path = os.path.join(object_dir, 'stitch_labeled.npz')
        object_arrays = {'xyz': stitch_labeled[:, :3].astype(np.float32), 'intensity': stitch_labeled[:, 3].astype(np.float64), 'semantic_id': stitch_labeled[:, 4].astype(np.int16), 'ground_id': stitch_labeled[:, 5].astype(np.int8), 'instance_id': stitch_labeled[:, 6].astype(np.int32)}
        if args.save_combined_data_labeled:
            object_arrays['data_labeled'] = stitch_labeled.astype(np.float64)
            object_arrays['data_labeled_columns'] = np.asarray(['x_object', 'y_object', 'z_object', 'intensity', 'semantic_id', 'ground_id', 'instance_id'])
        np.savez(stitch_labeled_path, **object_arrays)
        waymo_track_id = {object_folder: waymo_id for waymo_id, object_folder in id_map.items()}[object_id]
        object_info[object_id]['waymo_track_id'] = waymo_track_id
        object_info[object_id]['instance_id'] = int(point_instance_id_map[waymo_track_id])
        object_info[object_id]['semantic_id'] = track_semantic_id(waymo_track_id, tracks[waymo_track_id][0]['type'], track_references)
        info_path = os.path.join(object_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(object_info[object_id], f)
        print(f'  Object {object_id}: {len(stitch):,} stitched points')

    report = {}

    for track_id, values in track_stats.items():
        values = dict(values)
        if track_id in id_map:
            values['lidargs_object_id'] = int(id_map[track_id])
        else:
            values['lidargs_object_id'] = None
        values['point_instance_id'] = point_instance_id_map.get(track_id, 0)
        report[track_id] = values
    report_path = os.path.join(temp_case_dir, 'track_classification.json')

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    mapping_path = os.path.join(temp_case_dir, 'dynamic_track_id_mapping.json')

    with open(mapping_path, 'w') as f:
        json.dump({'waymo_to_lidargs': {waymo_id: int(object_id) for waymo_id, object_id in id_map.items()}, 'lidargs_to_waymo': {object_id: waymo_id for waymo_id, object_id in id_map.items()}}, f, indent=2)
    point_mapping_path = os.path.join(temp_case_dir, 'point_instance_id_mapping.json')

    with open(point_mapping_path, 'w') as f:
        json.dump({'undefined_or_no_instance': 0, 'waymo_to_point_instance': point_instance_id_map, 'point_instance_to_waymo': {str(instance_value): waymo_id for waymo_id, instance_value in point_instance_id_map.items()}}, f, indent=2)
    stage_a_tracks = {}

    for output_idx, meta_frame in enumerate(meta_frames):
        labels = meta_frame['obj_label']
        for box_idx, waymo_track_id in enumerate(labels['gt_boxes_token'].tolist()):
            waymo_track_id = str(waymo_track_id)
            instance_value = int(labels['gt_box_instance_ids'][box_idx])
            entry = stage_a_tracks.setdefault(waymo_track_id, {'waymo_track_id': waymo_track_id, 'instance_id': instance_value, 'semantic_id': int(labels['gt_box_semantic_ids'][box_idx]), 'waymo_type': int(labels['gt_box_waymo_types'][box_idx]), 'type_name': str(labels['gt_names'][box_idx]), 'is_dynamic': bool(labels['gt_box_is_dynamic'][box_idx]), 'lidargs_object_id': int(labels['gt_box_lidargs_object_ids'][box_idx]), 'frames': {}})
            entry['frames'][str(output_idx)] = {'output_frame_index': int(output_idx), 'source_frame_index': int(meta_frame['source_frame_index']), 'timestamp_micros': int(meta_frame['log_time_stamp']), 'box_vehicle': labels['gt_boxes'][box_idx].tolist(), 'box_pose_vehicle': labels['gt_box_pose_vehicle'][box_idx].reshape(-1).tolist(), 'box_pose_world': labels['gt_box_pose_world'][box_idx].reshape(-1).tolist(), 'box_pose_top_lidar': labels['gt_box_pose_top_lidar'][box_idx].reshape(-1).tolist(), 'velocity_vehicle_xy': labels['gt_boxes_velocity'][box_idx].tolist(), 'num_lidar_points_in_box': int(labels['gt_box_num_lidar_points'][box_idx])}
    stage_a_tracks_path = os.path.join(temp_case_dir, 'stage_a_tracks.json')

    with open(stage_a_tracks_path, 'w') as f:
        json.dump({'scene': case, 'coordinate_convention': {'box_vehicle': '[center_x, center_y, center_z, length, width, height, heading] in Waymo vehicle frame', 'box_pose_*': 'row-major 4x4 transform from box-local coordinates to the named coordinate frame'}, 'tracks': stage_a_tracks}, f, indent=2)
    label_config_path = os.path.join(temp_case_dir, 'label_propagation_config.json')

    with open(label_config_path, 'w') as f:
        json.dump({'frame_selection': args.frame_selection, 'segmentation_labelled_source_frames': labelled_frame_indices, 'output_to_source_frame': {str(output_idx): int(source_idx) for output_idx, source_idx in enumerate(output_source_indices)}, 'source_to_output_frame': {str(source_idx): int(output_idx) for source_idx, output_idx in source_to_output_index.items()}, 'static_knn': {'k': args.static_k, 'max_distance_m': args.static_max_distance, 'min_vote_fraction': args.static_min_vote_fraction}, 'object_knn': {'k': args.object_k, 'max_distance_m': args.object_max_distance, 'distance_margin_m': args.object_distance_margin}, 'ground_id_convention': {'-1': 'undefined or NN-unassigned', '0': 'non-ground semantic class 1-16', '1': 'ground semantic class 17-22'}, 'instance_id_convention': {'0': 'undefined, background, or no assigned instance', 'positive': 'stable scene-level tracked instance'}}, f, indent=2)
    # ---------------------------------------------------------------------
    # Integrated strict static post-filter. The raw accumulated files above
    # are never overwritten. To reduce peak RAM, release preprocessing-only
    # objects before loading the saved raw static NPZ for filtering.
    # ---------------------------------------------------------------------
    static_filter_report = None
    static_filter_dir = os.path.join(static_dir, 'static_filter')
    if args.run_static_filter:
        import gc
        # These objects can be very large. At this point all preprocessing,
        # dynamic-object writing and metadata writing are already complete.
        del frames
        del static_world_parts, static_semantic_parts, static_ground_parts, static_instance_parts
        del static_intensity_parts, static_label_confidence_parts, static_observation_frame_parts
        del object_local_parts, object_labeled_parts
        del static_reference_xyz, static_reference_semantic, static_reference_tree, track_references
        del static_scene, static_semantic, static_ground, static_instance, static_intensity
        del static_label_confidence, static_observation_frame, static_arrays
        gc.collect()
        static_filter_report = run_integrated_static_filter(static_labeled_path, static_filter_dir, output_source_indices, labelled_frame_indices, args)

    completion_marker_path = os.path.join(temp_case_dir, 'PREPROCESSING_COMPLETE.json')
    marker = {
        'case': case, 'status': 'completed', 'output_frames': int(len(output_source_indices)),
        'source_frame_start': int(output_source_indices[0]), 'source_frame_end': int(output_source_indices[-1]),
        'save_combined_data_labeled': bool(args.save_combined_data_labeled), 'cpu_workers': int(CPU_WORKERS),
        'raw_static_cloud_preserved': True, 'raw_static_npz': static_labeled_path, 'raw_static_pcd': static_path,
        'static_filter_requested': bool(args.run_static_filter), 'static_filter_completed': bool(static_filter_report is not None),
        'static_filter_output_dir': static_filter_dir if args.run_static_filter else None,
        'static_filter_ground_max_dz_m': float(args.static_filter_ground_max_dz) if args.run_static_filter else None
    }
    with open(completion_marker_path, 'w') as f:
        json.dump(marker, f, indent=2)

    print('\n' + '=' * 72)
    print('PREPROCESSING COMPLETE')
    print('=' * 72)
    print(f'Case                     : {case}')
    print(f'TFRecord frames          : {num_frames}')
    print(f'LiDAR-GS output frames   : {len(output_source_indices)}')
    print(f'Frame selection          : {args.frame_selection}')
    print(f'Output source range      : {output_source_indices[0]}..{output_source_indices[-1]}')
    print(f'Segmentation frames      : {len(labelled_frame_indices)}')
    print(f'Total TOP points         : {total_top_points:,}')
    print(f'Dynamic complete tracks  : {len(dynamic_tracks)}')
    print(f'Static complete tracks   : {len(tracks) - len(dynamic_tracks)}')
    print(f'Dynamic points removed   : {total_dynamic_removed:,}')
    print(f'Quarantined foreground   : {total_quarantined_foreground:,}')
    print(f'Static points accumulated: {total_static_points:,}')
    print(f'Speed threshold          : {args.speed_threshold:.3f} m/s')
    print(f'Static semantic KNN      : K={args.static_k}, {args.static_max_distance:.2f} m')
    print(f'Dynamic object KNN       : K={args.object_k}, {args.object_max_distance:.2f} m')
    print()
    print('Static scene processing:')
    print('  RAW accumulation       : PRESERVED')
    print('  voxel downsampling     : NO')
    print('  statistical filtering  : NO')
    print('  radius filtering       : NO')
    print(f'  strict static filter   : {"YES" if args.run_static_filter else "NO"}')
    if args.run_static_filter:
        print(f'  ground vertical check  : |dz| <= {args.static_filter_ground_max_dz:.2f} m')
        print(f'  strict output dir      : {static_filter_dir}')
    print()
    print(f'Output root:\n{root}')
    
if __name__ == '__main__':
    main()