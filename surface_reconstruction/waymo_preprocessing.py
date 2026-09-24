# UNIFIED WAYMO PREPROCESSING PIPELINE - CPU/CUDA BACKEND EDITION
# Stages: TFRecord accumulation -> strict static filter -> ground densification.
# Backend switch: --compute-backend cpu|cuda; multi-GPU: --gpu-ids 0,1,2.
# Canonical final output: recon_related/<case>/static_recon_labels.npz
# Intermediate diagnostics/products: recon_related/<case>/preprocessing_residues/

#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
import subprocess
import time
import zipfile
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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path
import numpy as np
from numpy.lib import format as npy_format
import open3d as o3d
import tensorflow as tf
import yaml
from scipy.spatial import cKDTree
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils

# -----------------------------------------------------------------------------
# SELECTABLE CPU / CUDA SPATIAL SEARCH BACKEND
# -----------------------------------------------------------------------------
# CPU mode uses scipy.spatial.cKDTree exactly as before.
# CUDA mode uses cupyx.scipy.spatial.KDTree for the large exact k-NN searches.
# Small trees and variable-length radius/connected-component searches stay on CPU.
# The numerical preprocessing stages and output schema are otherwise unchanged.
class SpatialSearchManager:
    def __init__(self):
        self.mode = 'cpu'
        self.gpu_ids = [0]
        self.gpu_min_tree_points = 50000
        self.gpu_query_chunk_size = 200000
        self.cp = None
        self.CuKDTree = None
        self.lock = Lock()
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'tree_build_calls': 0, 'cpu_tree_build_calls': 0, 'gpu_tree_build_calls': 0,
            'query_calls': 0, 'cpu_query_calls': 0, 'gpu_query_calls': 0,
            'reference_points_built': 0, 'query_points': 0,
            'tree_build_s': 0.0, 'cpu_tree_build_s': 0.0, 'gpu_tree_build_s': 0.0,
            'query_s': 0.0, 'cpu_query_s': 0.0, 'gpu_query_s': 0.0,
            'gpu_replica_builds': 0
        }

    def configure(self, mode='cpu', gpu_ids='0', gpu_min_tree_points=50000, gpu_query_chunk_size=200000):
        self.mode = str(mode).lower()
        self.gpu_min_tree_points = max(1, int(gpu_min_tree_points))
        self.gpu_query_chunk_size = max(1, int(gpu_query_chunk_size))
        if isinstance(gpu_ids, str):
            ids = [int(x.strip()) for x in gpu_ids.split(',') if x.strip()]
        else:
            ids = [int(x) for x in gpu_ids]
        self.gpu_ids = ids or [0]
        self.reset_stats()
        if self.mode == 'cpu':
            return
        if self.mode != 'cuda':
            raise ValueError(f'Unknown compute backend: {mode}')
        try:
            import cupy as cp
            from cupyx.scipy.spatial import KDTree as CuKDTree
        except Exception as exc:
            raise RuntimeError('CUDA backend requested but CuPy/cupyx.scipy.spatial.KDTree is unavailable. Install a CuPy build matching your CUDA toolkit and the cuVS dependency required by cupyx.scipy.spatial.') from exc
        count = int(cp.cuda.runtime.getDeviceCount())
        bad = [i for i in self.gpu_ids if i < 0 or i >= count]
        if bad:
            raise RuntimeError(f'Invalid GPU IDs {bad}; visible CUDA devices are 0..{count-1}.')
        self.cp = cp
        self.CuKDTree = CuKDTree

    @property
    def tag(self):
        return 'cpu' if self.mode == 'cpu' else 'cuda_gpu' + '-'.join(str(i) for i in self.gpu_ids)

    def should_use_cuda(self, n_points):
        return self.mode == 'cuda' and int(n_points) >= self.gpu_min_tree_points

    def add_stats(self, **values):
        with self.lock:
            for key, value in values.items():
                self.stats[key] = self.stats.get(key, 0) + value

    def snapshot(self):
        result = dict(self.stats)
        result.update({
            'requested_backend': self.mode,
            'gpu_ids': list(self.gpu_ids) if self.mode == 'cuda' else [],
            'gpu_min_tree_points': int(self.gpu_min_tree_points),
            'gpu_query_chunk_size': int(self.gpu_query_chunk_size),
            'backend_tag': self.tag
        })
        return result

SPATIAL_SEARCH = SpatialSearchManager()

class BackendKDTree:
    """SciPy-like exact k-NN tree with CPU or replicated multi-GPU CUDA backend."""
    def __init__(self, data, force_cpu=False):
        self.data = np.asarray(data)
        if self.data.ndim != 2:
            raise ValueError(f'KDTree data must be 2-D, got {self.data.shape}')
        self.n = int(len(self.data))
        self.m = int(self.data.shape[1]) if self.data.ndim == 2 else 0
        self.use_cuda = (not force_cpu) and SPATIAL_SEARCH.should_use_cuda(self.n)
        self.cpu_tree = None
        self.gpu_data = {}
        self.gpu_trees = {}
        started = time.perf_counter()
        if self.use_cuda:
            cp = SPATIAL_SEARCH.cp
            for gpu_id in SPATIAL_SEARCH.gpu_ids:
                with cp.cuda.Device(gpu_id):
                    gpu_data = cp.asarray(self.data, dtype=cp.float64)
                    gpu_tree = SPATIAL_SEARCH.CuKDTree(gpu_data)
                    cp.cuda.get_current_stream().synchronize()
                    self.gpu_data[gpu_id] = gpu_data
                    self.gpu_trees[gpu_id] = gpu_tree
            elapsed = time.perf_counter() - started
            SPATIAL_SEARCH.add_stats(tree_build_calls=1, gpu_tree_build_calls=1, reference_points_built=self.n, tree_build_s=elapsed, gpu_tree_build_s=elapsed, gpu_replica_builds=len(SPATIAL_SEARCH.gpu_ids))
        else:
            self.cpu_tree = cKDTree(self.data)
            elapsed = time.perf_counter() - started
            SPATIAL_SEARCH.add_stats(tree_build_calls=1, cpu_tree_build_calls=1, reference_points_built=self.n, tree_build_s=elapsed, cpu_tree_build_s=elapsed)

    def _gpu_query_group(self, gpu_id, chunks, k):
        cp = SPATIAL_SEARCH.cp
        results = []
        with cp.cuda.Device(gpu_id):
            tree = self.gpu_trees[gpu_id]
            for order, begin, end, query_chunk in chunks:
                q = cp.asarray(query_chunk, dtype=cp.float64)
                distances, indices = tree.query(q, k=k)
                cp.cuda.get_current_stream().synchronize()
                results.append((order, begin, end, cp.asnumpy(distances), cp.asnumpy(indices)))
                del q, distances, indices
        return results

    def query(self, x, k=1, workers=None, **kwargs):
        query = np.asarray(x)
        if not self.use_cuda:
            started = time.perf_counter()
            worker_count = CPU_WORKERS if workers is None or int(workers) == -1 else int(workers)
            result = self.cpu_tree.query(query, k=k, workers=worker_count, **kwargs)
            elapsed = time.perf_counter() - started
            qn = 1 if query.ndim == 1 else len(query)
            SPATIAL_SEARCH.add_stats(query_calls=1, cpu_query_calls=1, query_points=qn, query_s=elapsed, cpu_query_s=elapsed)
            return result
        if kwargs:
            unsupported = ', '.join(sorted(kwargs))
            raise TypeError(f'CUDA KDTree wrapper currently supports exact query(x, k=...) only; unsupported options: {unsupported}')
        if query.ndim == 1:
            query2 = query[None, :]
            squeeze_query = True
        else:
            query2 = query
            squeeze_query = False
        n_query = len(query2)
        started = time.perf_counter()
        if n_query == 0:
            if int(k) == 1:
                distances = np.empty((0,), dtype=np.float64); indices = np.empty((0,), dtype=np.int64)
            else:
                distances = np.empty((0, int(k)), dtype=np.float64); indices = np.empty((0, int(k)), dtype=np.int64)
        else:
            chunk_size = SPATIAL_SEARCH.gpu_query_chunk_size
            chunks = []
            order = 0
            for begin in range(0, n_query, chunk_size):
                end = min(begin + chunk_size, n_query)
                chunks.append((order, begin, end, query2[begin:end]))
                order += 1
            groups = {gpu_id: [] for gpu_id in SPATIAL_SEARCH.gpu_ids}
            for i, chunk in enumerate(chunks):
                groups[SPATIAL_SEARCH.gpu_ids[i % len(SPATIAL_SEARCH.gpu_ids)]].append(chunk)
            active = [(gpu_id, group) for gpu_id, group in groups.items() if group]
            if len(active) == 1:
                collected = self._gpu_query_group(active[0][0], active[0][1], int(k))
            else:
                collected = []
                with ThreadPoolExecutor(max_workers=len(active)) as pool:
                    futures = [pool.submit(self._gpu_query_group, gpu_id, group, int(k)) for gpu_id, group in active]
                    for future in futures:
                        collected.extend(future.result())
            collected.sort(key=lambda item: item[0])
            first_d = collected[0][3]
            first_i = collected[0][4]
            if first_d.ndim == 1:
                distances = np.empty((n_query,), dtype=first_d.dtype)
                indices = np.empty((n_query,), dtype=first_i.dtype)
            else:
                distances = np.empty((n_query,) + first_d.shape[1:], dtype=first_d.dtype)
                indices = np.empty((n_query,) + first_i.shape[1:], dtype=first_i.dtype)
            for _, begin, end, d, idx in collected:
                distances[begin:end] = d
                indices[begin:end] = idx
        elapsed = time.perf_counter() - started
        SPATIAL_SEARCH.add_stats(query_calls=1, gpu_query_calls=1, query_points=n_query, query_s=elapsed, gpu_query_s=elapsed)
        if squeeze_query:
            distances = distances[0]
            indices = indices[0]
        return distances, indices

def spatial_tree(data, force_cpu=False):
    return BackendKDTree(data, force_cpu=force_cpu)

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
            final_expected = os.path.join(output_root, 'recon_related', job['case'], 'static_recon_labels.npz')
            try:
                with open(completion_marker, 'r') as f:
                    marker_data = json.load(f)
                existing_complete = bool(marker_data.get('static_filter_completed', False)) and bool(marker_data.get('densification_completed', not args.run_densification)) and os.path.isfile(final_expected)
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
            reference.positive_tree = spatial_tree(positive)
        if reference.negative_parts:
            negative = np.concatenate(reference.negative_parts, axis=0)
            negative = subsample_rows(negative, args.max_object_reference_points, rng)
            reference.negative_tree = spatial_tree(negative)

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

    return (track_references, static_xyz, static_semantic, spatial_tree(static_xyz))

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
    tree = spatial_tree(class_xyz)
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

    global_tree = spatial_tree(trusted_xyz)
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
        class_tree = spatial_tree(trusted_xyz[rmask])
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
        xy_tree = spatial_tree(reference_xyz[:, :2])
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

    tree = spatial_tree(reference)
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
    tree = spatial_tree(points_xy); accepted = []; accepted_count = 0
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


def fit_generic_candidates(cand_xy,family_points,family_tree,k,rmse_max,support_radius,min_sep,boundary_tree_xy=None,boundary_radius=0.12,max_abs_grade_percent=15.0,max_z_span_m=0.35,max_height_residual_m=0.08,geometry_guard=True,chunk=50000):
    """Fit generic ground support from ORIGINAL-derived same-family samples only.

    Generic coverage is constrained to a 2.5-D ground surface. A candidate is
    rejected when the fitted patch is too steep, the local vertical spread is
    too large, or the predicted Z is inconsistent with the local robust ground
    height. Generated support is never used as input to this fit.
    """
    empty=(np.empty((0,3),np.float32),np.empty((0,3),np.float32),np.empty((0,),np.float32),np.empty((0,),np.float32))
    stats={"input_candidates":int(len(cand_xy)),"rejected_insufficient_support":0,"rejected_rmse":0,"rejected_grade":0,"rejected_z_span":0,"rejected_height":0,"rejected_boundary":0,"accepted":0}
    if len(cand_xy)==0: return (*empty,stats)
    xyz_out,normals_out,rmse_out,nearest_out=[],[],[],[]
    kk=min(k,len(family_points)); max_grade=float(max_abs_grade_percent); max_slope=max_grade/100.0
    for begin in range(0,len(cand_xy),chunk):
        qxy=cand_xy[begin:begin+chunk]
        dxy,idx=family_tree.query(qxy,k=kk,workers=-1)
        if kk==1: dxy=dxy[:,None]; idx=idx[:,None]
        nearest=dxy[:,0]; enough=(nearest>=min_sep)&(dxy[:,-1]<=support_radius)
        stats["rejected_insufficient_support"]+=int((~enough).sum())
        if not np.any(enough): continue

        qxy2,idx2=qxy[enough],idx[enough]; nearest2=nearest[enough]
        neigh=family_points[idx2]; mean=neigh.mean(axis=1); centered=neigh-mean[:,None,:]
        cov=np.einsum("nki,nkj->nij",centered,centered)/max(1,kk)
        eigvals,eigvecs=np.linalg.eigh(cov); normal=eigvecs[:,:,0]; normal[normal[:,2]<0]*=-1.0
        rmse=np.sqrt(np.maximum(eigvals[:,0],0.0)); nz=np.clip(normal[:,2],1e-9,None)
        grade=100.0*np.linalg.norm(normal[:,:2],axis=1)/nz
        zlo=np.percentile(neigh[:,:,2],5,axis=1); zhi=np.percentile(neigh[:,:,2],95,axis=1); zspan=zhi-zlo

        valid=np.ones(len(qxy2),dtype=bool)
        bad=valid&(rmse>rmse_max); stats["rejected_rmse"]+=int(bad.sum()); valid&=~bad
        if geometry_guard:
            bad=valid&(grade>max_grade); stats["rejected_grade"]+=int(bad.sum()); valid&=~bad
            bad=valid&(zspan>max_z_span_m); stats["rejected_z_span"]+=int(bad.sum()); valid&=~bad
        if not np.any(valid): continue

        qxy3=qxy2[valid]; m=mean[valid]; n=normal[valid]; rmse3=rmse[valid]; nearest3=nearest2[valid]; neigh3=neigh[valid]
        z=m[:,2]-(n[:,0]*(qxy3[:,0]-m[:,0])+n[:,1]*(qxy3[:,1]-m[:,1]))/np.clip(n[:,2],1e-9,None)

        if geometry_guard:
            med_z=np.median(neigh3[:,:,2],axis=1); med_xy=np.median(neigh3[:,:,:2],axis=1)
            horizontal_offset=np.linalg.norm(qxy3-med_xy,axis=1)
            allowed_dz=float(max_height_residual_m)+max_slope*horizontal_offset
            height_residual=np.abs(z-med_z); hkeep=height_residual<=allowed_dz
            stats["rejected_height"]+=int((~hkeep).sum())
            if not np.any(hkeep): continue
            qxy3,m,n,z,rmse3,nearest3=qxy3[hkeep],m[hkeep],n[hkeep],z[hkeep],rmse3[hkeep],nearest3[hkeep]

        xyz3=np.column_stack([qxy3,z]).astype(np.float32); keep=np.ones(len(xyz3),dtype=bool)
        if boundary_tree_xy is not None:
            bd,_=boundary_tree_xy.query(xyz3[:,:2],k=1,workers=-1); keep&=bd>=boundary_radius
            stats["rejected_boundary"]+=int((~keep).sum())
        if np.any(keep):
            xyz_out.append(xyz3[keep]); normals_out.append(n[keep].astype(np.float32)); rmse_out.append(rmse3[keep].astype(np.float32)); nearest_out.append(nearest3[keep].astype(np.float32)); stats["accepted"]+=int(keep.sum())
    if not xyz_out: return (*empty,stats)
    return np.concatenate(xyz_out),np.concatenate(normals_out),np.concatenate(rmse_out),np.concatenate(nearest_out),stats


def filter_ring_support(xyz, family_tree_xy, min_sep, boundary_tree_xy=None, boundary_radius=0.12):
    if len(xyz) == 0:
        return np.empty((0,), dtype=bool), np.empty((0,), np.float32)
    nearest, _ = family_tree_xy.query(xyz[:, :2], k=1, workers=-1)
    keep = nearest >= min_sep
    if boundary_tree_xy is not None:
        bd, _ = boundary_tree_xy.query(xyz[:, :2], k=1, workers=-1); keep &= bd >= boundary_radius
    return keep, nearest.astype(np.float32)


def load_dynamic_box_cache(path):
    path=Path(path)
    if not path.is_file(): return {}
    with path.open() as f: tracks=json.load(f).get("tracks",{}).values()
    cache={}
    for track in tracks:
        if not bool(track.get("is_dynamic",False)): continue
        for frame_text,record in track.get("frames",{}).items():
            pose=np.asarray(record["box_pose_world"],np.float64).reshape(4,4); dims=np.asarray(record["box_vehicle"][3:6],np.float64)
            cache.setdefault(int(frame_text),[]).append((pose[:3,:3],pose[:3,3],dims[:2]/2.0,int(track.get("lidargs_object_id",-1)),int(track.get("semantic_id",0))))
    return cache


def reject_generic_support_in_dynamic_footprints_source_frame(xyz,source_type,source_index,observation_frames,box_cache,xy_margin=0.15):
    keep=np.ones(len(xyz),dtype=bool); generic=np.flatnonzero(np.asarray(source_type)==4)
    if not len(generic) or not box_cache: return keep,{}
    frames=np.asarray(observation_frames)[np.asarray(source_index,dtype=np.int64)]; rejected_by_frame={}
    for frame in np.unique(frames[generic]):
        boxes=box_cache.get(int(frame),())
        if not boxes: continue
        idx=generic[frames[generic]==frame]; pts=np.asarray(xyz[idx],np.float64); rejected=np.zeros(len(idx),dtype=bool)
        for rotation,translation,half_xy,_,_ in boxes:
            local=(pts-translation[None,:])@rotation; rejected|=(np.abs(local[:,0])<=half_xy[0]+xy_margin)&(np.abs(local[:,1])<=half_xy[1]+xy_margin)
        new=rejected&keep[idx]
        if np.any(new): keep[idx[new]]=False; rejected_by_frame[int(frame)]=int(new.sum())
    return keep,rejected_by_frame


def reject_generic_support_in_dynamic_footprints_swept(xyz,source_type,box_cache,xy_margin=0.15):
    """Reject generic support inside ANY dynamic-object XY footprint in the scene.

    This only removes generated source_type=4 support. Original measured ground
    remains untouched. If a moving object occupied an XY location in any frame, generic coverage is
    not allowed to invent ground there.
    """
    keep=np.ones(len(xyz),dtype=bool); generic=np.flatnonzero(np.asarray(source_type)==4)
    if not len(generic) or not box_cache: return keep,{}
    gxy=np.asarray(xyz[generic,:2],np.float64); tree=cKDTree(gxy); rejected_by_frame={}
    for frame in sorted(box_cache):
        newly=0
        for rotation,translation,half_xy,_,_ in box_cache[frame]:
            hx=float(half_xy[0])+xy_margin; hy=float(half_xy[1])+xy_margin; radius=math.hypot(hx,hy)
            local_ids=tree.query_ball_point(np.asarray(translation[:2],np.float64),r=radius)
            if not local_ids: continue
            local_ids=np.asarray(local_ids,dtype=np.int64); idx=generic[local_ids]; active=keep[idx]
            if not np.any(active): continue
            idx=idx[active]; pts=np.asarray(xyz[idx],np.float64); local=(pts-translation[None,:])@rotation
            inside=(np.abs(local[:,0])<=hx)&(np.abs(local[:,1])<=hy)
            if np.any(inside): keep[idx[inside]]=False; newly+=int(inside.sum())
        if newly: rejected_by_frame[int(frame)]=newly
    return keep,rejected_by_frame


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
            tree = spatial_tree(original_xyz[source_indices])
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
            tree = spatial_tree(original_xyz[source_indices])
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


def add_densification_arguments(parser):
    """Add integrated densification controls. Defaults match the current semantic-MLS baseline."""
    parser.add_argument('--run-densification', action=argparse.BooleanOptionalAction, default=True, help='Run ground-surface densification after the strict static filter and write the final canonical static_recon_labels.npz.')
    parser.add_argument('--save-densified-pcd', action=argparse.BooleanOptionalAction, default=True, help='Also save the densified x/y/z/semantic/intensity PCD under preprocessing_residues/densification.')
    parser.add_argument('--analysis_voxel', type=float, default=0.05)
    parser.add_argument('--fill_spacing', type=float, default=0.03, help='Fixed spacing used in fixed mode and by generic coverage.')
    parser.add_argument('--fill_spacing_mode', choices=['fixed','local_ring_density','along_ring_density'], default='local_ring_density', help='fixed; generic XY-NN adaptive; or density measured along the two actual rings bracketing each gap.')
    parser.add_argument('--adaptive_density_voxel', type=float, default=0.01)
    parser.add_argument('--adaptive_density_neighbors', type=int, default=3)
    parser.add_argument('--adaptive_query_neighbors', type=int, default=16)
    parser.add_argument('--adaptive_fill_scale', type=float, default=1.0)
    parser.add_argument('--adaptive_fill_min', type=float, default=0.01)
    parser.add_argument('--adaptive_fill_max', type=float, default=0.08)
    parser.add_argument('--ring_density_duplicate_threshold', type=float, default=0.003)
    parser.add_argument('--ring_density_min_unique_samples', type=int, default=3)
    parser.add_argument('--radial_spacing_scale', type=float, default=1.0)
    parser.add_argument('--deduplicate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dedup_spacing', type=float, default=0.005, help='Generated-point dedup cell size. 0=derive automatically.')
    parser.add_argument('--full_resolution', action='store_true')
    parser.add_argument('--disable_safety_filters', action='store_true', help='Disable ring-anchor outlier, grade, abnormal-gap and semantic-boundary rejection. The generic 2.5-D guard remains active unless explicitly disabled.')
    parser.add_argument('--ring_mode', choices=['auto','on','off'], default='auto')
    parser.add_argument('--ring_origin_spread_max', type=float, default=1.0)
    parser.add_argument('--ring_sector_deg', type=float, default=0.5)
    parser.add_argument('--ring_split_gap', type=float, default=0.16)
    parser.add_argument('--ring_profile_bin_m', type=float, default=5.0)
    parser.add_argument('--ring_max_gap_factor', type=float, default=2.75)
    parser.add_argument('--ring_min_points_per_run', type=int, default=8)
    parser.add_argument('--ring_min_sector_points', type=int, default=24)
    parser.add_argument('--ring_max_abs_grade_percent', type=float, default=12.0)
    parser.add_argument('--ring_anchor_loo_floor_m', type=float, default=0.03)
    parser.add_argument('--ring_anchor_loo_sigma', type=float, default=6.0)
    parser.add_argument('--ring_neighbor_half_window', type=int, default=1)
    parser.add_argument('--ring_neighbor_z_tol_m', type=float, default=0.05)
    parser.add_argument('--ring_neighbor_min_matches', type=int, default=1)
    parser.add_argument('--generic_coverage', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--coverage_radius', type=float, default=0.55)
    parser.add_argument('--coverage_knn', type=int, default=12)
    parser.add_argument('--coverage_min_quadrants', type=int, default=3)
    parser.add_argument('--generic_ground_geometry_guard', action=argparse.BooleanOptionalAction, default=True, help='Keep generic ground as a 2.5-D height field and reject steep, vertically spread, or height-inconsistent fits.')
    parser.add_argument('--generic_max_abs_grade_percent', type=float, default=15.0)
    parser.add_argument('--generic_max_z_span_m', type=float, default=0.35)
    parser.add_argument('--generic_max_height_residual_m', type=float, default=0.08)
    parser.add_argument('--reject_generic_dynamic_footprints', action=argparse.BooleanOptionalAction, default=True, help='Reject generated generic ground inside dynamic-object XY footprints.')
    parser.add_argument('--dynamic_footprint_mode', choices=['swept','source_frame'], default='source_frame', help='swept rejects any XY occupied by a dynamic box in any frame; source_frame checks the inherited observation frame.')
    parser.add_argument('--tracks_path', default=None, help='Optional stage_a_tracks.json override. Default: <output-root>/temp/<case>/stage_a_tracks.json.')
    parser.add_argument('--dynamic_footprint_xy_margin', type=float, default=0.15)
    parser.add_argument('--plane_knn', type=int, default=16)
    parser.add_argument('--plane_rmse_max', type=float, default=0.06)
    parser.add_argument('--plane_support_radius', type=float, default=1.25)
    parser.add_argument('--boundary_radius', type=float, default=0.12)
    parser.add_argument('--min_separation', type=float, default=0.01)
    parser.add_argument('--max_candidates_per_family', type=int, default=0)
    parser.add_argument('--max_support', type=int, default=0)
    parser.add_argument('--npz_compression', choices=['compressed','stored'], default='stored', help='NPZ storage mode. stored matches the current semantic-MLS baseline and is faster for very large clouds.')
    parser.add_argument('--pcd_chunk_size', type=int, default=1000000)


def validate_densification_arguments(args):
    if args.fill_spacing <= 0: raise ValueError('--fill_spacing must be > 0')
    if args.adaptive_density_voxel < 0: raise ValueError('--adaptive_density_voxel must be >= 0')
    if args.adaptive_fill_min <= 0 or args.adaptive_fill_max <= 0 or args.adaptive_fill_min > args.adaptive_fill_max: raise ValueError('adaptive fill limits must satisfy 0 < min <= max')
    if args.adaptive_fill_scale <= 0: raise ValueError('--adaptive_fill_scale must be > 0')
    if args.ring_density_duplicate_threshold < 0: raise ValueError('--ring_density_duplicate_threshold must be >= 0')
    if args.ring_density_min_unique_samples < 2: raise ValueError('--ring_density_min_unique_samples must be >= 2')
    if args.radial_spacing_scale <= 0: raise ValueError('--radial_spacing_scale must be > 0')
    if args.generic_max_abs_grade_percent <= 0: raise ValueError('--generic_max_abs_grade_percent must be > 0')
    if args.generic_max_z_span_m <= 0: raise ValueError('--generic_max_z_span_m must be > 0')
    if args.generic_max_height_residual_m < 0: raise ValueError('--generic_max_height_residual_m must be >= 0')
    if args.dynamic_footprint_xy_margin < 0: raise ValueError('--dynamic_footprint_xy_margin must be >= 0')
    if args.max_candidates_per_family < 0 or args.max_support < 0: raise ValueError('candidate/support caps must be >= 0; use 0 for unlimited')

def run_integrated_densification(root, case, strict_path, static_dir, densification_dir, args):
    """Run the supplied densification algorithm and emit the canonical final static NPZ."""
    validate_densification_arguments(args)
    final_npz=os.path.join(static_dir, "static_recon_labels.npz")
    densified_pcd=os.path.join(densification_dir, "static_recon_semantic_intensity_densified.pcd") if args.save_densified_pcd else None
    timing_json=os.path.join(densification_dir, f"densification_timing_{SPATIAL_SEARCH.tag}.json")
    tracks_path=Path(args.tracks_path) if args.tracks_path else Path(root)/"temp"/case/"stage_a_tracks.json"
    if args.full_resolution:
        # Density/thinning filters are disabled. The original ground cloud is used
        # directly for analysis and local-density estimation.
        args.analysis_voxel = 0.0
        args.adaptive_density_voxel = 0.0
        args.min_separation = 0.0
        args.deduplicate = False
        args.max_candidates_per_family = 0
        args.max_support = 0

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
        if args.generic_ground_geometry_guard: print("[scene support] generic 2.5-D ground geometry guard REMAINS ACTIVE; disable explicitly with --no-generic_ground_geometry_guard only for ablation")
    stage_started = time.perf_counter()
    centers, csem = voxel_centroids(xyz, sem, args.analysis_voxel); fam = semantic_family(csem)
    timing["analysis_representation_s"] = time.perf_counter() - stage_started
    analysis_mode = "FULL RAW POINT CLOUD (no analysis voxelization)" if args.analysis_voxel <= 0 else f"{args.analysis_voxel:g} m voxel centroids"
    print(f"[scene support] analysis representation: {analysis_mode}")
    print(f"[scene support] analysis points: {len(centers):,}; ground={int((fam>0).sum()):,}; curb={int((csem==17).sum()):,}")

    origins = load_sensor_origins(root, case); tstats = trajectory_stats(origins)
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
        tree_xy = spatial_tree(pts[:, :2])
        other_ground_xyz = centers[(fam > 0) & (fam != family_id)]; curb_xyz = centers[csem == CURB_SEMANTIC]
        boundary_xyz = np.concatenate([curb_xyz, other_ground_xyz], axis=0) if len(curb_xyz)+len(other_ground_xyz) else np.empty((0,3),np.float32)
        boundary_tree_xy = spatial_tree(boundary_xyz[:, :2]) if len(boundary_xyz) else None
        if args.disable_safety_filters:
            boundary_tree_xy = None
        family_stage_timing["tree_build_s"] = time.perf_counter() - stage_started

        fam_xyz=[]; fam_sem=[]; fam_source=[]; fam_normal=[]; fam_rmse=[]; fam_nearest=[]
        fam_gap=[]; fam_expected=[]; fam_ratio=[]; fam_r0=[]; fam_z0=[]; fam_r1=[]; fam_z1=[]; fam_t=[]; fam_grade=[]; fam_loo=[]; fam_neigh=[]; fam_spacing=[]; fam_radial_spacing=[]
        ring_meta={"active":False}; ring_count=0; generic_count=0; generic_guard_meta={}
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

            stage_started=time.perf_counter()
            gxyz,gn,grmse,gnear,generic_guard_meta=fit_generic_candidates(
                gxy,pts,tree_xy,args.plane_knn,plane_rmse_max,plane_support_radius,generic_min_separation,
                generic_boundary_tree,args.boundary_radius,args.generic_max_abs_grade_percent,args.generic_max_z_span_m,
                args.generic_max_height_residual_m,args.generic_ground_geometry_guard)
            family_stage_timing["generic_plane_fit_s"]=time.perf_counter()-stage_started
            generic_count=len(gxyz)
            print(f"[generic guard] family={family_id} candidates={generic_guard_meta.get('input_candidates',0):,} accepted={generic_guard_meta.get('accepted',0):,} reject_support={generic_guard_meta.get('rejected_insufficient_support',0):,} reject_rmse={generic_guard_meta.get('rejected_rmse',0):,} reject_grade={generic_guard_meta.get('rejected_grade',0):,} reject_zspan={generic_guard_meta.get('rejected_z_span',0):,} reject_height={generic_guard_meta.get('rejected_height',0):,} reject_boundary={generic_guard_meta.get('rejected_boundary',0):,}")
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
        family_meta[str(family_id)]={"analysis_points":int(len(pts)),"raw_points":int(len(raw_pts)),"support":support_count,"ring_support":ring_final,"generic_support":generic_final,"ring":ring_meta,"generic_guard":generic_guard_meta,"timing_s":family_stage_timing}
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

    support_source_index=None; dynamic_guard_rejected=0; dynamic_guard_by_frame={}
    if args.reject_generic_dynamic_footprints and np.any(st==4):
        stage_started=time.perf_counter(); box_cache=load_dynamic_box_cache(tracks_path)
        if not box_cache: raise FileNotFoundError(f"No dynamic boxes loaded from {tracks_path}")
        if args.dynamic_footprint_mode=="swept":
            keep,dynamic_guard_by_frame=reject_generic_support_in_dynamic_footprints_swept(sx,st,box_cache,args.dynamic_footprint_xy_margin)
        else:
            with np.load(strict_path,allow_pickle=False) as d:
                original_xyz_for_guard=np.asarray(d["xyz"],dtype=np.float32); original_sem_for_guard=np.asarray(d["semantic_id"],dtype=np.int16)
                if "observation_frame_index" not in d.files: raise KeyError("--dynamic_footprint_mode source_frame requires observation_frame_index in strict NPZ")
                original_frame_for_guard=np.asarray(d["observation_frame_index"],dtype=np.int64)
            support_source_index=assign_support_source_indices(original_xyz_for_guard,original_sem_for_guard,sx.astype(np.float32),ss.astype(np.int16))
            keep,dynamic_guard_by_frame=reject_generic_support_in_dynamic_footprints_source_frame(sx,st,support_source_index,original_frame_for_guard,box_cache,args.dynamic_footprint_xy_margin)
        dynamic_guard_rejected=int((~keep).sum())
        if dynamic_guard_rejected:
            sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing=[a[keep] for a in (sx,ss,sf,st,sn,srmse,snear,sg,se,sgr,sr0,sz0,sr1,sz1,sit,sgrade,sloo,sneigh,sspacing,sradial_spacing)]
            if support_source_index is not None: support_source_index=np.asarray(support_source_index,dtype=np.int64)[keep]
        print(f"[dynamic footprint guard] mode={args.dynamic_footprint_mode} rejected={dynamic_guard_rejected:,} generic support points; remaining={len(sx):,}; first-hit frames={dynamic_guard_by_frame}")
        timing["dynamic_footprint_guard_s"]=time.perf_counter()-stage_started

    out=Path(densification_dir) / "scene_ground_support.npz"; out.parent.mkdir(parents=True,exist_ok=True)
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
    meta={"caseid":case,"support_npz":str(out),"num_support":int(len(sx)),"num_ring_support":int((st==3).sum()),"num_generic_support":int((st==4).sum()),
          "method":"Robust same-sector measured ring interpolation; isolated bad anchors rejected; neighbouring sectors validate but never overwrite Z",
          "analysis_voxel_m":args.analysis_voxel,"fill_spacing_m":args.fill_spacing,"fill_spacing_mode":args.fill_spacing_mode,
          "adaptive_density_voxel_m":args.adaptive_density_voxel,"adaptive_density_neighbors":args.adaptive_density_neighbors,
          "adaptive_query_neighbors":args.adaptive_query_neighbors,"adaptive_fill_scale":args.adaptive_fill_scale,
          "adaptive_fill_min_m":args.adaptive_fill_min,"adaptive_fill_max_m":args.adaptive_fill_max,
          "ring_density_duplicate_threshold_m":args.ring_density_duplicate_threshold,
          "ring_density_min_unique_samples":args.ring_density_min_unique_samples,
          "radial_spacing_scale":args.radial_spacing_scale,
          "deduplicate":bool(args.deduplicate),"dedup_spacing_m":args.dedup_spacing,
          "generic_ground_geometry_guard":bool(args.generic_ground_geometry_guard),"generic_max_abs_grade_percent":args.generic_max_abs_grade_percent,"generic_max_z_span_m":args.generic_max_z_span_m,"generic_max_height_residual_m":args.generic_max_height_residual_m,
          "reject_generic_dynamic_footprints":bool(args.reject_generic_dynamic_footprints),"dynamic_footprint_mode":args.dynamic_footprint_mode,"dynamic_footprint_xy_margin_m":args.dynamic_footprint_xy_margin,"dynamic_footprint_rejected":int(dynamic_guard_rejected),"dynamic_footprint_rejected_by_frame":dynamic_guard_by_frame,"tracks_path":str(tracks_path),
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
    if densified_pcd is not None or final_npz is not None:
        print("[densified export] mapping generated support to original metadata...")

        with np.load(strict_path, allow_pickle=False) as d:
            original_xyz_for_metadata = np.asarray(d["xyz"], dtype=np.float32)
            original_sem_for_metadata = np.asarray(d["semantic_id"], dtype=np.int16)

        stage_started = time.perf_counter()
        if support_source_index is None:
            support_source_index = assign_support_source_indices(original_xyz_for_metadata,original_sem_for_metadata,sx.astype(np.float32),ss.astype(np.int16))
        timing["metadata_source_mapping_s"] = time.perf_counter() - stage_started

        if densified_pcd is not None:
            stage_started = time.perf_counter()
            export_densified_pcd(
                strict_path,
                sx.astype(np.float32),
                ss.astype(np.int16),
                support_source_index,
                densified_pcd,
                chunk_size=args.pcd_chunk_size,
            )
            timing["write_densified_pcd_s"] = time.perf_counter() - stage_started

        if final_npz is not None:
            stage_started = time.perf_counter()
            export_densified_npz(
                strict_path,
                sx.astype(np.float32),
                ss.astype(np.int16),
                st.astype(np.int8),
                sf.astype(np.int8),
                final_npz,
                source_index=support_source_index,
                compression=args.npz_compression,
            )
            timing["write_densified_npz_s"] = time.perf_counter() - stage_started

    timing["family_total_s"] = float(sum(family_timings.values()))
    timing["families_s"] = family_timings
    timing["total_s"] = time.perf_counter() - total_started
    print("[timing] " + json.dumps(timing, sort_keys=True))
    if timing_json is not None:
        timing_path = Path(timing_json)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w") as stream:
            json.dump(timing, stream, indent=2, sort_keys=True)
    return {"final_npz":final_npz,"support_npz":str(out),"support_meta":str(out)+".json","densified_pcd":densified_pcd,"timing_json":timing_json,"num_support":int(len(sx))}

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
    parser.add_argument('--compute-backend', choices=['cpu','cuda'], default='cpu', help='Spatial-search backend. cpu keeps SciPy cKDTree; cuda sends large exact k-NN trees/queries to cupyx.scipy.spatial.KDTree.')
    parser.add_argument('--gpu-ids', default='0', help='Comma-separated CUDA device IDs used for replicated KD-trees and query-chunk distribution, e.g. 0 or 0,1,2.')
    parser.add_argument('--gpu-min-tree-points', type=int, default=50000, help='In CUDA mode, trees smaller than this remain on CPU to avoid GPU setup overhead.')
    parser.add_argument('--gpu-query-chunk-size', type=int, default=200000, help='Maximum query rows transferred to one GPU at a time.')
    parser.add_argument('--continue-on-error', action=argparse.BooleanOptionalAction, default=False, help='Batch mode only: continue processing later scenes after a missing TFRecord or failed scene.')
    parser.add_argument('--skip-existing-scenes', action=argparse.BooleanOptionalAction, default=False, help='Batch mode only: skip a case only when its final PREPROCESSING_COMPLETE.json marker exists.')
    parser.add_argument('--save-combined-data-labeled', action=argparse.BooleanOptionalAction, default=True, help='Also save the convenience N x 7 float64 data_labeled matrix. Disable for large batches because it duplicates the named arrays.')
    parser.add_argument('--debug-foreground-source-frame',type=int,action='append',default=[])

    # Integrated strict static-cloud filtering. Raw accumulation is preserved under preprocessing_residues/raw_accumulation.
    parser.add_argument('--run-static-filter', action=argparse.BooleanOptionalAction, default=True, help='After raw static accumulation, create a strict filtered static cloud. Raw accumulation is preserved under preprocessing_residues/raw_accumulation.')
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
    add_densification_arguments(parser)

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
    SPATIAL_SEARCH.configure(args.compute_backend, args.gpu_ids, args.gpu_min_tree_points, args.gpu_query_chunk_size)
    if args.run_densification and not args.run_static_filter:
        parser.error('--run-densification requires --run-static-filter because densification consumes the strict filtered cloud.')
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
    print(f'Compute backend          : {SPATIAL_SEARCH.mode}')
    if SPATIAL_SEARCH.mode == 'cuda':
        print(f'CUDA GPUs                : {SPATIAL_SEARCH.gpu_ids}')
        print(f'GPU tree threshold       : {SPATIAL_SEARCH.gpu_min_tree_points:,} points')
        print(f'GPU query chunk          : {SPATIAL_SEARCH.gpu_query_chunk_size:,} rows')

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

    pipeline_started = time.perf_counter()
    root = os.path.abspath(args.output_root)
    meta_dir = os.path.join(root, 'meta_infos')
    pcd_dir = os.path.join(root, 'pcds_new', case)
    calib_dir = os.path.join(root, 'laser_calibrations', case, 'laser_calibrations')
    beam_dir = os.path.join(root, 'temp', case, 'beam_inclinations')
    static_dir = os.path.join(root, 'recon_related', case)
    residue_dir = os.path.join(static_dir, 'preprocessing_residues')
    raw_static_dir = os.path.join(residue_dir, 'raw_accumulation')
    static_filter_dir = os.path.join(residue_dir, 'static_filter')
    densification_dir = os.path.join(residue_dir, 'densification')
    timing_dir = os.path.join(residue_dir, 'timing')
    dynamic_objects_root = os.path.join(root, 'temp', case, 'occ', 'preproc', 'dynamic', 'objects')
    temp_case_dir = os.path.join(root, 'temp', case)

    for directory in [meta_dir, pcd_dir, calib_dir, beam_dir, static_dir, residue_dir, raw_static_dir, static_filter_dir, densification_dir, timing_dir, dynamic_objects_root, temp_case_dir]:
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
    static_path = os.path.join(raw_static_dir, 'static_recon_voxels_raw.pcd')

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
    static_labeled_path = os.path.join(raw_static_dir, 'static_recon_labels_raw.npz')
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

    densification_report = None
    if args.run_densification:
        if not args.run_static_filter or static_filter_report is None:
            raise RuntimeError('Densification requires --run-static-filter because it consumes the strict filtered cloud.')
        strict_path = static_filter_report['outputs']['strict_npz']
        densification_report = run_integrated_densification(root, case, strict_path, static_dir, densification_dir, args)
    else:
        final_npz = os.path.join(static_dir, 'static_recon_labels.npz')
        if args.run_static_filter and static_filter_report is not None:
            import shutil
            shutil.copy2(static_filter_report['outputs']['strict_npz'], final_npz)
        else:
            import shutil
            shutil.copy2(static_labeled_path, final_npz)

    total_wall_s = time.perf_counter() - pipeline_started
    timing_summary = {'case': case, 'backend': SPATIAL_SEARCH.mode, 'backend_tag': SPATIAL_SEARCH.tag, 'gpu_ids': list(SPATIAL_SEARCH.gpu_ids) if SPATIAL_SEARCH.mode == 'cuda' else [], 'cpu_workers': int(CPU_WORKERS), 'total_wall_s': float(total_wall_s), 'spatial_search': SPATIAL_SEARCH.snapshot(), 'densification_timing_json': densification_report.get('timing_json') if densification_report else None}
    timing_summary_path = os.path.join(timing_dir, f'preprocessing_timing_{SPATIAL_SEARCH.tag}.json')
    with open(timing_summary_path, 'w') as f:
        json.dump(timing_summary, f, indent=2, sort_keys=True)
    completion_marker_path = os.path.join(temp_case_dir, 'PREPROCESSING_COMPLETE.json')
    marker = {
        'case': case, 'status': 'completed', 'output_frames': int(len(output_source_indices)),
        'compute_backend': SPATIAL_SEARCH.mode, 'backend_tag': SPATIAL_SEARCH.tag, 'gpu_ids': list(SPATIAL_SEARCH.gpu_ids) if SPATIAL_SEARCH.mode == 'cuda' else [], 'timing_summary': timing_summary_path,
        'source_frame_start': int(output_source_indices[0]), 'source_frame_end': int(output_source_indices[-1]),
        'save_combined_data_labeled': bool(args.save_combined_data_labeled), 'cpu_workers': int(CPU_WORKERS),
        'raw_static_cloud_preserved': True, 'raw_static_npz': static_labeled_path, 'raw_static_pcd': static_path,
        'residue_dir': residue_dir, 'static_filter_requested': bool(args.run_static_filter), 'static_filter_completed': bool(static_filter_report is not None),
        'static_filter_output_dir': static_filter_dir if args.run_static_filter else None, 'static_filter_ground_max_dz_m': float(args.static_filter_ground_max_dz) if args.run_static_filter else None,
        'densification_requested': bool(args.run_densification), 'densification_completed': bool(densification_report is not None),
        'densification_output_dir': densification_dir if args.run_densification else None, 'final_static_npz': os.path.join(static_dir, 'static_recon_labels.npz')
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
        print(f'  strict residue dir     : {static_filter_dir}')
    print(f'  densification          : {"YES" if args.run_densification else "NO"}')
    print(f'  final static NPZ       : {os.path.join(static_dir, "static_recon_labels.npz")}')
    print(f'  residues               : {residue_dir}')
    print(f'  compute backend        : {SPATIAL_SEARCH.mode}')
    if SPATIAL_SEARCH.mode == 'cuda': print(f'  CUDA GPUs              : {SPATIAL_SEARCH.gpu_ids}')
    print(f'  total wall time        : {total_wall_s:.3f} s')
    print(f'  timing summary         : {timing_summary_path}')
    print()
    print(f'Output root:\n{root}')
    
if __name__ == '__main__':
    main()