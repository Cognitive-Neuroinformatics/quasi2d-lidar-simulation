#!/usr/bin/env python3
"""Run semantic MLS reconstruction + SCALA-2 raycasting for an ordered scene list.

This is the batch equivalent of benchmark_one_scene_unified.py. It intentionally
runs one scene at a time so that PCL reconstruction and multi-GPU raycasting can
use the full machine without cross-scene RAM/GPU contention.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def case_name(entry: str) -> str:
    name = Path(entry.strip()).name
    return name[:-len('.tfrecord')] if name.endswith('.tfrecord') else name


def read_split_files(paths: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError(f"Split file not found: {path}")
        with path.open() as stream:
            for line_number, raw in enumerate(stream, start=1):
                value = raw.split('#', 1)[0].strip()
                if not value: continue
                case = case_name(value)
                if case in seen: continue
                rows.append({'case': case, 'entry': value, 'split_file': str(path), 'line_number': line_number})
                seen.add(case)
    return rows


def option_value(arguments: list[str], name: str, default=None):
    for i, token in enumerate(arguments):
        if token == name and i + 1 < len(arguments): return arguments[i + 1]
        if token.startswith(name + '='): return token.split('=', 1)[1]
    return default


def has_flag(arguments: list[str], name: str) -> bool:
    return name in arguments


def standard_paths(dataset_root: Path, case: str, rasterizer: str, cuda_precision: str):
    reconstruction = dataset_root / 'semantic_aware_surface_reconstruction' / 'semantic_static_mls_cpu_optimized' / case
    raycast = reconstruction / ('scala2_raycast_cpu_optimized' if rasterizer == 'cpu' else f'scala2_raycast_cuda_{cuda_precision}')
    return reconstruction, raycast


def outputs_complete(dataset_root: Path, case: str, rasterizer: str, cuda_precision: str, skip_reconstruct: bool, skip_raycast: bool) -> bool:
    reconstruction, raycast = standard_paths(dataset_root, case, rasterizer, cuda_precision)
    reconstruct_ok = True if skip_reconstruct else False
    if not skip_reconstruct:
        manifest = reconstruction / 'static_manifest.json'
        if manifest.is_file():
            try: reconstruct_ok = bool(json.loads(manifest.read_text()).get('complete', False))
            except Exception: reconstruct_ok = False
    raycast_ok = True if skip_raycast else (raycast / 'raycast_summary.json').is_file()
    return reconstruct_ok and raycast_ok


def save_json_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument('--split-file', action='append', type=Path, required=True, help='Text file containing case IDs or TFRecord names. Repeat to combine lists.')
    parser.add_argument('--dataset-root', type=Path, required=True, help='Preprocessed Waymo dataset root containing recon_related/, meta_infos/, laser_calibrations/, and temp/.')
    parser.add_argument('--continue-on-error', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--skip-complete', action=argparse.BooleanOptionalAction, default=True, help='Skip scenes that already have a complete reconstruction manifest and raycast summary for the standard output layout.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--batch-root', type=Path, default=None, help='Batch logs/manifests root. Default: <dataset-root>/semantic_aware_surface_reconstruction/batch_runs/<split-name>.')
    args, pipeline_args = parser.parse_known_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir(): raise NotADirectoryError(dataset_root)
    if option_value(pipeline_args, '--reconstruction-root') is not None:
        raise ValueError('--reconstruction-root is intentionally not supported in list mode because one fixed path would collide across scenes. Use the standard per-case output layout.')

    scenes = read_split_files(args.split_file)
    if not scenes: raise RuntimeError('No scenes found in split file(s).')
    split_name = '__'.join(path.stem for path in args.split_file)
    batch_root = args.batch_root.expanduser().resolve() if args.batch_root else dataset_root / 'semantic_aware_surface_reconstruction' / 'batch_runs' / split_name
    log_dir = batch_root / 'logs'; batch_root.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_root / 'manifest.json'

    rasterizer = str(option_value(pipeline_args, '--rasterizer', 'cuda'))
    cuda_precision = str(option_value(pipeline_args, '--cuda-precision', 'float64'))
    skip_reconstruct = has_flag(pipeline_args, '--skip-reconstruct')
    skip_raycast = has_flag(pipeline_args, '--skip-raycast')

    old_status = {}
    if manifest_path.is_file():
        try: old_status = {row['case']: row.get('status') for row in json.loads(manifest_path.read_text()).get('scenes', [])}
        except Exception: old_status = {}

    manifest = {
        'schema_version': 1,
        'split_files': [str(x.expanduser().resolve()) for x in args.split_file],
        'dataset_root': str(dataset_root),
        'execution_mode': 'sequential_one_scene_at_a_time',
        'pipeline_script': str((Path(__file__).resolve().parent / 'benchmark_one_scene_unified.py')),
        'forwarded_pipeline_args': pipeline_args,
        'rasterizer': rasterizer,
        'scenes': [{**row, 'status': 'pending'} for row in scenes],
    }

    project = Path(__file__).resolve().parent
    single_scene = project / 'benchmark_one_scene_unified.py'
    counts = Counter()

    print('=' * 84)
    print('SEMANTIC MLS + SCALA-2 BATCH PIPELINE')
    print('=' * 84)
    print(f'Scenes          : {len(scenes)}')
    print(f'Dataset root    : {dataset_root}')
    print(f'Rasterizer      : {rasterizer}')
    print('Parallel scenes : NO')
    print(f'Batch logs      : {log_dir}')
    print(f'Manifest        : {manifest_path}')

    for index, row in enumerate(manifest['scenes'], start=1):
        case = row['case']
        static_npz = dataset_root / 'recon_related' / case / 'static_recon_labels.npz'
        meta_info = dataset_root / 'meta_infos' / f'{case}.pkl'
        calibration = dataset_root / 'laser_calibrations' / case / 'laser_calibrations' / 'laser_calibrations.npz'
        missing = [str(p) for p in (static_npz, meta_info, calibration) if not p.is_file()]
        if missing:
            row['status'] = 'missing_preprocessed_input'; row['missing'] = missing; counts[row['status']] += 1
            print(f'[{index:02d}/{len(scenes):02d}] MISSING {case}')
            for path in missing: print(f'  {path}')
            save_json_atomic(manifest_path, manifest)
            if not args.continue_on_error: break
            continue

        if args.skip_complete and outputs_complete(dataset_root, case, rasterizer, cuda_precision, skip_reconstruct, skip_raycast):
            row['status'] = 'skipped_complete'; counts[row['status']] += 1
            print(f'[{index:02d}/{len(scenes):02d}] SKIP    {case}')
            save_json_atomic(manifest_path, manifest)
            continue

        command = [sys.executable, '-u', str(single_scene), '--dataset-root', str(dataset_root), '--caseid', case, *pipeline_args]
        log_path = log_dir / f'{case}.log'; row['log'] = str(log_path); row['command'] = command
        print('\n' + '-' * 84); print(f'[{index:02d}/{len(scenes):02d}] RUN {case}'); print(' '.join(command)); print('-' * 84)
        if args.dry_run:
            row['status'] = 'dry_run'; counts[row['status']] += 1; save_json_atomic(manifest_path, manifest); continue

        row['status'] = 'running'; row['started_unix'] = time.time(); save_json_atomic(manifest_path, manifest)
        started = time.perf_counter()
        with log_path.open('a', buffering=1) as log:
            log.write('\n' + '=' * 100 + '\nCOMMAND: ' + ' '.join(command) + '\n' + '=' * 100 + '\n')
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=project, env=os.environ.copy())
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end=''); log.write(line)
            return_code = process.wait()
        row['return_code'] = int(return_code); row['wall_seconds'] = float(time.perf_counter() - started); row['finished_unix'] = time.time()
        row['status'] = 'completed' if return_code == 0 else 'failed'; counts[row['status']] += 1; save_json_atomic(manifest_path, manifest)
        if return_code != 0 and not args.continue_on_error: break

    counts = Counter(row['status'] for row in manifest['scenes'])
    manifest['counts'] = dict(counts); save_json_atomic(manifest_path, manifest)
    print('\n' + '=' * 84); print('BATCH COMPLETE'); print('=' * 84)
    for key in ('completed','skipped_complete','missing_preprocessed_input','failed','pending','dry_run'): print(f'{key:28s}: {counts.get(key,0)}')
    print(f'Manifest: {manifest_path}')
    if (counts.get('failed',0) or counts.get('missing_preprocessed_input',0)) and not args.continue_on_error: raise RuntimeError('Batch stopped after an error.')


if __name__ == '__main__': main()
