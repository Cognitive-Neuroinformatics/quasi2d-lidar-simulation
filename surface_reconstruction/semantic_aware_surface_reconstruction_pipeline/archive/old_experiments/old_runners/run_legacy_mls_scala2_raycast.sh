#!/usr/bin/env bash
set -euo pipefail
CASEID="${1:?usage: $0 CASEID RECON_ROOT [START_FRAME] [END_FRAME]}"
RECON_ROOT="${2:?usage: $0 CASEID RECON_ROOT [START_FRAME] [END_FRAME]}"
START_FRAME="${3:-0}";END_FRAME="${4:-42}"
PIPELINE_DIR="${PIPELINE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/Waymo_Surface_Reconstruction}"
OUTPUT_ROOT="$RECON_ROOT/scala2_raycast_cuda_float64"
python "$PIPELINE_DIR/raycaster_mls_legacy/raycast_mls_scala2_cuda.py" --dataset-root "$DATASET_ROOT" --caseid "$CASEID" --reconstruction-root "$RECON_ROOT" --output-root "$OUTPUT_ROOT" --sensors front_center rear_center --start-frame "$START_FRAME" --end-frame "$END_FRAME" --first-mirror-side 0 --minimum-range 0.5 --max-range 80 --intersection-mode tangent_patch --patch-radius 0.08 --hit-radius 0.08 --point-batch-size 2000000 --static-tile-cache 64 --property-cache 64 --gpu-cache-gb 5 --devices 0 1 2 --precision float64 --npz-compression stored --overwrite
