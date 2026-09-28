#!/usr/bin/env bash
set -euo pipefail

CASEID="${1:-segment-191862526745161106_1400_000_1420_000_with_camera_labels}"
START_FRAME="${2:-0}"
END_FRAME="${3:-42}"

PIPELINE_DIR="${PIPELINE_DIR:-/home/samanti/git_repos/object_detection_dl/LiDAR-GS/semantic_aware_surface_reconstruction_pipeline}"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/Waymo_Surface_Reconstruction}"

RECON_ROOT="$DATASET_ROOT/semantic_aware_surface_reconstruction/hybrid_semantic_surface/$CASEID"
OUTPUT_ROOT="$RECON_ROOT/scala2_raycast_cuda_float64"

python "$PIPELINE_DIR/raycaster_hybrid/raycast_hybrid_scala2_cuda.py" \
  --dataset-root "$DATASET_ROOT" \
  --caseid "$CASEID" \
  --reconstruction-root "$RECON_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --sensors front_center rear_center \
  --start-frame "$START_FRAME" \
  --end-frame "$END_FRAME" \
  --first-mirror-side 0 \
  --minimum-range 0.5 \
  --max-range 80 \
  --intersection-mode tangent_patch \
  --patch-radius 0.08 \
  --hit-radius 0.08 \
  --point-batch-size 2000000 \
  --static-tile-cache 64 \
  --property-cache 64 \
  --gpu-cache-gb 5 \
  --devices 0 1 2 \
  --precision float64 \
  --npz-compression stored \
  --overwrite

echo
echo "Raycast output:"
echo "$OUTPUT_ROOT"
