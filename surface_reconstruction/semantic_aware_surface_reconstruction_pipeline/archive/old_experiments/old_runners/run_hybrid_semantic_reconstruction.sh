#!/usr/bin/env bash
set -euo pipefail

CASEID="${1:-segment-191862526745161106_1400_000_1420_000_with_camera_labels}"

PIPELINE_DIR="${PIPELINE_DIR:-/home/samanti/git_repos/object_detection_dl/LiDAR-GS/semantic_aware_surface_reconstruction_pipeline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/Waymo_Surface_Reconstruction}"

OUTPUT_ROOT="$DATASET_ROOT/semantic_aware_surface_reconstruction/hybrid_semantic_surface/$CASEID"

python "$SCRIPT_DIR/reconstruct_hybrid_semantic_surface.py" \
  --dataset-root "$DATASET_ROOT" \
  --caseid "$CASEID" \
  --hybrid-config "$SCRIPT_DIR/hybrid_semantic_surface_v1.json" \
  --base-mls-config "$PIPELINE_DIR/semantic_static_mls_v1.json" \
  --mls-script "$PIPELINE_DIR/reconstruct_semantic_static_mls.py" \
  --pcl-executable "$PIPELINE_DIR/build_pcl_mls/pcl_mls_reconstruct" \
  --output-root "$OUTPUT_ROOT" \
  --tile-size 25 \
  --tile-halo 0.5 \
  --minimum-label-confidence 0.66 \
  --surface-workers 12 \
  --pcl-threads 8 \
  --mls-workers 3 \
  --attribute-workers 1 \
  --npz-compression stored \
  --include-dynamics \
  --overwrite

echo
echo "Hybrid reconstruction:"
echo "$OUTPUT_ROOT"
