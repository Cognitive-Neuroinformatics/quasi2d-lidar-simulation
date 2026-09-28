#!/usr/bin/env bash
set -euo pipefail
CASEID="${1:-segment-191862526745161106_1400_000_1420_000_with_camera_labels}"
END_FRAME="${2:-42}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/Waymo_Surface_Reconstruction}"
STATIC_IN="$DATASET_ROOT/recon_related/$CASEID/static_recon_labels.npz"
REG_IN="$DATASET_ROOT/recon_related/$CASEID/static_recon_labels_trajectory_regularized.npz"
RECON_ROOT="$DATASET_ROOT/semantic_aware_surface_reconstruction/trajectory_regularized_ground/$CASEID"
RAY_ROOT="$RECON_ROOT/scala2_raycast_cuda_float64"

echo '=== 1/3 trajectory-guided ROAD/LANE regularization ==='
python "$PROJECT_DIR/trajectory_regularize_ground.py" \
  --dataset-root "$DATASET_ROOT" --caseid "$CASEID" \
  --input-npz "$STATIC_IN" --output-npz "$REG_IN" \
  --families road_lane \
  --trajectory-sample-m 0.20 \
  --trajectory-z-median-window-m 3.0 \
  --trajectory-z-smooth-window-m 5.0 \
  --fit-knot-spacing-m 0.50 \
  --fit-half-window-m 0.75 \
  --fit-lateral-max-m 12.0 \
  --profile-smooth-window-m 4.0 \
  --cross-slope-smooth-window-m 5.0 \
  --trajectory-prior-blend 0.25 \
  --max-generated-shift-m 0.15 \
  --compression stored

echo '=== 2/3 semantic MLS reconstruction, no ground upsampling ==='
python "$PROJECT_DIR/reconstruct_semantic_static_mls.py" \
  --dataset-root "$DATASET_ROOT" --caseid "$CASEID" \
  --static-input "$REG_IN" \
  --pcl-executable "$PROJECT_DIR/build_pcl_mls/pcl_mls_reconstruct" \
  --config "$PROJECT_DIR/semantic_static_mls_v1_ground_no_upsampling.json" \
  --output-root "$RECON_ROOT" \
  --stages background static_objects dynamic_objects \
  --tile-size 25 --tile-halo 0.5 \
  --pcl-threads 8 --mls-workers 3 --attribute-workers 1 \
  --minimum-label-confidence 0.66 \
  --no-exclude-points-in-tracked-boxes \
  --require-no-voxel-downsampling \
  --npz-compression stored --overwrite

echo '=== 3/3 CUDA tangent-patch raycast, front_center ==='
python "$PROJECT_DIR/raycaster_mls_legacy/raycast_mls_scala2_cuda.py" \
  --dataset-root "$DATASET_ROOT" --caseid "$CASEID" \
  --reconstruction-root "$RECON_ROOT" --output-root "$RAY_ROOT" \
  --sensors front_center --start-frame 0 --end-frame "$END_FRAME" \
  --first-mirror-side 0 --minimum-range 0.5 --max-range 80 \
  --intersection-mode tangent_patch --patch-radius 0.08 --hit-radius 0.08 \
  --point-batch-size 2000000 --static-tile-cache 64 --property-cache 64 \
  --gpu-cache-gb 5 --devices 0 1 2 --precision float64 \
  --npz-compression stored --overwrite

echo "DONE"
echo "Regularized input: $REG_IN"
echo "Reconstruction   : $RECON_ROOT"
echo "Raycast          : $RAY_ROOT"