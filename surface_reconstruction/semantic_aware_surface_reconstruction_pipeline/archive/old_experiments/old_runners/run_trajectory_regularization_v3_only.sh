#!/usr/bin/env bash
set -euo pipefail

CASEID="${1:-segment-191862526745161106_1400_000_1420_000_with_camera_labels}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/Waymo_Surface_Reconstruction}"

python "$PROJECT_DIR/trajectory_regularize_ground_v3.py" \
  --dataset-root "$DATASET_ROOT" \
  --caseid "$CASEID" \
  --families road_lane \
  --trajectory-sample-m 0.20 \
  --trajectory-z-median-window-m 3.0 \
  --trajectory-z-smooth-window-m 5.0 \
  --fit-knot-spacing-m 0.50 \
  --fit-half-window-m 0.75 \
  --fit-lateral-max-m 12.0 \
  --huber-delta-m 0.08 \
  --refit-observed-max-residual-m 0.20 \
  --profile-smooth-window-m 4.0 \
  --cross-slope-smooth-window-m 5.0 \
  --trajectory-prior-blend 0.20 \
  --edge-taper-m 3.0 \
  --max-abs-grade-percent 45.0 \
  --max-abs-cross-slope-percent 20.0 \
  --generated-project-max-residual-m 0.20 \
  --generated-severe-residual-m 0.50 \
  --max-generated-shift-m 0.20 \
  --projection-end-margin-m 2.0 \
  --compression stored