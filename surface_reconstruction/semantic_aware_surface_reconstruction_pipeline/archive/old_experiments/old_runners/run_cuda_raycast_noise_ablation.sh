#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""; CASEID=""; RECON_ROOT=""; OUTPUT_ROOT=""; DEVICES="0,1,2"; START_FRAME="0"; END_FRAME=""; SENSORS="front_left front_center front_right rear_left rear_center rear_right"; PRECISION="float64"; BATCH="2000000"; GPU_CACHE_GB="5"; NPZ_MODE="stored"; OVERWRITE=1; SKIP_RAYCAST=0
RANGE_SIGMA="0.05"; AZ_SIGMA="0.1"; POLAR_SIGMA="0.6"; INCIDENCE_CAP="75.0"; NOISE_SEED="12345"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2;; --caseid) CASEID="$2"; shift 2;; --reconstruction-root) RECON_ROOT="$2"; shift 2;; --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;; --start-frame) START_FRAME="$2"; shift 2;; --end-frame) END_FRAME="$2"; shift 2;; --sensors) SENSORS="$2"; shift 2;; --precision) PRECISION="$2"; shift 2;;
    --point-batch-size) BATCH="$2"; shift 2;; --gpu-cache-gb) GPU_CACHE_GB="$2"; shift 2;; --npz-compression) NPZ_MODE="$2"; shift 2;;
    --range-sigma-m) RANGE_SIGMA="$2"; shift 2;; --azimuth-sigma-deg) AZ_SIGMA="$2"; shift 2;; --polar-sigma-deg) POLAR_SIGMA="$2"; shift 2;; --incidence-max-angle-deg) INCIDENCE_CAP="$2"; shift 2;; --noise-seed) NOISE_SEED="$2"; shift 2;;
    --skip-raycast) SKIP_RAYCAST=1; shift;; --no-overwrite) OVERWRITE=0; shift;; *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

if [[ -z "$DATASET_ROOT" || -z "$CASEID" ]]; then echo "Required: --dataset-root PATH --caseid CASE" >&2; exit 2; fi
if [[ -z "$RECON_ROOT" ]]; then RECON_ROOT="$DATASET_ROOT/semantic_aware_surface_reconstruction/semantic_static_mls_cpu_optimized/$CASEID"; fi
if [[ -z "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$RECON_ROOT/scala2_raycast_cuda_${PRECISION}"; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ARGS=(); IFS=',' read -ra DEV_ARR <<< "$DEVICES"; for d in "${DEV_ARR[@]}"; do GPU_ARGS+=("$d"); done
SENSOR_ARGS=(); read -ra SENSOR_ARR <<< "$SENSORS"; for s in "${SENSOR_ARR[@]}"; do SENSOR_ARGS+=("$s"); done
END_ARGS=(); [[ -n "$END_FRAME" ]] && END_ARGS=(--end-frame "$END_FRAME")
OVERWRITE_ARGS=(); [[ "$OVERWRITE" == "1" ]] && OVERWRITE_ARGS=(--overwrite)

if [[ "$SKIP_RAYCAST" == "0" ]]; then
  python "$SCRIPT_DIR/check_cuda_environment.py"
  python "$SCRIPT_DIR/raycaster_mls_legacy/raycast_mls_scala2_cuda.py" --dataset-root "$DATASET_ROOT" --caseid "$CASEID" --reconstruction-root "$RECON_ROOT" --output-root "$OUTPUT_ROOT" --sensors "${SENSOR_ARGS[@]}" --start-frame "$START_FRAME" "${END_ARGS[@]}" --first-mirror-side 0 --minimum-range 0.5 --max-range 80 --intersection-mode tangent_patch --patch-radius 0.08 --hit-radius 0.03 --point-batch-size "$BATCH" --static-tile-cache 64 --property-cache 64 --gpu-cache-gb "$GPU_CACHE_GB" --devices "${GPU_ARGS[@]}" --precision "$PRECISION" --npz-compression "$NPZ_MODE" --noise-output clean "${OVERWRITE_ARGS[@]}"
fi

ABLATION_OVERWRITE=(); [[ "$OVERWRITE" == "1" ]] && ABLATION_OVERWRITE=(--overwrite)
python "$SCRIPT_DIR/raycaster_mls_legacy/generate_scala2_noise_ablation.py" --raycast-root "$OUTPUT_ROOT" --sensors "${SENSOR_ARGS[@]}" --range-sigma-m "$RANGE_SIGMA" --azimuth-sigma-deg "$AZ_SIGMA" --polar-sigma-deg "$POLAR_SIGMA" --incidence-max-angle-deg "$INCIDENCE_CAP" --seed "$NOISE_SEED" --npz-compression "$NPZ_MODE" "${ABLATION_OVERWRITE[@]}"

echo "========================================================================"
echo "NOISE ABLATION COMPLETE"
echo "========================================================================"
echo "Clean              : <sensor>/points/"
echo "Range only         : <sensor>/points_noise_range_only/"
echo "Angular only       : <sensor>/points_noise_angular_only/"
echo "Datasheet full     : <sensor>/points_noise_datasheet/"
echo "Incidence full     : <sensor>/points_noise_incidence/"
echo "Summary            : $OUTPUT_ROOT/noise_ablation_summary.json"
echo "Per-frame metrics  : $OUTPUT_ROOT/noise_ablation_per_frame.csv"
