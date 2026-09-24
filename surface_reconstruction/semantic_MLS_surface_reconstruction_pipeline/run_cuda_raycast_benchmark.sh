#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=""
CASEID=""
RECON_ROOT=""
OUTPUT_ROOT=""
DEVICES="0,1,2"
START_FRAME="0"
END_FRAME=""
SENSORS="front_left front_center front_right rear_left rear_center rear_right"
PRECISION="float64"
BATCH="2000000"
GPU_CACHE_GB="5"
NPZ_MODE="stored"
PATCH_RADIUS="0.08"
HIT_RADIUS="0.03"
NOISE_OUTPUT="both"
NOISE_RANGE_SIGMA_M="0.05"
NOISE_AZIMUTH_SIGMA_DEG="0.1"
NOISE_POLAR_SIGMA_DEG="0.6"
NOISE_SEED="12345"
OVERWRITE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2;;
    --caseid) CASEID="$2"; shift 2;;
    --reconstruction-root) RECON_ROOT="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --devices) DEVICES="$2"; shift 2;;
    --start-frame) START_FRAME="$2"; shift 2;;
    --end-frame) END_FRAME="$2"; shift 2;;
    --sensors) SENSORS="$2"; shift 2;;
    --precision) PRECISION="$2"; shift 2;;
    --point-batch-size) BATCH="$2"; shift 2;;
    --gpu-cache-gb) GPU_CACHE_GB="$2"; shift 2;;
    --npz-compression) NPZ_MODE="$2"; shift 2;;
    --patch-radius) PATCH_RADIUS="$2"; shift 2;;
    --hit-radius) HIT_RADIUS="$2"; shift 2;;
    --noise-output) NOISE_OUTPUT="$2"; shift 2;;
    --noise-range-sigma-m) NOISE_RANGE_SIGMA_M="$2"; shift 2;;
    --noise-azimuth-sigma-deg) NOISE_AZIMUTH_SIGMA_DEG="$2"; shift 2;;
    --noise-polar-sigma-deg) NOISE_POLAR_SIGMA_DEG="$2"; shift 2;;
    --noise-seed) NOISE_SEED="$2"; shift 2;;
    --no-overwrite) OVERWRITE=0; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

if [[ -z "$DATASET_ROOT" || -z "$CASEID" ]]; then echo "Required: --dataset-root PATH --caseid CASE" >&2; exit 2; fi
if [[ -z "$RECON_ROOT" ]]; then RECON_ROOT="$DATASET_ROOT/semantic_aware_mls/semantic_static_mls_cpu_optimized/$CASEID"; fi
if [[ -z "$OUTPUT_ROOT" ]]; then OUTPUT_ROOT="$RECON_ROOT/scala2_raycast_cuda_${PRECISION}"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_ROOT"
LOG_ROOT="$OUTPUT_ROOT/benchmark_logs"; mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
GPU_LOG="$LOG_ROOT/gpu_usage_${STAMP}.csv"
TIME_LOG="$LOG_ROOT/time_${STAMP}.txt"
STDOUT_LOG="$LOG_ROOT/raycast_${STAMP}.log"

python "$SCRIPT_DIR/check_cuda_environment.py"
GPU_ARGS=(); IFS=',' read -ra DEV_ARR <<< "$DEVICES"; for d in "${DEV_ARR[@]}"; do GPU_ARGS+=("$d"); done
SENSOR_ARGS=(); read -ra SENSOR_ARR <<< "$SENSORS"; for s in "${SENSOR_ARR[@]}"; do SENSOR_ARGS+=("$s"); done
END_ARGS=(); if [[ -n "$END_FRAME" ]]; then END_ARGS=(--end-frame "$END_FRAME"); fi
OVERWRITE_ARGS=(); if [[ "$OVERWRITE" == "1" ]]; then OVERWRITE_ARGS=(--overwrite); fi

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2 > "$GPU_LOG" & NVIDIA_PID=$!; else NVIDIA_PID=""; fi
cleanup() { if [[ -n "${NVIDIA_PID:-}" ]]; then kill "$NVIDIA_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM

CMD=(python "$SCRIPT_DIR/raycaster/raycast_mls_scala2_cuda.py" --dataset-root "$DATASET_ROOT" --caseid "$CASEID" --reconstruction-root "$RECON_ROOT" --output-root "$OUTPUT_ROOT" --sensors "${SENSOR_ARGS[@]}" --start-frame "$START_FRAME" "${END_ARGS[@]}" --first-mirror-side 0 --minimum-range 0.5 --max-range 80 --intersection-mode tangent_patch --patch-radius "$PATCH_RADIUS" --hit-radius "$HIT_RADIUS" --point-batch-size "$BATCH" --static-tile-cache 64 --property-cache 64 --gpu-cache-gb "$GPU_CACHE_GB" --devices "${GPU_ARGS[@]}" --precision "$PRECISION" --npz-compression "$NPZ_MODE" --noise-output "$NOISE_OUTPUT" --noise-range-sigma-m "$NOISE_RANGE_SIGMA_M" --noise-azimuth-sigma-deg "$NOISE_AZIMUTH_SIGMA_DEG" --noise-polar-sigma-deg "$NOISE_POLAR_SIGMA_DEG" --noise-seed "$NOISE_SEED" "${OVERWRITE_ARGS[@]}")

echo "========================================================================"
echo "CUDA RAYCAST BENCHMARK"
echo "========================================================================"
echo "MLS input   : $RECON_ROOT"
echo "Output      : $OUTPUT_ROOT"
echo "Noise       : $NOISE_OUTPUT | range=$NOISE_RANGE_SIGMA_M m azimuth=$NOISE_AZIMUTH_SIGMA_DEG deg polar=$NOISE_POLAR_SIGMA_DEG deg seed=$NOISE_SEED"
echo "GPU log     : $GPU_LOG"
echo "Time log    : $TIME_LOG"
echo "Console log : $STDOUT_LOG"
printf 'Command     :'; printf ' %q' "${CMD[@]}"; echo

/usr/bin/time -v -o "$TIME_LOG" "${CMD[@]}" 2>&1 | tee "$STDOUT_LOG"
cleanup; trap - EXIT INT TERM

echo "========================================================================"
echo "BENCHMARK COMPLETE"
echo "========================================================================"
echo "Raycast summary: $OUTPUT_ROOT/raycast_summary.json"
echo "GPU samples    : $GPU_LOG"
echo "Resource time  : $TIME_LOG"