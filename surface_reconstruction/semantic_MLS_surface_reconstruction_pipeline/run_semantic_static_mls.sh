#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CASE="${CASE:-segment-17791493328130181905_1480_000_1500_000_with_camera_labels}"

DATASET_ROOT="${DATASET_ROOT:-/media/samanti/9137dc79-aec9-43a3-8017-08ed5130ad58/home/samanti/waymo/road_reconstruction_study}"

STATIC_DIR="${STATIC_DIR:-${DATASET_ROOT}/recon_related/${CASE}/static_filter}"

DYNAMIC_ROOT="${DYNAMIC_ROOT:-${DATASET_ROOT}/temp/${CASE}/occ/preproc/dynamic/objects}"

# ----------------------------------------------------------------------
# Densification inputs/outputs
# ----------------------------------------------------------------------

STRICT_NPZ="${STRICT_NPZ:-${STATIC_DIR}/static_recon_labels_strict.npz}"

DENSIFY_SCRIPT="${DENSIFY_SCRIPT:-${SCRIPT_DIR}/preprocessing/reconstruct_static_densified_pointcloud.py}"

SUPPORT_NPZ="${SUPPORT_NPZ:-${STATIC_DIR}/scene_ground_support_adaptive.npz}"

DENSIFIED_NPZ="${DENSIFIED_NPZ:-${STATIC_DIR}/static_recon_labels_strict_densified_adaptive.npz}"

DENSIFIED_PCD="${DENSIFIED_PCD:-${STATIC_DIR}/static_recon_labels_strict_densified_adaptive.pcd}"

# Adaptive densification parameters.
ANALYSIS_VOXEL="${ANALYSIS_VOXEL:-0.05}"
FILL_SPACING="${FILL_SPACING:-0.03}"
FILL_SPACING_MODE="${FILL_SPACING_MODE:-local_ring_density}"

ADAPTIVE_DENSITY_VOXEL="${ADAPTIVE_DENSITY_VOXEL:-0.01}"
ADAPTIVE_DENSITY_NEIGHBORS="${ADAPTIVE_DENSITY_NEIGHBORS:-3}"
ADAPTIVE_QUERY_NEIGHBORS="${ADAPTIVE_QUERY_NEIGHBORS:-16}"

ADAPTIVE_FILL_SCALE="${ADAPTIVE_FILL_SCALE:-1.0}"
ADAPTIVE_FILL_MIN="${ADAPTIVE_FILL_MIN:-0.01}"
ADAPTIVE_FILL_MAX="${ADAPTIVE_FILL_MAX:-0.08}"

MIN_SEPARATION="${MIN_SEPARATION:-0.01}"
DEDUP_SPACING="${DEDUP_SPACING:-0.005}"

# 0 means unlimited.
MAX_CANDIDATES_PER_FAMILY="${MAX_CANDIDATES_PER_FAMILY:-0}"
MAX_SUPPORT="${MAX_SUPPORT:-0}"

# Static MLS reconstruction uses ORIGINAL + generated support.
INPUT_NPZ="${INPUT_NPZ:-${DENSIFIED_NPZ}}"

# ----------------------------------------------------------------------
# MLS reconstruction outputs/configuration
# ----------------------------------------------------------------------

OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/semantic_aware_mls/semantic_static_mls/${CASE}}"

LOG_ROOT="${LOG_ROOT:-${DATASET_ROOT}/semantic_aware_mls/logs/${CASE}}"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_pcl_mls}"

PCL_EXECUTABLE="${PCL_EXECUTABLE:-${BUILD_DIR}/pcl_mls_reconstruct}"

PYTHON_BIN="${PYTHON_BIN:-python}"

PCL_THREADS="${PCL_THREADS:-8}"

CONFIG="${SCRIPT_DIR}/semantic_static_mls_v1.json"


build() {

  env -u LD_LIBRARY_PATH /usr/bin/cmake \
    -S "${SCRIPT_DIR}/pcl_mls_cpp" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl

  env -u LD_LIBRARY_PATH /usr/bin/cmake \
    --build "${BUILD_DIR}" \
    --parallel "$(nproc)"
}


densify() {

  [[ -f "${STRICT_NPZ}" ]] || {
    echo "ERROR: strict input not found: ${STRICT_NPZ}" >&2
    exit 1
  }

  [[ -f "${DENSIFY_SCRIPT}" ]] || {
    echo "ERROR: adaptive densification script not found: ${DENSIFY_SCRIPT}" >&2
    exit 1
  }

  mkdir -p "${STATIC_DIR}"

  echo "======================================================================"
  echo "ADAPTIVE GROUND DENSIFICATION"
  echo "======================================================================"
  echo "Case                       : ${CASE}"
  echo "Strict input               : ${STRICT_NPZ}"
  echo "Support output             : ${SUPPORT_NPZ}"
  echo "Densified NPZ              : ${DENSIFIED_NPZ}"
  echo "Densified PCD              : ${DENSIFIED_PCD}"
  echo "CPU cores                  : $(nproc)"
  echo
  echo "Analysis voxel             : ${ANALYSIS_VOXEL} m"
  echo "Generic fill spacing       : ${FILL_SPACING} m"
  echo "Ring fill-spacing mode     : ${FILL_SPACING_MODE}"
  echo "Density-estimation voxel   : ${ADAPTIVE_DENSITY_VOXEL} m"
  echo "Density NN                 : ${ADAPTIVE_DENSITY_NEIGHBORS}"
  echo "Density query neighbours   : ${ADAPTIVE_QUERY_NEIGHBORS}"
  echo "Adaptive fill scale        : ${ADAPTIVE_FILL_SCALE}"
  echo "Adaptive fill min/max      : ${ADAPTIVE_FILL_MIN} / ${ADAPTIVE_FILL_MAX} m"
  echo "Minimum separation         : ${MIN_SEPARATION} m"
  echo "Deduplication spacing      : ${DEDUP_SPACING} m"
  echo "Candidate cap/family       : ${MAX_CANDIDATES_PER_FAMILY} (0=unlimited)"
  echo "Global support cap         : ${MAX_SUPPORT} (0=unlimited)"
  echo "Generic coverage           : enabled"
  echo "======================================================================"

  /usr/bin/time -v \
    env \
      OMP_NUM_THREADS="$(nproc)" \
      OPENBLAS_NUM_THREADS="$(nproc)" \
      MKL_NUM_THREADS="$(nproc)" \
      NUMEXPR_NUM_THREADS="$(nproc)" \
      "${PYTHON_BIN}" "${DENSIFY_SCRIPT}" \
        -s "${DATASET_ROOT}" \
        --caseid "${CASE}" \
        --strict_path "${STRICT_NPZ}" \
        -o "${SUPPORT_NPZ}" \
        --generic_coverage \
        --analysis_voxel "${ANALYSIS_VOXEL}" \
        --fill_spacing "${FILL_SPACING}" \
        --fill_spacing_mode "${FILL_SPACING_MODE}" \
        --adaptive_density_voxel "${ADAPTIVE_DENSITY_VOXEL}" \
        --adaptive_density_neighbors "${ADAPTIVE_DENSITY_NEIGHBORS}" \
        --adaptive_query_neighbors "${ADAPTIVE_QUERY_NEIGHBORS}" \
        --adaptive_fill_scale "${ADAPTIVE_FILL_SCALE}" \
        --adaptive_fill_min "${ADAPTIVE_FILL_MIN}" \
        --adaptive_fill_max "${ADAPTIVE_FILL_MAX}" \
        --min_separation "${MIN_SEPARATION}" \
        --deduplicate \
        --dedup_spacing "${DEDUP_SPACING}" \
        --max_candidates_per_family "${MAX_CANDIDATES_PER_FAMILY}" \
        --max_support "${MAX_SUPPORT}" \
        --densified_pcd "${DENSIFIED_PCD}" \
        --densified_npz "${DENSIFIED_NPZ}"

  [[ -f "${DENSIFIED_NPZ}" ]] || {
    echo "ERROR: densification finished but merged NPZ was not created: ${DENSIFIED_NPZ}" >&2
    exit 1
  }

  echo
  echo "Densification complete."
  echo "Static MLS input will be: ${DENSIFIED_NPZ}"
  echo
}


reconstruct() {

  [[ -f "${INPUT_NPZ}" ]] || {
    echo "ERROR: MLS input not found: ${INPUT_NPZ}" >&2
    echo "Run '$0 densify' first, or set INPUT_NPZ explicitly." >&2
    exit 1
  }

  [[ -x "${PCL_EXECUTABLE}" ]] || {
    echo "ERROR: PCL executable not found. Run '$0 build' outside Conda first." >&2
    exit 1
  }

  if [[ ! -d "${DYNAMIC_ROOT}" ]]; then
    echo "WARNING: dynamic-object directory not found:"
    echo "         ${DYNAMIC_ROOT}"
    echo "         Static reconstruction will still run, but dynamic_objects will be skipped."
  fi

  local run_id log_dir log_file time_file provenance_file status

  run_id="$(date '+%Y%m%d_%H%M%S')"

  log_dir="${LOG_ROOT}/${run_id}"

  mkdir -p "${log_dir}"

  log_file="${log_dir}/experiment.log"

  time_file="${log_dir}/timing.txt"

  provenance_file="${log_dir}/provenance.txt"

  cp "${CONFIG}" "${log_dir}/semantic_static_mls_v1.json"

  cp "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" \
    "${log_dir}/reconstruct_semantic_static_mls.py"

  cp "${SCRIPT_DIR}/pcl_mls_cpp/pcl_mls_reconstruct.cpp" \
    "${log_dir}/pcl_mls_reconstruct.cpp"

  cp "${DENSIFY_SCRIPT}" \
    "${log_dir}/densify_static_ground_adaptive_spacing.py"

  {
    echo "======================================================================"
    echo "SEMANTIC STATIC + DYNAMIC MLS EXPERIMENT"
    echo "======================================================================"
    echo "Run ID          : ${run_id}"
    echo "Start time      : $(date --iso-8601=seconds)"
    echo "Hostname        : $(hostname)"
    echo "Case            : ${CASE}"
    echo "Strict source   : ${STRICT_NPZ}"
    echo "Densified input : ${INPUT_NPZ}"
    echo "Support file    : ${SUPPORT_NPZ}"
    echo "Dynamic input   : ${DYNAMIC_ROOT}"
    echo "Output          : ${OUTPUT_ROOT}"
    echo "Config          : ${CONFIG}"
    echo "PCL executable  : ${PCL_EXECUTABLE}"
    echo "PCL threads     : ${PCL_THREADS}"
    echo "Tile size       : 25 m"
    echo "Tile halo       : 0.5 m"
    echo "Min confidence  : 0.66"
    echo "Stages          : background static_objects dynamic_objects"
    echo "Voxel sampling  : disabled/enforced"
    echo "Box re-filter   : disabled"
    echo

    echo "STATIC INPUT FILE:"
    ls -lh "${INPUT_NPZ}"
    echo

    echo "DYNAMIC OBJECT INPUT:"
    if [[ -d "${DYNAMIC_ROOT}" ]]; then
      echo "Object folders  : $(find "${DYNAMIC_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
      echo "Stitch files    : $(find "${DYNAMIC_ROOT}" -mindepth 2 -maxdepth 2 -name stitch_labeled.npz -type f | wc -l)"
    else
      echo "NOT FOUND"
    fi
    echo

    echo "SYSTEM:"
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)' || true
    free -h || true
    echo

    echo "CONFIG SNAPSHOT:"
    cat "${CONFIG}"
    echo

    echo "======================================================================"
    echo "RECONSTRUCTION START"
    echo "======================================================================"

  } | tee "${log_file}"

  {
    echo "Run ID: ${run_id}"
    echo "Recorded: $(date --iso-8601=seconds)"
    echo

    echo "SHA256:"
    sha256sum \
      "${STRICT_NPZ}" \
      "${INPUT_NPZ}" \
      "${CONFIG}" \
      "${DENSIFY_SCRIPT}" \
      "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" \
      "${SCRIPT_DIR}/pcl_mls_cpp/pcl_mls_reconstruct.cpp" \
      "${PCL_EXECUTABLE}"

    echo

    echo "COMPILER:"
    /usr/bin/g++ --version | head -n 1

    echo

    echo "PCL EXECUTABLE LIBRARIES:"
    env -u LD_LIBRARY_PATH ldd "${PCL_EXECUTABLE}" || true

  } > "${provenance_file}"

  set +e

  /usr/bin/time -v -o "${time_file}" \
    env -u LD_LIBRARY_PATH "${PYTHON_BIN}" \
      "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" \
      --dataset-root "${DATASET_ROOT}" \
      --caseid "${CASE}" \
      --static-input "${INPUT_NPZ}" \
      --dynamic-input-root "${DYNAMIC_ROOT}" \
      --pcl-executable "${PCL_EXECUTABLE}" \
      --config "${CONFIG}" \
      --output-root "${OUTPUT_ROOT}" \
      --stages background static_objects dynamic_objects \
      --tile-size 25 \
      --tile-halo 0.5 \
      --pcl-threads "${PCL_THREADS}" \
      --minimum-label-confidence 0.66 \
      --no-exclude-points-in-tracked-boxes \
      --require-no-voxel-downsampling \
      --overwrite \
      2>&1 | tee -a "${log_file}"

  status=${PIPESTATUS[0]}

  set -e

  {
    echo
    echo "======================================================================"
    echo "RECONSTRUCTION END"
    echo "======================================================================"
    echo "End time        : $(date --iso-8601=seconds)"
    echo "Exit status     : ${status}"
    echo

    echo "RESOURCE/TIMING SUMMARY:"
    cat "${time_file}"
    echo

    echo "Experiment log  : ${log_file}"
    echo "Timing file     : ${time_file}"
    echo "Provenance      : ${provenance_file}"
    echo "======================================================================"

  } | tee -a "${log_file}"

  return "${status}"
}


view() {

  "${PYTHON_BIN}" "${SCRIPT_DIR}/view_semantic_static_mls.py" \
    --reconstruction-root "${OUTPUT_ROOT}"
}


case "${ACTION}" in
  build)
    build
    ;;
  densify)
    densify
    ;;
  reconstruct)
    reconstruct
    ;;
  pipeline)
    densify
    reconstruct
    ;;
  view)
    view
    ;;
  all)
    build
    densify
    reconstruct
    ;;
  *)
    echo "Usage: $0 {build|densify|reconstruct|pipeline|view|all}" >&2
    exit 2
    ;;
esac
