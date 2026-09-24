#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="${CASE:-segment-89454214745557131_3160_000_3180_000_with_camera_labels}"
DATASET_ROOT="${DATASET_ROOT:-/data/waymo/surface_reconstruction}"
STATIC_NPZ="${STATIC_NPZ:-${DATASET_ROOT}/recon_related/${CASE}/static_recon_labels.npz}"
DYNAMIC_ROOT="${DYNAMIC_ROOT:-${DATASET_ROOT}/temp/${CASE}/occ/preproc/dynamic/objects}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATASET_ROOT}/semantic_aware_mls/semantic_static_mls/${CASE}}"
LOG_ROOT="${LOG_ROOT:-${DATASET_ROOT}/semantic_aware_mls/logs/${CASE}}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_pcl_mls}"
PCL_EXECUTABLE="${PCL_EXECUTABLE:-${BUILD_DIR}/pcl_mls_reconstruct}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PCL_THREADS="${PCL_THREADS:-8}"
CONFIG="${SCRIPT_DIR}/semantic_static_mls_v1.json"

build() {
  env -u LD_LIBRARY_PATH /usr/bin/cmake -S "${SCRIPT_DIR}/pcl_mls_cpp" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl
  env -u LD_LIBRARY_PATH /usr/bin/cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
}

reconstruct() {
  [[ -f "${STATIC_NPZ}" ]] || { echo "ERROR: final preprocessed static cloud not found: ${STATIC_NPZ}" >&2; echo "Run waymo_preprocessing_cpu_cuda.py first." >&2; exit 1; }
  [[ -x "${PCL_EXECUTABLE}" ]] || { echo "ERROR: PCL executable not found. Run '$0 build' outside Conda first." >&2; exit 1; }
  [[ -d "${DYNAMIC_ROOT}" ]] || echo "WARNING: dynamic-object directory not found: ${DYNAMIC_ROOT}"
  local run_id log_dir log_file time_file provenance_file status
  run_id="$(date '+%Y%m%d_%H%M%S')"; log_dir="${LOG_ROOT}/${run_id}"; mkdir -p "${log_dir}"
  log_file="${log_dir}/experiment.log"; time_file="${log_dir}/timing.txt"; provenance_file="${log_dir}/provenance.txt"
  cp "${CONFIG}" "${log_dir}/semantic_static_mls_v1.json"; cp "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" "${log_dir}/reconstruct_semantic_static_mls.py"; cp "${SCRIPT_DIR}/pcl_mls_cpp/pcl_mls_reconstruct.cpp" "${log_dir}/pcl_mls_reconstruct.cpp"
  {
    echo "======================================================================"; echo "SEMANTIC STATIC + DYNAMIC MLS EXPERIMENT"; echo "======================================================================"
    echo "Run ID          : ${run_id}"; echo "Start time      : $(date --iso-8601=seconds)"; echo "Hostname        : $(hostname)"; echo "Case            : ${CASE}"
    echo "Static input    : ${STATIC_NPZ}"; echo "Dynamic input   : ${DYNAMIC_ROOT}"; echo "Output          : ${OUTPUT_ROOT}"; echo "Config          : ${CONFIG}"; echo "PCL executable  : ${PCL_EXECUTABLE}"; echo "PCL threads     : ${PCL_THREADS}"
    echo "Static contract : accumulated + filtered + densified upstream"; echo "Voxel sampling  : disabled/enforced"; echo "Box re-filter   : disabled"; echo
    ls -lh "${STATIC_NPZ}"; echo; lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)' || true; free -h || true
  } | tee "${log_file}"
  {
    echo "Run ID: ${run_id}"; echo "Recorded: $(date --iso-8601=seconds)"; echo; echo "SHA256:"
    sha256sum "${STATIC_NPZ}" "${CONFIG}" "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" "${SCRIPT_DIR}/pcl_mls_cpp/pcl_mls_reconstruct.cpp" "${PCL_EXECUTABLE}"
  } > "${provenance_file}"
  set +e
  /usr/bin/time -v -o "${time_file}" env -u LD_LIBRARY_PATH "${PYTHON_BIN}" "${SCRIPT_DIR}/reconstruct_semantic_static_mls.py" --dataset-root "${DATASET_ROOT}" --caseid "${CASE}" --static-input "${STATIC_NPZ}" --dynamic-input-root "${DYNAMIC_ROOT}" --pcl-executable "${PCL_EXECUTABLE}" --config "${CONFIG}" --output-root "${OUTPUT_ROOT}" --stages background static_objects dynamic_objects --tile-size 25 --tile-halo 0.5 --pcl-threads "${PCL_THREADS}" --minimum-label-confidence 0.66 --no-exclude-points-in-tracked-boxes --require-no-voxel-downsampling --overwrite 2>&1 | tee -a "${log_file}"
  status=${PIPESTATUS[0]}; set -e
  { echo; echo "End time        : $(date --iso-8601=seconds)"; echo "Exit status     : ${status}"; echo; cat "${time_file}"; } | tee -a "${log_file}"
  return "${status}"
}

view() { "${PYTHON_BIN}" "${SCRIPT_DIR}/view_semantic_static_mls.py" --reconstruction-root "${OUTPUT_ROOT}"; }

case "${ACTION}" in
  build) build ;;
  reconstruct|pipeline) reconstruct ;;
  view) view ;;
  all) build; reconstruct ;;
  *) echo "Usage: $0 {build|reconstruct|pipeline|view|all}" >&2; exit 2 ;;
esac
