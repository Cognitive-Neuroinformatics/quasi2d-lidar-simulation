#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)";BUILD="$ROOT/build/pcl_mls"
cmake -S "$ROOT/reconstruction/pcl_mls_cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j "${JOBS:-$(nproc)}"
echo "Built: $BUILD/pcl_mls_reconstruct"
