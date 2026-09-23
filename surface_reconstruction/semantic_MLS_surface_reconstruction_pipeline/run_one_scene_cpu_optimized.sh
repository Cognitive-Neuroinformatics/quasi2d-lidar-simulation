#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Exact CPU-optimized full pipeline:
# densification -> static+dynamic PCL MLS -> six-sensor SCALA2 raycasting.
# Scientific geometry parameters remain unchanged from the uploaded pipeline.
exec python benchmark_one_scene_cpu.py "$@"
