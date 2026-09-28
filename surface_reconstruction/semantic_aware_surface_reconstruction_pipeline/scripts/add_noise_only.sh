#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)";CASE="${1:?usage: $0 CASE [options]}";shift
exec python "$ROOT/pipeline.py" noise "$CASE" "$@"
