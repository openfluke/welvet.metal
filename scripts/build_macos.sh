#!/usr/bin/env bash
# macOS Apple silicon host build for welvet.metal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "welvet.metal build_macos.sh requires macOS" >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet_metal.dylib"
echo "  tool: $BUILD/welvet_dense_fwd"
echo "  ex:   $BUILD/example_metal_smoke"
echo
echo "smoke:  $BUILD/example_metal_smoke"
echo "gaps:   $BUILD/welvet_dense_fwd --gaps"
