#!/usr/bin/env bash
# Build welvet.metal then run w2a go_v_metal Dense forward suite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="$(cd "$ROOT/../w2a" && pwd)"

"$ROOT/scripts/build_macos.sh"

export WELVET_METAL_DENSE_FWD="${WELVET_METAL_DENSE_FWD:-$ROOT/build/welvet_dense_fwd}"
# Optional C++ speed column:
if [[ -x "$ROOT/../welvet.cpp/build/welvet_dense_fwd" ]]; then
  export WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-$ROOT/../welvet.cpp/build/welvet_dense_fwd}"
fi

cd "$W2A"
echo "WELVET_METAL_DENSE_FWD=$WELVET_METAL_DENSE_FWD"
# Default: Dense. Pass MHA=1 for MHA-only: MHA=1 ./scripts/macos_build_w2a.sh
if [[ "${MHA:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalMHA$' -timeout 30m
else
  go test ./tests/go_v_metal -v -count=1 -run TestGoVMetalDenseForward -timeout 45m
fi
