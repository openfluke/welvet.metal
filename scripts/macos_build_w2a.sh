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
# Default: Dense. MHA=1 / SWIGLU=1 / RMSNORM=1 / LAYERNORM=1 for layer-only suites.
if [[ "${LAYERNORM:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalLayerNorm$' -timeout 45m
elif [[ "${RMSNORM:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalRMSNorm$' -timeout 45m
elif [[ "${SWIGLU:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSwiGLU$' -timeout 45m
elif [[ "${MHA:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalMHA$' -timeout 45m
else
  go test ./tests/go_v_metal -v -count=1 -run TestGoVMetalDenseForward -timeout 45m
fi
