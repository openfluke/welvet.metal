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
# Default: Dense. Layer-only: … / SEQMIX / CNN.
if [[ "${CONVT:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalConvT$' -timeout 60m
elif [[ "${CNN:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalCNN$' -timeout 60m
elif [[ "${SEQMIX:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSeqmix$' -timeout 10m
elif [[ "${RESIDUAL:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalResidual$' -timeout 45m
elif [[ "${SEQUENTIAL:-0}" == "1" || "${SEQ:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSequential$' -timeout 45m
elif [[ "${EMBEDDING:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalEmbedding$' -timeout 45m
elif [[ "${SOFTMAX:-0}" == "1" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSoftmax$' -timeout 45m
elif [[ "${LAYERNORM:-0}" == "1" ]]; then
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
