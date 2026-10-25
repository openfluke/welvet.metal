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

MODE=""
for arg in "$@"; do
  case "$arg" in
    --dense-train) MODE="dense_train" ;;
    --rmsnorm-train) MODE="rmsnorm_train" ;;
    --layernorm-train) MODE="layernorm_train" ;;
    --swiglu-train) MODE="swiglu_train" ;;
    --softmax-train) MODE="softmax_train" ;;
    --embedding-train) MODE="embedding_train" ;;
    --mha-train) MODE="mha_train" ;;
    --cnn-train) MODE="cnn_train" ;;
    --convt-train) MODE="convt_train" ;;
    --rnn-train) MODE="rnn_train" ;;
    --help|-h)
      echo "usage: $0 [--dense-train|--rmsnorm-train|--layernorm-train|--swiglu-train|--softmax-train|--embedding-train|--mha-train|--cnn-train|--convt-train|--rnn-train]"
      echo "  env: CONVT=1 CNN=1 … MHA=1 for layer forward suites"
      exit 0
      ;;
  esac
done

if [[ "$MODE" == "dense_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalDenseTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "rmsnorm_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalRMSNormTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "layernorm_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalLayerNormTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "swiglu_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSwiGLUTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "softmax_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSoftmaxTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "embedding_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalEmbeddingTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "mha_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalMHATrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "cnn_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalCNNTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "convt_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalConvTTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "rnn_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalRNNTrain$' -timeout 60m
  exit 0
fi

# Default: Dense forward. Layer-only: … / SEQMIX / CNN.
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
