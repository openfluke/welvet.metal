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
    --meshbp-train) MODE="meshbp_train" ;;
    --step-tween-all) MODE="step_tween_all" ;;
    --normalbp-all) MODE="normalbp_all" ;;
    --metal-only) MODE="metal_only" ;;
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
    --lstm-train) MODE="lstm_train" ;;
    --sequential-train) MODE="sequential_train" ;;
    --residual-train) MODE="residual_train" ;;
    --mamba-train) MODE="mamba_train" ;;
    --metacognition-train) MODE="metacognition_train" ;;
    --gdn-train) MODE="gdn_train" ;;
    --stack-train) MODE="stack_train" ;;
    --tween-next-all) MODE="tween_next_all" ;;
    --tweensplit-train) MODE="tweensplit_train" ;;
    --steptweensplit-train) MODE="steptweensplit_train" ;;
    --tweenalt-train) MODE="tweenalt_train" ;;
    --steptweenalt-train) MODE="steptweenalt_train" ;;
    --meshtween-train) MODE="meshtween_train" ;;
    --meshtweenchain-train) MODE="meshtweenchain_train" ;;
    --steptween-train) MODE="steptween_train" ;;
    --steptweenchain-train) MODE="steptweenchain_train" ;;
    --tweenchain-train) MODE="tweenchain_train" ;;
    --tween-train) MODE="tween_train" ;;
    --stepbp-train) MODE="stepbp_train" ;;
    --help|-h)
      echo "MeshBP: --meshbp-train (all-family clock + StepBP/volume + NormalBP + Metal-only checks)"
      echo "training modes 2–6: --step-tween-all (complete NormalBP + five requested modes + Metal-only checks)"
      echo "training batch: --tween-next-all (eight requested modes + Metal-only checks)"
      echo "usage: $0 [--normalbp-all|--metal-only|--dense-train|…|--stack-train|--stepbp-train|--tween-train|--tweenchain-train]"
      echo "  env: CONVT=1 CNN=1 … MHA=1 for layer forward suites"
      exit 0
      ;;
  esac
done

if [[ "$MODE" == "meshbp_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalMeshBPTrain|TestGoVMetalStepBPTrain|TestGoVMetalNormalBP|TestGoVMetalOnly)$' -timeout 45m
  exit 0
fi

if [[ "$MODE" == "step_tween_all" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalNormalBP|TestGoVMetalOnly|TestGoVMetalStepBPTrain|TestGoVMetalTweenTrain|TestGoVMetalTweenChainTrain|TestGoVMetalStepTweenTrain|TestGoVMetalStepTweenChainTrain)$' -timeout 45m
  exit 0
fi

if [[ "$MODE" == "normalbp_all" ]]; then
  # Correctness first: all cases/dtypes with one measured iteration per fixture.
  # Set WELVET_METAL_BENCH_ITERS=2000 for longer performance measurements.
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalNormalBP|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "metal_only" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalOnly$' -timeout 10m
  exit 0
fi

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
if [[ "$MODE" == "lstm_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalLSTMTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "sequential_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalSequentialTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "residual_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalResidualTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "mamba_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalMambaTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "metacognition_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalMetacognitionTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "gdn_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalGDNTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "stack_train" ]]; then
  go test ./tests/go_v_metal -v -count=1 -run '^TestGoVMetalStackTrain$' -timeout 60m
  exit 0
fi
if [[ "$MODE" == "tween_next_all" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetal(TweenSplit|StepTweenSplit|TweenAlt|StepTweenAlt|MeshTween|MeshTweenChain|StepTween|StepTweenChain)Train|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "tweensplit_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalTweenSplitTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "steptweensplit_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalStepTweenSplitTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "tweenalt_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalTweenAltTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "steptweenalt_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalStepTweenAltTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "meshtween_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalMeshTweenTrain|TestGoVMetalMeshBPTrain|TestGoVMetalTweenTrain|TestGoVMetalOnly)$' -timeout 45m
  exit 0
fi

if [[ "$MODE" == "meshtweenchain_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalMeshTweenChainTrain|TestGoVMetalMeshTweenTrain|TestGoVMetalTweenChainTrain|TestGoVMetalOnly)$' -timeout 45m
  exit 0
fi

if [[ "$MODE" == "steptween_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalStepTweenTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "steptweenchain_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalStepTweenChainTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "tweenchain_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalTweenChainTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "tween_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalTweenTrain|TestGoVMetalOnly)$' -timeout 30m
  exit 0
fi

if [[ "$MODE" == "stepbp_train" ]]; then
  export WELVET_METAL_BENCH_ITERS="${WELVET_METAL_BENCH_ITERS:-1}"
  go test ./tests/go_v_metal -v -count=1 -run '^(TestGoVMetalStepBPTrain|TestGoVMetalOnly)$' -timeout 30m
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
