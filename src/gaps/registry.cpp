#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry> &build() {
  static const std::vector<Entry> k = {
      // --- M0 Metal host (this scaffold) ---
      {"metal.device.init", "metal", "MTLDevice probe + runtime identity", true},
      {"metal.library.compile", "metal",
       "Runtime compile embedded Metal source (no checkout required)", true},
      {"metal.buffer.shared", "metal", "Shared MTLBuffer alloc/upload/download", true},
      {"metal.pipeline.smoke", "metal", "Compute pipeline for welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_metal_init C ABI", true},
      {"platform.macos", "platform", "macOS Apple silicon host build", true},

      // --- Core / weights (Go FormatNone semantics; compute on Metal only) ---
      {"core.dtype.all34", "core", "34 DType + FormatNone encode/decode (staging)", true},
      {"core.activate", "core", "Metal activation and derivative kernels; no host activation API",
       true},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack (FormatNone)", true},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34", true},
      {"weights.matvec.metal", "weights",
       "Metal soft-double GEMV (Apple has no HW double; Go f64 contract)", true},
      {"weights.resident", "weights", "gap? upload-once / reuse across fwd/bwd/update", false},
      {"metal.no_fallback", "metal",
       "CPU numerical layer paths rejected, including direct APIs; no runtime opt-out", true},

      // --- Layers (same 23 kinds as welvet.cpp / Go) ---
      {"dense.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"dense.normalbp.metal", "layers",
       "Metal MSE/NormalBP and native FormatNone SGD across all 34 dtypes", true},
      {"dense.bwd.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"mha.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"mha.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"swiglu.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"swiglu.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"rmsnorm.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"rmsnorm.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"layernorm.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"layernorm.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"softmax.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"softmax.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"embedding.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"embedding.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"cnn1.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"cnn2.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"cnn3.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"cnn.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"convt1.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"convt2.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"convt3.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"convt.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"rnn.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"rnn.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"lstm.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"lstm.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"sequential.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"sequential.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"residual.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"residual.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"kmeans.forward.metal", "layers", "Metal kernel pending; CPU execution is forbidden", false},
      {"kmeans.normalbp.metal", "layers", "Metal kernel pending; CPU execution is forbidden",
       false},
      {"mamba.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"mamba.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"metacognition.forward.metal", "layers",
       "Metal numerical kernels; host scheduling and storage staging only", true},
      {"metacognition.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"gdn.forward.metal", "layers", "Metal f32 truncated-BPTT, state/conv stop-grad, GPU SGD",
       true},
      {"gdn.normalbp.metal", "layers", "Metal f32 truncated-BPTT, state/conv stop-grad, GPU SGD",
       true},
      {"stack.metal", "layers", "Metal numerical kernels; host scheduling and storage staging only",
       true},
      {"stack.normalbp.metal", "layers",
       "Metal forward/backward kernels; native FormatNone SGD on GPU", true},
      {"parallel.forward.metal", "layers", "Metal kernel pending; CPU execution is forbidden",
       false},
      {"parallel.normalbp.metal", "layers", "Metal kernel pending; CPU execution is forbidden",
       false},
      {"seqmix.metal", "layers",
       "seqmix Kind/Contract helper + MHA/Mamba/GDN tags + WSX1 smoke (no Forward)", true},
      {"seqmix.conv_mix.metal", "layers", "gap? seqmix KindConvMix implementor Metal", false},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture",
       "Grid hop_order + remote links + bind_* (WVF1/WMV1/WRF1/WLF1/WSF1/WPF1/WKF1/WSG1/SQG1…MCG1)",
       true},
      {"runtime.forward", "runtime",
       "Volume scheduling dispatches supported Metal layers; others return Unsupported", true},
      {"runtime.forward.volume_wires", "runtime",
       "Volume wires retained; availability follows per-layer Metal capability", true},
      {"runtime.backward", "runtime", "Reverse tape dispatch for supported Metal layers only",
       true},
      {"train.normalbp.dense", "training",
       "Metal MSE/NormalBP and native FormatNone SGD across all 34 dtypes", true},
      {"runtime.step", "runtime",
       "Dense/Sequential/NestedStack Float32 Line FamilyBP; other layers/modes unsupported", true},
      {"runtime.mesh", "runtime",
       "Dense Float32 MeshBP; Metal gradient accumulation; host scheduling", true},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.camsync", "training", "gap? CamSync / BlendStores Metal", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "ModeTween projected local credit on Metal", true},
      {"systems.tween_chain", "systems", "ModeTweenChain full-rate chain-rule BP on Metal", true},
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tanhi", "systems", "gap? TANHI", false},
      {"systems.telemetry", "systems", "gap? telemetry", false},
      {"model.tokenizer", "model", "gap? tokenizer", false},
      {"model.entity", "model", "gap? ENTITY", false},
      {"model.transformer", "model", "gap? transformer", false},
      {"model.sampling", "model", "gap? sampling", false},
      {"model.hf", "model", "gap? HF load", false},
      {"lucy.lpd", "lucy", "gap? Lucy/LPD", false},

      // --- Delivery / validation ---
      {"platform.ios", "platform", "gap? iOS device framework artifact", false},
      {"accel.ane", "backends", "gap? Apple Neural Engine / Core ML path", false},
      {"w2a.go_v_metal", "interop", "gap? w2a Go_v_Metal suites (parity + speed)", false},
      {"nac.go_delegate", "interop", "gap? Go Welvet NAC → Metal dispatch", false},
      {"determinism.policy", "metal", "gap? documented reduction order + exact_bit bands", false},
  };
  return k;
}

} // namespace

const std::vector<Entry> &all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto &e : all()) {
    if (!e.implemented)
      ++n;
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto &e : all()) {
    if (e.implemented)
      ++n;
  }
  return n;
}

void print_census(std::ostream &os) {
  os << "welvet.metal gaps: done=" << count_done() << " open=" << count_gaps() << "\n";
  for (const auto &e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) { return std::string("gap? ") + std::string(what); }

} // namespace welvet::gaps
