#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 Metal host (this scaffold) ---
      {"metal.device.init", "metal", "MTLDevice probe + runtime identity", true},
      {"metal.library.compile", "metal", "Runtime compile shaders/welvet_dense.metal", true},
      {"metal.buffer.shared", "metal", "Shared MTLBuffer alloc/upload/download", true},
      {"metal.pipeline.smoke", "metal", "Compute pipeline for welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_metal_init C ABI", true},
      {"platform.macos", "platform", "macOS Apple silicon host build", true},

      // --- Core / weights (Go FormatNone semantics; compute on Metal only) ---
      {"core.dtype.all34", "core", "34 DType + FormatNone encode/decode (staging)", true},
      {"core.activate", "core", "Activate f64 on Metal bias_act kernel (Go codes)", true},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack (FormatNone)", true},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34", true},
      {"weights.matvec.metal", "weights",
       "Metal soft-double GEMV (Apple has no HW double; Go f64 contract)", true},
      {"weights.resident", "weights", "gap? upload-once / reuse across fwd/bwd/update", false},
      {"metal.no_fallback", "metal", "no host MatVec / no f32-acc substitute", true},

      // --- Layers (same 23 kinds as welvet.cpp / Go) ---
      {"dense.forward.metal", "layers",
       "Dense Metal forward FormatNone×34 (soft-double GEMV + bias/act)", true},
      {"dense.normalbp.metal", "layers", "gap? Dense NormalBP dX/dW/db + SGD Metal", false},
      {"mha.forward.metal", "layers",
       "MHA ModeSelf forward (Metal Dense Q/K/V/O + host f64 attn; WVM1)", true},
      {"mha.normalbp.metal", "layers", "gap? MHA NormalBP Metal", false},
      {"swiglu.forward.metal", "layers",
       "SwiGLU forward (Metal Dense Gate/Up/Down + host SiLU⊙; WVS1)", true},
      {"swiglu.normalbp.metal", "layers", "gap? SwiGLU NormalBP Metal", false},
      {"rmsnorm.forward.metal", "layers",
       "RMSNorm forward FormatNone×34 host f64 (Go/cpp exact_bit; WVR1)", true},
      {"rmsnorm.normalbp.metal", "layers", "gap? RMSNorm NormalBP Metal", false},
      {"layernorm.metal", "layers", "gap? LayerNorm fwd+NormalBP Metal", false},
      {"softmax.metal", "layers", "gap? Softmax all Kind + dX Metal", false},
      {"embedding.metal", "layers", "gap? Embedding gather/scatter Metal", false},
      {"cnn.metal", "layers", "gap? CNN1/2/3 fwd+NormalBP Metal", false},
      {"convt.metal", "layers", "gap? ConvT1/2/3 fwd+NormalBP Metal", false},
      {"rnn.metal", "layers", "gap? RNN BPTT Metal", false},
      {"lstm.metal", "layers", "gap? LSTM BPTT Metal", false},
      {"sequential.metal", "layers", "gap? Sequential Dense chain Metal", false},
      {"residual.metal", "layers", "gap? Residual F(x)+x Metal", false},
      {"kmeans.metal", "layers", "gap? KMeans soft assignment Metal", false},
      {"mamba.metal", "layers", "gap? Mamba scan Metal", false},
      {"metacognition.metal", "layers", "gap? Metacognition Metal", false},
      {"gdn.metal", "layers", "gap? GDN f32 Metal", false},
      {"stack.metal", "layers", "gap? Stack NestedStack-of-Dense Metal", false},
      {"parallel.metal", "layers", "gap? Parallel branches + BranchModes Metal", false},
      {"seqmix.metal", "layers", "gap? seqmix helper Metal", false},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture", "gap? Grid topology + remote links Metal residency", false},
      {"runtime.forward", "runtime", "gap? volume forward Metal", false},
      {"runtime.backward", "runtime", "gap? volume backward Metal", false},
      {"runtime.step", "runtime", "gap? Line StepBP/StepTween* Metal", false},
      {"runtime.mesh", "runtime", "gap? MeshBP/MeshTween* Metal", false},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.camsync", "training", "gap? CamSync / BlendStores Metal", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "gap? Tween* projected credit Metal", false},
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

const std::vector<Entry>& all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (!e.implemented) ++n;
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (e.implemented) ++n;
  }
  return n;
}

void print_census(std::ostream& os) {
  os << "welvet.metal gaps: done=" << count_done() << " open=" << count_gaps() << "\n";
  for (const auto& e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
