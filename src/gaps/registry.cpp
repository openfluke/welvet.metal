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
      {"dense.normalbp.metal", "layers",
       "Dense NormalBP resident soft-double W + host MatVecT/SGD FormatNone (WDB1/WVN1)", true},
      {"mha.forward.metal", "layers",
       "MHA ModeSelf forward (Metal Dense Q/K/V/O + host f64 attn; WVM1)", true},
      {"mha.normalbp.metal", "layers",
       "MHA NormalBP resident Dense Q/K/V/O + host attn/RoPEᵀ/MatVecT (WHB1/WHV1)", true},
      {"swiglu.forward.metal", "layers",
       "SwiGLU forward (Metal Dense Gate/Up/Down + host SiLU⊙; WVS1)", true},
      {"swiglu.normalbp.metal", "layers",
       "SwiGLU NormalBP resident Gate/Up/Down + host SiLU′/MatVecT (WSB1/WSV1)", true},
      {"rmsnorm.forward.metal", "layers",
       "RMSNorm forward FormatNone×34 host f64 (Go/cpp exact_bit; WVR1)", true},
      {"rmsnorm.normalbp.metal", "layers",
       "RMSNorm NormalBP dX/dγ + SGD (host f64; WRB1/WRV1)", true},
      {"layernorm.forward.metal", "layers",
       "LayerNorm forward FormatNone×34 host f64 (Go/cpp exact_bit; WVL1)", true},
      {"layernorm.normalbp.metal", "layers",
       "LayerNorm NormalBP dX/dγβ + SGD (host f64; WLB1/WLV1)", true},
      {"softmax.forward.metal", "layers",
       "Softmax all Kind FormatNone×34 host (Go/cpp exact_bit; WSM1)", true},
      {"softmax.normalbp.metal", "layers",
       "Softmax NormalBP dX-only (host Jacobian; WXB1/WXV1; all Kind)", true},
      {"embedding.forward.metal", "layers",
       "Embedding gather FormatNone×34 decode_row_f64 (Go/cpp exact_bit; WEM1)", true},
      {"embedding.normalbp.metal", "layers",
       "Embedding NormalBP dX=0 + dW scatter + SGD (WEB1/WEV1)", true},
      {"cnn1.forward.metal", "layers",
       "CNN1 im2col→Metal Dense FormatNone×34→loom (WVC1)", true},
      {"cnn2.forward.metal", "layers",
       "CNN2 im2col→Metal Dense FormatNone×34→loom (WVC2)", true},
      {"cnn3.forward.metal", "layers",
       "CNN3 im2col→Metal Dense FormatNone×34→loom (WVC3)", true},
      {"cnn.normalbp.metal", "layers", "gap? CNN1/2/3 NormalBP Metal", false},
      {"convt1.forward.metal", "layers",
       "ConvT1 host f64 scatter FormatNone×34 (WVT1 exact_bit)", true},
      {"convt2.forward.metal", "layers",
       "ConvT2 host f64 scatter FormatNone×34 (WVT2 exact_bit)", true},
      {"convt3.forward.metal", "layers",
       "ConvT3 host f64 scatter FormatNone×34 (WVT3 exact_bit)", true},
      {"convt.normalbp.metal", "layers", "gap? ConvT1/2/3 NormalBP Metal", false},
      {"rnn.forward.metal", "layers",
       "RNN Dense soft-double + host f64 tanh FormatNone×34 (WRN1)", true},
      {"rnn.normalbp.metal", "layers", "gap? RNN BPTT Metal", false},
      {"lstm.forward.metal", "layers",
       "LSTM Dense soft-double + host f64 gates FormatNone×34 (WLS1)", true},
      {"lstm.normalbp.metal", "layers", "gap? LSTM BPTT Metal", false},
      {"sequential.forward.metal", "layers",
       "Sequential Dense chain FormatNone×34 via Metal soft-double (WSQ1)", true},
      {"sequential.normalbp.metal", "layers", "gap? Sequential NormalBP Metal", false},
      {"residual.forward.metal", "layers",
       "Residual F(x)+x FormatNone×34 via Metal soft-double Dense + f64 skip-add (WRE1)", true},
      {"residual.normalbp.metal", "layers", "gap? Residual NormalBP Metal", false},
      {"kmeans.forward.metal", "layers",
       "KMeans soft assign FormatNone×34 decode_row_f64 + host f64 softMax (WKM1)", true},
      {"kmeans.normalbp.metal", "layers", "gap? KMeans NormalBP Metal", false},
      {"mamba.forward.metal", "layers",
       "Mamba Dense soft-double In/Out + host f64 scan FormatNone×34 (WMB1)", true},
      {"mamba.normalbp.metal", "layers", "gap? Mamba BPTT Metal", false},
      {"metacognition.forward.metal", "layers",
       "Metacognition Observed Dense soft-double + host rules WMC1", true},
      {"metacognition.normalbp.metal", "layers", "gap? Metacognition NormalBP Metal", false},
      {"gdn.forward.metal", "layers",
       "GDN host f32 FormatNone forward exact_bit vs Go (WGD1)", true},
      {"gdn.normalbp.metal", "layers", "gap? GDN truncated-BPTT Metal", false},
      {"stack.metal", "layers", "gap? Stack NestedStack-of-Dense Metal", false},
      {"parallel.forward.metal", "layers",
       "Parallel Dense NewConfigured soft-double + host combine WPL1 (7 modes)", true},
      {"parallel.normalbp.metal", "layers", "gap? Parallel NormalBP / BranchModes Metal", false},
      {"seqmix.metal", "layers",
       "seqmix Kind/Contract helper + MHA/Mamba/GDN tags + WSX1 smoke (no Forward)", true},
      {"seqmix.conv_mix.metal", "layers", "gap? seqmix KindConvMix implementor Metal", false},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture",
       "Grid hop_order + remote links + bind_* (WVF1/WMV1/WRF1/WLF1/WSF1/WPF1/WKF1/WSG1/SQG1…MCG1)", true},
      {"runtime.forward", "runtime", "volume_dense hop/remote/posts + cell() all Metal layers", true},
      {"runtime.forward.volume_wires", "runtime",
       "volume mount-all WVB2 (Dense/MHA/RMS/LN/Softmax/Parallel/KMeans/SwiGLU/Seq/Res/RNN/LSTM/CNN*/ConvT*/Emb/Mamba/GDN/Meta)", true},
      {"runtime.backward", "runtime",
       "volume_dense reverse tape + Dense cell bwd (other kinds still gap)", true},
      {"train.normalbp.dense", "training",
       "MSE/MSEGrad + step_dense Dense NormalBP (WDB1/WVN1)", true},
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
