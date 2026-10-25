#include "welvet/embedding.hpp"

#include "welvet/activate.hpp"

#include <string>
#include <vector>

namespace welvet::embedding {
namespace {

struct Layout {
  int batch = 0;
  int seq = 0;
  int emb = 0;
  int vocab = 0;
  int n_tok = 0;
  int in_rank = 0;
  int in_last = 0;
};

} // namespace

int weight_count(const Config& cfg) { return cfg.vocab_size * cfg.embedding_dim; }

Status validate(const Config& cfg) {
  if (cfg.vocab_size <= 0 || cfg.embedding_dim <= 0 || cfg.seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument,
                       "embedding: need positive VocabSize/EmbeddingDim/SeqLen");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg, DType dt, Activation act, const float* init_or_null,
                      Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "embedding: nil out");
  if (auto st = validate(cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  layer.activation = act;
  if (auto st =
          WeightStore::make(cfg.vocab_size, cfg.embedding_dim, dt, init_or_null, &layer.weights);
      !st) {
    return Status::Err(st.code, std::string("embedding weights: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "embedding.forward: Metal kernel not implemented; CPU execution is forbidden");
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  return Status::Err(StatusCode::Unsupported, "embedding.backward_f32: Metal kernel not implemented; CPU execution is forbidden");
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  return Status::Err(StatusCode::Unsupported, "embedding.apply_grad_sgd: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::embedding
