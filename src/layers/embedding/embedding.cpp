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

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: empty input");
  }
  lay->emb = cfg.embedding_dim;
  lay->vocab = cfg.vocab_size;
  const auto& sh = input.shape;
  if (sh.size() == 1) {
    lay->batch = 1;
    lay->seq = static_cast<int>(sh[0]);
    lay->in_rank = 1;
    lay->in_last = 1;
  } else if (sh.size() == 2) {
    lay->batch = static_cast<int>(sh[0]);
    lay->seq = static_cast<int>(sh[1]);
    lay->in_rank = 2;
    lay->in_last = 1;
  } else if (sh.size() == 3) {
    lay->batch = static_cast<int>(sh[0]);
    lay->seq = static_cast<int>(sh[1]);
    lay->in_rank = 3;
    lay->in_last = static_cast<int>(sh[2]);
    if (lay->in_last != 1 && lay->in_last != cfg.embedding_dim) {
      return Status::Err(StatusCode::InvalidArgument, "embedding: input last dim want 1 or emb");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument, "embedding: input rank want 1..3");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: bad batch/seq");
  }
  lay->n_tok = lay->batch * lay->seq;
  const int want = lay->n_tok * lay->in_last;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: input short");
  }
  return Status::Ok();
}

int token_id(const TensorF32& input, const Layout& lay, int tok) {
  double v = 0.0;
  if (lay.in_rank == 3 && lay.in_last > 1) {
    v = static_cast<double>(input.data[static_cast<std::size_t>(tok * lay.in_last)]);
  } else {
    v = static_cast<double>(input.data[static_cast<std::size_t>(tok)]);
  }
  int id = static_cast<int>(v);
  if (id < 0) id = -id;
  if (lay.vocab > 0) id = id % lay.vocab;
  return id;
}

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
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: nil pre/post");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;

  if (lay.in_rank == 1) {
    pre->shape = {lay.seq, lay.emb};
  } else {
    pre->shape = {lay.batch, lay.seq, lay.emb};
  }
  const int n = lay.n_tok * lay.emb;
  pre->data.assign(static_cast<std::size_t>(n), 0.0f);

  std::vector<double> row(static_cast<std::size_t>(lay.emb));
  for (int t = 0; t < lay.n_tok; ++t) {
    const int id = token_id(input, lay, t);
    if (auto st = layer.weights.decode_row_f64(id, row.data()); !st) {
      return Status::Err(st.code, std::string("embedding: token OOB/decode: ") + st.message);
    }
    const int ob = t * lay.emb;
    for (int j = 0; j < lay.emb; ++j) {
      pre->data[static_cast<std::size_t>(ob + j)] =
          static_cast<float>(row[static_cast<std::size_t>(j)]);
    }
  }

  if (layer.activation != Activation::Linear) {
    for (int i = 0; i < n; ++i) {
      const double v = activate_f64(static_cast<double>(pre->data[static_cast<std::size_t>(i)]),
                                    layer.activation);
      pre->data[static_cast<std::size_t>(i)] = static_cast<float>(v);
    }
  }

  *post = *pre;
  return Status::Ok();
}

Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: nil grad_in/grad_w");
  }
  if (auto st = validate(layer.cfg); !st) return st;

  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;
  if (static_cast<int>(grad_out.data.size()) < lay.n_tok * lay.emb) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: gradOut short");
  }

  const int emb = lay.emb;
  const int nw = lay.vocab * emb;
  grad_w->shape = {lay.vocab, emb};
  grad_w->data.assign(static_cast<std::size_t>(nw), 0.0f);

  for (int t = 0; t < lay.n_tok; ++t) {
    const int id = token_id(input, lay, t);
    const int ob = t * emb;
    const int rb = id * emb;
    if (rb + emb > nw) {
      return Status::Err(StatusCode::InvalidArgument, "embedding: token OOB in backward");
    }
    for (int j = 0; j < emb; ++j) {
      grad_w->data[static_cast<std::size_t>(rb + j)] +=
          grad_out.data[static_cast<std::size_t>(ob + j)];
    }
  }

  // zeros — no gradient through discrete token IDs
  grad_in->shape = input.shape;
  grad_in->data.assign(input.data.size(), 0.0f);
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: ApplyGradSGD nil");
  }
  const int n = layer->weights.rows * layer->weights.cols;
  if (static_cast<int>(dW.data.size()) < n) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: dW short");
  }
  std::vector<double> buf(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    buf[static_cast<std::size_t>(i)] = static_cast<double>(dW.data[static_cast<std::size_t>(i)]);
  }
  return layer->weights.apply_sgd(buf.data(), lr);
}

} // namespace welvet::embedding
