#include "welvet/architecture.hpp"

#include <sstream>

namespace welvet::architecture {
namespace {

Status bad_coord(const char* which, int z, int y, int x, int l) {
  std::ostringstream oss;
  oss << "architecture: bad " << which << " coord z=" << z << " y=" << y << " x=" << x
      << " l=" << l;
  return Status::Err(StatusCode::InvalidArgument, oss.str());
}

template <typename BindFn>
Status bind_at(Grid& g, int z, int y, int x, int l, LayerType type, OpKind kind, BindFn&& fn) {
  Cell* c = at(g, z, y, x, l);
  if (!c) return bad_coord("bind", z, y, x, l);
  c->meta.type = type;
  c->meta.z = z;
  c->meta.y = y;
  c->meta.x = x;
  c->meta.l = l;
  c->op_kind = kind;
  fn(*c);
  return Status::Ok();
}

} // namespace

Grid new_grid(int depth, int rows, int cols, int layers_per_cell) {
  if (depth < 1) depth = 1;
  if (rows < 1) rows = 1;
  if (cols < 1) cols = 1;
  if (layers_per_cell < 1) layers_per_cell = 1;

  Grid g;
  g.depth = depth;
  g.rows = rows;
  g.cols = cols;
  g.layers_per_cell = layers_per_cell;
  g.cells.resize(static_cast<std::size_t>(depth * rows * cols * layers_per_cell));

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < rows; ++y) {
      for (int x = 0; x < cols; ++x) {
        for (int l = 0; l < layers_per_cell; ++l) {
          Cell& c = g.cells[static_cast<std::size_t>(index(g, z, y, x, l))];
          c.meta.type = LayerType::Dense;
          c.meta.dtype = DType::Float32;
          c.meta.activation = Activation::ReLU;
          c.meta.z = z;
          c.meta.y = y;
          c.meta.x = x;
          c.meta.l = l;
          c.op_kind = OpKind::None;
        }
      }
    }
  }
  return g;
}

int index(const Grid& g, int z, int y, int x, int l) {
  return (((z * g.rows + y) * g.cols + x) * g.layers_per_cell + l);
}

Cell* at(Grid& g, int z, int y, int x, int l) {
  if (z < 0 || z >= g.depth || y < 0 || y >= g.rows || x < 0 || x >= g.cols || l < 0 ||
      l >= g.layers_per_cell) {
    return nullptr;
  }
  const int idx = index(g, z, y, x, l);
  if (idx < 0 || idx >= static_cast<int>(g.cells.size())) return nullptr;
  return &g.cells[static_cast<std::size_t>(idx)];
}

const Cell* at(const Grid& g, int z, int y, int x, int l) {
  return at(const_cast<Grid&>(g), z, y, x, l);
}

std::vector<Coord> hop_order(const Grid& g) {
  std::vector<Coord> out;
  out.reserve(static_cast<std::size_t>(stack_layer_count(g)));
  for (int z = 0; z < g.depth; ++z) {
    for (int y = 0; y < g.rows; ++y) {
      for (int x = 0; x < g.cols; ++x) {
        for (int l = 0; l < g.layers_per_cell; ++l) {
          out.push_back(Coord{z, y, x, l});
        }
      }
    }
  }
  return out;
}

Status set_remote_link(Grid& g, int z, int y, int x, int l, int tz, int ty, int tx, int tl) {
  Cell* c = at(g, z, y, x, l);
  if (!c) return bad_coord("source", z, y, x, l);
  if (!at(g, tz, ty, tx, tl)) return bad_coord("target", tz, ty, tx, tl);
  c->is_remote_link = true;
  c->target_z = tz;
  c->target_y = ty;
  c->target_x = tx;
  c->target_l = tl;
  return Status::Ok();
}

Status bind_dense(Grid& g, int z, int y, int x, int l, dense::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Dense, OpKind::Dense, [&](Cell& c) {
    c.meta.dtype = op.weights.dtype;
    c.meta.activation = op.activation;
    c.meta.input_height = op.input_height;
    c.meta.output_height = op.output_height;
    c.dense_op = std::move(op);
  });
}

Status bind_rmsnorm(Grid& g, int z, int y, int x, int l, rmsnorm::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::RMSNorm, OpKind::RMSNorm,
                 [&](Cell& c) { c.rmsnorm_op = std::move(op); });
}
Status bind_layernorm(Grid& g, int z, int y, int x, int l, layernorm::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::LayerNorm, OpKind::LayerNorm,
                 [&](Cell& c) { c.layernorm_op = std::move(op); });
}
Status bind_swiglu(Grid& g, int z, int y, int x, int l, swiglu::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::SwiGLU, OpKind::SwiGLU,
                 [&](Cell& c) { c.swiglu_op = std::move(op); });
}
Status bind_softmax(Grid& g, int z, int y, int x, int l, softmax::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Softmax, OpKind::Softmax,
                 [&](Cell& c) { c.softmax_op = std::move(op); });
}
Status bind_embedding(Grid& g, int z, int y, int x, int l, embedding::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Embedding, OpKind::Embedding,
                 [&](Cell& c) { c.embedding_op = std::move(op); });
}
Status bind_mha(Grid& g, int z, int y, int x, int l, mha::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::MultiHeadAttention, OpKind::MHA,
                 [&](Cell& c) { c.mha_op = std::move(op); });
}
Status bind_cnn1(Grid& g, int z, int y, int x, int l, cnn1::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::CNN1, OpKind::CNN1,
                 [&](Cell& c) { c.cnn1_op = std::move(op); });
}
Status bind_cnn2(Grid& g, int z, int y, int x, int l, cnn2::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::CNN2, OpKind::CNN2,
                 [&](Cell& c) { c.cnn2_op = std::move(op); });
}
Status bind_cnn3(Grid& g, int z, int y, int x, int l, cnn3::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::CNN3, OpKind::CNN3,
                 [&](Cell& c) { c.cnn3_op = std::move(op); });
}
Status bind_convt1(Grid& g, int z, int y, int x, int l, convt1::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::ConvTransposed1D, OpKind::ConvT1,
                 [&](Cell& c) { c.convt1_op = std::move(op); });
}
Status bind_convt2(Grid& g, int z, int y, int x, int l, convt2::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::ConvTransposed2D, OpKind::ConvT2,
                 [&](Cell& c) { c.convt2_op = std::move(op); });
}
Status bind_convt3(Grid& g, int z, int y, int x, int l, convt3::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::ConvTransposed3D, OpKind::ConvT3,
                 [&](Cell& c) { c.convt3_op = std::move(op); });
}
Status bind_rnn(Grid& g, int z, int y, int x, int l, rnn::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::RNN, OpKind::RNN,
                 [&](Cell& c) { c.rnn_op = std::move(op); });
}
Status bind_lstm(Grid& g, int z, int y, int x, int l, lstm::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::LSTM, OpKind::LSTM,
                 [&](Cell& c) { c.lstm_op = std::move(op); });
}
Status bind_sequential(Grid& g, int z, int y, int x, int l, sequential::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Sequential, OpKind::Sequential,
                 [&](Cell& c) { c.sequential_op = std::move(op); });
}
Status bind_residual(Grid& g, int z, int y, int x, int l, residual::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Residual, OpKind::Residual,
                 [&](Cell& c) { c.residual_op = std::move(op); });
}
Status bind_mamba(Grid& g, int z, int y, int x, int l, mamba::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Mamba, OpKind::Mamba,
                 [&](Cell& c) { c.mamba_op = std::move(op); });
}
Status bind_metacognition(Grid& g, int z, int y, int x, int l, metacognition::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Metacognition, OpKind::Metacognition, [&](Cell& c) {
    c.meta.input_height = op.cfg.dim;
    c.meta.output_height = op.cfg.dim;
    c.metacognition_op = std::move(op);
  });
}
Status bind_gdn(Grid& g, int z, int y, int x, int l, gdn::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::GDN, OpKind::GDN,
                 [&](Cell& c) { c.gdn_op = std::move(op); });
}
Status bind_kmeans(Grid& g, int z, int y, int x, int l, kmeans::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::KMeans, OpKind::KMeans,
                 [&](Cell& c) { c.kmeans_op = std::move(op); });
}
Status bind_parallel(Grid& g, int z, int y, int x, int l, parallel::Layer op) {
  return bind_at(g, z, y, x, l, LayerType::Parallel, OpKind::Parallel,
                 [&](Cell& c) { c.parallel_op = std::move(op); });
}

} // namespace welvet::architecture
