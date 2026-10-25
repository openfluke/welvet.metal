#pragma once

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/gdn.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layer.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mha.hpp"
#include "welvet/parallel.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/residual.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/swiglu.hpp"

#include <cstdint>
#include <vector>

namespace welvet::architecture {

struct Coord {
  int z = 0, y = 0, x = 0, l = 0;
  bool operator==(const Coord& o) const {
    return z == o.z && y == o.y && x == o.x && l == o.l;
  }
};

enum class OpKind : int32_t {
  None = 0,
  Dense = 1,
  RMSNorm = 2,
  LayerNorm = 3,
  SwiGLU = 4,
  Softmax = 5,
  Embedding = 6,
  MHA = 7,
  CNN1 = 8,
  CNN2 = 9,
  CNN3 = 10,
  ConvT1 = 11,
  ConvT2 = 12,
  ConvT3 = 13,
  RNN = 14,
  LSTM = 15,
  Sequential = 16,
  Residual = 17,
  Mamba = 18,
  Stack = 19,
  Metacognition = 20,
  GDN = 21,
  KMeans = 22,
  Parallel = 23,
};

struct Cell {
  LayerMeta meta{};
  OpKind op_kind = OpKind::None;
  dense::Layer dense_op;
  rmsnorm::Layer rmsnorm_op;
  layernorm::Layer layernorm_op;
  swiglu::Layer swiglu_op;
  softmax::Layer softmax_op;
  embedding::Layer embedding_op;
  mha::Layer mha_op;
  cnn1::Layer cnn1_op;
  cnn2::Layer cnn2_op;
  cnn3::Layer cnn3_op;
  convt1::Layer convt1_op;
  convt2::Layer convt2_op;
  convt3::Layer convt3_op;
  rnn::Layer rnn_op;
  lstm::Layer lstm_op;
  sequential::Layer sequential_op;
  residual::Layer residual_op;
  mamba::Layer mamba_op;
  stack::Layer stack_op;
  metacognition::Layer metacognition_op;
  gdn::Layer gdn_op;
  kmeans::Layer kmeans_op;
  parallel::Layer parallel_op;

  bool is_remote_link = false;
  int target_z = 0, target_y = 0, target_x = 0, target_l = 0;
};

struct Grid {
  int depth = 1;
  int rows = 1;
  int cols = 1;
  int layers_per_cell = 1;
  std::vector<Cell> cells;
};

Grid new_grid(int depth, int rows, int cols, int layers_per_cell);
int index(const Grid& g, int z, int y, int x, int l);
Cell* at(Grid& g, int z, int y, int x, int l);
const Cell* at(const Grid& g, int z, int y, int x, int l);
inline int stack_layer_count(const Grid& g) {
  return g.depth * g.rows * g.cols * g.layers_per_cell;
}
std::vector<Coord> hop_order(const Grid& g);
Status set_remote_link(Grid& g, int z, int y, int x, int l, int tz, int ty, int tx, int tl);

Status bind_dense(Grid& g, int z, int y, int x, int l, dense::Layer op);
Status bind_rmsnorm(Grid& g, int z, int y, int x, int l, rmsnorm::Layer op);
Status bind_layernorm(Grid& g, int z, int y, int x, int l, layernorm::Layer op);
Status bind_swiglu(Grid& g, int z, int y, int x, int l, swiglu::Layer op);
Status bind_softmax(Grid& g, int z, int y, int x, int l, softmax::Layer op);
Status bind_embedding(Grid& g, int z, int y, int x, int l, embedding::Layer op);
Status bind_mha(Grid& g, int z, int y, int x, int l, mha::Layer op);
Status bind_cnn1(Grid& g, int z, int y, int x, int l, cnn1::Layer op);
Status bind_cnn2(Grid& g, int z, int y, int x, int l, cnn2::Layer op);
Status bind_cnn3(Grid& g, int z, int y, int x, int l, cnn3::Layer op);
Status bind_convt1(Grid& g, int z, int y, int x, int l, convt1::Layer op);
Status bind_convt2(Grid& g, int z, int y, int x, int l, convt2::Layer op);
Status bind_convt3(Grid& g, int z, int y, int x, int l, convt3::Layer op);
Status bind_rnn(Grid& g, int z, int y, int x, int l, rnn::Layer op);
Status bind_lstm(Grid& g, int z, int y, int x, int l, lstm::Layer op);
Status bind_sequential(Grid& g, int z, int y, int x, int l, sequential::Layer op);
Status bind_residual(Grid& g, int z, int y, int x, int l, residual::Layer op);
Status bind_mamba(Grid& g, int z, int y, int x, int l, mamba::Layer op);
Status bind_stack(Grid& g, int z, int y, int x, int l, stack::Layer op);
Status bind_metacognition(Grid& g, int z, int y, int x, int l, metacognition::Layer op);
Status bind_gdn(Grid& g, int z, int y, int x, int l, gdn::Layer op);
Status bind_kmeans(Grid& g, int z, int y, int x, int l, kmeans::Layer op);
Status bind_parallel(Grid& g, int z, int y, int x, int l, parallel::Layer op);

} // namespace welvet::architecture
