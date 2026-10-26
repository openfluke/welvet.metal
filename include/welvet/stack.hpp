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
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/status.hpp"
#include "welvet/swiglu.hpp"
#include "welvet/tensor.hpp"

#include <cstdint>
#include <vector>

namespace welvet::stack {

// Go parallel.splitAccel — LinearCache / HeadProxyAsync / Sparse scratch.
struct SplitAccel {
  std::vector<std::vector<float>> cache;
  double cache_norm = 0.0; // legacy metadata; numerical norm lives in GPU-produced norm_xy
  std::vector<float> norm_xy;
  int tick = 0;
  int cache_every = 0; // 0 ⇒ default 20 (Go defaultLinearCacheEvery)
  std::vector<float> proxy; // HeadProxyAsync
  int sparse = 0;
};

// Matches architecture::OpKind values (Dense=1 … KMeans=22; Parallel=23).
enum class Kind : int32_t {
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
  NestedStack = 19,
  Metacognition = 20,
  GDN = 21,
  KMeans = 22,
  Parallel = 23, // nested Parallel-as-branch (nested = branches; dense_op = optional Filter gate)
};

// POD Parallel nest header (avoids circular include of parallel::Layer).
struct ParallelNest {
  int32_t combine = 1; // parallel::Combine
  int dim = 0;
  int out_feat = 0;
  int seq_len = 0;
  int sparse_k = 0;
  double disagree_beta = 0;
  int has_gate = 0; // when 1, dense_op holds Filter gate
};

// Thin heterogeneous chain host (Go parallel.Stack subset — no CamSync / BranchModes).
struct Child {
  Kind kind = Kind::None;
  SplitAccel accel{};
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
  metacognition::Layer metacognition_op;
  gdn::Layer gdn_op;
  kmeans::Layer kmeans_op;
  std::vector<Child> nested; // NestedStack children, or Parallel branches
  ParallelNest parallel_nest{}; // when kind == Parallel
};

struct Layer {
  std::vector<Child> children;
  LayerMeta meta{};
  SplitAccel accel{};
};

// Flatten NestedStack into leaf Child* order (Go flattenLineOps).
void flatten_ops(Layer& root, std::vector<Child*>* out);
void flatten_ops(Child& c, std::vector<Child*>* out);

Status child_forward(Child& c, const TensorF32& input, TensorF32* pre, TensorF32* post);
Status child_backward(Child& c, const TensorF32& grad_out, const TensorF32& input,
                      const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);
Status child_apply_sgd(Child* c, const TensorF32& dW, double lr);
int child_grad_w_size(const Child& c);
Status child_export_f32(const Child& c, float* dst);

// Full-chain NormalBP (volume cell host): pre/post = last leaf pre/post.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);
int grad_w_size(const Layer& layer);
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);
Status export_packed_f32(const Layer& layer, float* dst);

// FamilyBP credit on one leaf: Backward + ApplyGradSGD → grad_in.
Status child_family_bp(Child& c, const TensorF32& grad_out, const TensorF32& input,
                       const TensorF32& pre, double lr, TensorF32* grad_in);

} // namespace welvet::stack
