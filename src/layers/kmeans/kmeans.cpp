#include "welvet/kmeans.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::kmeans {
namespace {

struct LayInfo {
  int batch = 0;
  int d = 0;
  int k = 0;
  std::vector<int64_t> shape;
};

// Go softMax — float64 Exp (not Softmax layer's float32 path).

} // namespace

int out_dim(const Config& cfg) {
  return cfg.output_mode == OutputMode::Features ? cfg.feature_dim : cfg.num_clusters;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil config");
  if (cfg->num_clusters <= 0 || cfg->feature_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: need positive NumClusters/FeatureDim");
  }
  if (cfg->temperature < 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: Temperature < 0");
  }
  if (cfg->output_mode != OutputMode::Probabilities && cfg->output_mode != OutputMode::Features) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: unknown OutputMode");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* init_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  // dense::new_layer(in_h=D, out_h=K) → Weights Rows=K, Cols=D
  if (auto st = dense::new_layer(cfg.feature_dim, cfg.num_clusters, Activation::Linear, dt,
                                 init_or_null, &layer.centers);
      !st) {
    return Status::Err(st.code, std::string("kmeans centers: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  return Status::Err(StatusCode::Unsupported, "kmeans.forward: Metal kernel not implemented; CPU execution is forbidden");
}

} // namespace welvet::kmeans
