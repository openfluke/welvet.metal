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

Status parse_in(const Config& cfg, const TensorF32& input, LayInfo* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: empty input");
  }
  const int d = cfg.feature_dim;
  if (input.shape.empty() || static_cast<int>(input.shape.back()) != d) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: last dim want FeatureDim");
  }
  int batch = 1;
  for (std::size_t i = 0; i + 1 < input.shape.size(); ++i) {
    batch *= static_cast<int>(input.shape[i]);
  }
  if (batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: invalid batch");
  }
  const int want = batch * d;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: data short");
  }
  lay->batch = batch;
  lay->d = d;
  lay->k = cfg.num_clusters;
  lay->shape = input.shape;
  return Status::Ok();
}

Status flatten_w_f32(const WeightStore& ws, std::vector<float>* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil w");
  out->resize(static_cast<std::size_t>(ws.rows * ws.cols));
  std::vector<double> row(static_cast<std::size_t>(ws.cols));
  for (int r = 0; r < ws.rows; ++r) {
    if (auto st = ws.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < ws.cols; ++c) {
      (*out)[static_cast<std::size_t>(r * ws.cols + c)] =
          static_cast<float>(row[static_cast<std::size_t>(c)]);
    }
  }
  return Status::Ok();
}

// Go softMax — float64 Exp (not Softmax layer's float32 path).
std::vector<double> soft_max(const std::vector<double>& logits) {
  double max_z = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_z) max_z = logits[i];
  }
  std::vector<double> out(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double e = std::exp(logits[i] - max_z);
    out[i] = e;
    sum += e;
  }
  for (double& v : out) {
    v /= sum;
  }
  return out;
}

double temp_of(const Config& cfg) {
  return cfg.temperature == 0.0 ? 1.0 : cfg.temperature;
}

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
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  LayInfo lay;
  if (auto st = parse_in(cfg, input, &lay); !st) return st;

  std::vector<float> w;
  if (auto st = flatten_w_f32(layer.centers.weights, &w); !st) {
    return Status::Err(st.code, std::string("kmeans weights: ") + st.message);
  }
  if (static_cast<int>(w.size()) < lay.k * lay.d) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: weight short");
  }

  const double temp = temp_of(cfg);
  const double temp_sq = temp * temp;
  const int od = out_dim(cfg);

  pre->shape = lay.shape;
  if (!pre->shape.empty()) {
    pre->shape.back() = od;
  }
  const int out_n = lay.batch * od;
  pre->data.assign(static_cast<std::size_t>(out_n), 0.0f);
  post->shape = pre->shape;
  post->data.assign(static_cast<std::size_t>(out_n), 0.0f);

  for (int b = 0; b < lay.batch; ++b) {
    const int x_off = b * lay.d;
    std::vector<double> logits(static_cast<std::size_t>(lay.k));
    for (int k = 0; k < lay.k; ++k) {
      double sq = 0.0;
      const int c_off = k * lay.d;
      for (int d = 0; d < lay.d; ++d) {
        const double diff =
            static_cast<double>(input.data[static_cast<std::size_t>(x_off + d)]) -
            static_cast<double>(w[static_cast<std::size_t>(c_off + d)]);
        sq += diff * diff;
      }
      logits[static_cast<std::size_t>(k)] = -sq / (2.0 * temp_sq);
    }
    const auto a = soft_max(logits);
    const int o_off = b * od;
    if (cfg.output_mode == OutputMode::Features) {
      for (int d = 0; d < lay.d; ++d) {
        double v = 0.0;
        for (int k = 0; k < lay.k; ++k) {
          v += a[static_cast<std::size_t>(k)] *
               static_cast<double>(w[static_cast<std::size_t>(k * lay.d + d)]);
        }
        pre->data[static_cast<std::size_t>(o_off + d)] = static_cast<float>(v);
        post->data[static_cast<std::size_t>(o_off + d)] =
            static_cast<float>(activate_f64(v, cfg.activation));
      }
    } else {
      for (int k = 0; k < lay.k; ++k) {
        const double v = a[static_cast<std::size_t>(k)];
        pre->data[static_cast<std::size_t>(o_off + k)] = static_cast<float>(v);
        post->data[static_cast<std::size_t>(o_off + k)] =
            static_cast<float>(activate_f64(v, cfg.activation));
      }
    }
  }
  return Status::Ok();
}

} // namespace welvet::kmeans
