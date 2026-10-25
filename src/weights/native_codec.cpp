#include "welvet/metal/training.hpp"
#include "welvet/weights.hpp"
#include "welvet/float_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace welvet {
namespace {

#if defined(__LP64__) || defined(_WIN64) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
constexpr bool kHostInt64 = true;
#else
constexpr bool kHostInt64 = false;
#endif

float abs_max(const float* w, int n) {
  float m = 0;
  for (int i = 0; i < n; ++i) {
    const float a = std::fabs(w[i]);
    if (a > m) m = a;
  }
  return m;
}

void min_max(const float* w, int n, float* mn, float* mx) {
  *mn = *mx = w[0];
  for (int i = 1; i < n; ++i) {
    if (w[i] < *mn) *mn = w[i];
    if (w[i] > *mx) *mx = w[i];
  }
}

double clamp_i(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void put_u16(std::vector<uint8_t>& b, size_t off, uint16_t v) {
  b[off] = static_cast<uint8_t>(v & 0xff);
  b[off + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
}
void put_u32(std::vector<uint8_t>& b, size_t off, uint32_t v) {
  for (int i = 0; i < 4; ++i) b[off + i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
}
void put_u64(std::vector<uint8_t>& b, size_t off, uint64_t v) {
  for (int i = 0; i < 8; ++i) b[off + i] = static_cast<uint8_t>((v >> (8 * i)) & 0xff);
}
uint16_t get_u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t get_u32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t get_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
  return v;
}
float bits_to_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint32_t f32_to_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
double bits_to_f64(uint64_t u) {
  double d;
  std::memcpy(&d, &u, 8);
  return d;
}
uint64_t f64_to_bits(double d) {
  uint64_t u;
  std::memcpy(&u, &d, 8);
  return u;
}

constexpr float kNf4[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

std::vector<uint8_t> pack_bit_codes(const std::vector<uint64_t>& codes, int bits) {
  const int n = static_cast<int>(codes.size());
  const int total = n * bits;
  std::vector<uint8_t> b((total + 7) / 8, 0);
  for (int i = 0; i < n; ++i) {
    const int bit_off = i * bits;
    for (int k = 0; k < bits; ++k) {
      if ((codes[i] >> k) & 1) {
        b[(bit_off + k) / 8] |= static_cast<uint8_t>(1u << ((bit_off + k) % 8));
      }
    }
  }
  return b;
}

uint64_t get_bit_code(const uint8_t* raw, int i, int bits) {
  const int bit_off = i * bits;
  uint64_t c = 0;
  for (int k = 0; k < bits; ++k) {
    if (raw[(bit_off + k) / 8] & (1u << ((bit_off + k) % 8))) {
      c |= uint64_t(1) << k;
    }
  }
  return c;
}

int64_t sign_extend(uint64_t code, int bits) {
  const uint64_t sign_bit = uint64_t(1) << (bits - 1);
  const uint64_t mask = (uint64_t(1) << bits) - 1;
  code &= mask;
  if (code & sign_bit) {
    return static_cast<int64_t>(code) - static_cast<int64_t>(uint64_t(1) << bits);
  }
  return static_cast<int64_t>(code);
}

Status pack_signed_nbit(const float* w, int n, int bits, std::vector<uint8_t>* out, float* scale) {
  const int max_q = (1 << (bits - 1)) - 1;
  const int min_q = -(1 << (bits - 1));
  float s = abs_max(w, n) / static_cast<float>(max_q);
  if (s == 0) s = 1;
  std::vector<uint64_t> codes(n);
  for (int i = 0; i < n; ++i) {
    int q = static_cast<int>(std::round(w[i] / s));
    if (q > max_q) q = max_q;
    if (q < min_q) q = min_q;
    codes[i] = static_cast<uint64_t>(static_cast<unsigned>(q) & ((1u << bits) - 1));
  }
  *out = pack_bit_codes(codes, bits);
  *scale = s;
  return Status::Ok();
}

Status pack_unsigned_nbit(const float* w, int n, int bits, std::vector<uint8_t>* out, float* scale) {
  const int max_q = (1 << bits) - 1;
  float mn, mx;
  min_max(w, n, &mn, &mx);
  float span = mx - mn;
  if (span == 0) span = 1;
  std::vector<uint64_t> codes(n);
  for (int i = 0; i < n; ++i) {
    int q = static_cast<int>(clamp_i(std::round((w[i] - mn) / span * max_q), 0, max_q));
    codes[i] = static_cast<uint64_t>(q);
  }
  auto body = pack_bit_codes(codes, bits);
  out->assign(4 + body.size(), 0);
  put_u32(*out, 0, f32_to_bits(mn));
  std::copy(body.begin(), body.end(), out->begin() + 4);
  *scale = span / static_cast<float>(max_q);
  return Status::Ok();
}

Status pack_native(DType dt, const float* w, int n, std::vector<uint8_t>* out, float* scale) {
  out->clear();
  *scale = 1;
  switch (dt) {
  case DType::Float64: {
    out->resize(n * 8);
    for (int i = 0; i < n; ++i) put_u64(*out, i * 8, f64_to_bits(static_cast<double>(w[i])));
    return Status::Ok();
  }
  case DType::Float16: {
    out->resize(n * 2);
    for (int i = 0; i < n; ++i) put_u16(*out, i * 2, float32_to_float16(w[i]));
    return Status::Ok();
  }
  case DType::BFloat16: {
    out->resize(n * 2);
    for (int i = 0; i < n; ++i) put_u16(*out, i * 2, float32_to_bfloat16(w[i]));
    return Status::Ok();
  }
  case DType::FP8E4M3: {
    out->resize(n);
    for (int i = 0; i < n; ++i) (*out)[i] = float32_to_fp8e4m3(w[i]);
    return Status::Ok();
  }
  case DType::FP8E5M2: {
    out->resize(n);
    for (int i = 0; i < n; ++i) (*out)[i] = float32_to_fp8e5m2(w[i]);
    return Status::Ok();
  }
  case DType::FP4: {
    out->assign((n + 1) / 2, 0);
    for (int i = 0; i < n; ++i) {
      const uint8_t code = float32_to_fp4(w[i]) & 0xF;
      if (i % 2 == 0) (*out)[i / 2] = code;
      else (*out)[i / 2] |= static_cast<uint8_t>(code << 4);
    }
    return Status::Ok();
  }
  case DType::Int64: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->resize(n * 8);
    for (int i = 0; i < n; ++i) {
      // Go: Round(float64(v/scale*1e6)) with float32 arithmetic first.
      const float qf = w[i] / s * 1e6f;
      const int64_t q = static_cast<int64_t>(std::round(static_cast<double>(qf)));
      put_u64(*out, i * 8, static_cast<uint64_t>(q));
    }
    *scale = s / 1e6f;
    return Status::Ok();
  }
  case DType::Int32: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->resize(n * 4);
    for (int i = 0; i < n; ++i) {
      // Go: Round(float64(v/scale*2147483647)) with float32 arithmetic first
      // (2147483647 is not exact in f32 → 2147483648).
      const float qf = w[i] / s * 2147483647.0f;
      const int32_t q = static_cast<int32_t>(std::round(static_cast<double>(qf)));
      put_u32(*out, i * 4, static_cast<uint32_t>(q));
    }
    *scale = s / 2147483647.0f;
    return Status::Ok();
  }
  case DType::Int16: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->resize(n * 2);
    for (int i = 0; i < n; ++i) {
      // Go: Round(float64(v/scale*32767)) with float32 arithmetic first.
      const float qf = w[i] / s * 32767.0f;
      const int q = static_cast<int>(clamp_i(std::round(static_cast<double>(qf)), -32768, 32767));
      put_u16(*out, i * 2, static_cast<uint16_t>(static_cast<int16_t>(q)));
    }
    *scale = s / 32767.0f;
    return Status::Ok();
  }
  case DType::Int8: {
    float s = abs_max(w, n) / 127.0f;
    if (s == 0) s = 1;
    out->resize(n);
    for (int i = 0; i < n; ++i) {
      const int q = static_cast<int>(clamp_i(std::round(w[i] / s), -128, 127));
      (*out)[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::Uint8:
  case DType::Uint16:
  case DType::Uint32:
  case DType::Uint64: {
    float mn, mx;
    min_max(w, n, &mn, &mx);
    float span = mx - mn;
    if (span == 0) span = 1;
    if (dt == DType::Uint8) {
      out->assign(4 + n, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      for (int i = 0; i < n; ++i) {
        // Go: Round(float64((v-min)/span*255)) with float32 arithmetic first.
        const float qf = (w[i] - mn) / span * 255.0f;
        (*out)[4 + i] =
            static_cast<uint8_t>(clamp_i(std::round(static_cast<double>(qf)), 0, 255));
      }
      *scale = span / 255.0f;
    } else if (dt == DType::Uint16) {
      out->assign(4 + n * 2, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      for (int i = 0; i < n; ++i) {
        const float qf = (w[i] - mn) / span * 65535.0f;
        const uint16_t q =
            static_cast<uint16_t>(clamp_i(std::round(static_cast<double>(qf)), 0, 65535));
        put_u16(*out, 4 + i * 2, q);
      }
      *scale = span / 65535.0f;
    } else if (dt == DType::Uint32) {
      out->assign(4 + n * 4, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      // Go multiplies by float32(MaxUint32) before widening to float64.
      const float maxu_f = static_cast<float>(4294967295u);
      for (int i = 0; i < n; ++i) {
        const float qf = (w[i] - mn) / span * maxu_f;
        const uint32_t q = static_cast<uint32_t>(
            clamp_i(std::round(static_cast<double>(qf)), 0, 4294967295.0));
        put_u32(*out, 4 + i * 4, q);
      }
      *scale = span / maxu_f;
    } else {
      out->assign(4 + n * 8, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      for (int i = 0; i < n; ++i) {
        const float qf = (w[i] - mn) / span * 1e15f; // Go float32 intermediate
        const uint64_t q = static_cast<uint64_t>(clamp_i(std::round(static_cast<double>(qf)), 0, 1e15));
        put_u64(*out, 4 + i * 8, q);
      }
      *scale = span / 1e15f;
    }
    return Status::Ok();
  }
  case DType::Int4: {
    float s = abs_max(w, n) / 7.0f;
    if (s == 0) s = 1;
    out->assign((n + 1) / 2, 0);
    for (int i = 0; i < n; ++i) {
      const int q = static_cast<int>(clamp_i(std::round(w[i] / s), -8, 7)) & 0xF;
      if (i % 2 == 0) (*out)[i / 2] = static_cast<uint8_t>(q);
      else (*out)[i / 2] |= static_cast<uint8_t>(q << 4);
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::Uint4: {
    float mn, mx;
    min_max(w, n, &mn, &mx);
    float span = mx - mn;
    if (span == 0) span = 1;
    out->assign(4 + (n + 1) / 2, 0);
    put_u32(*out, 0, f32_to_bits(mn));
    for (int i = 0; i < n; ++i) {
      const float qf = (w[i] - mn) / span * 15.0f;
      const int q = static_cast<int>(clamp_i(std::round(static_cast<double>(qf)), 0, 15)) & 0xF;
      if (i % 2 == 0) (*out)[4 + i / 2] = static_cast<uint8_t>(q);
      else (*out)[4 + i / 2] |= static_cast<uint8_t>(q << 4);
    }
    *scale = span / 15.0f;
    return Status::Ok();
  }
  case DType::Int2: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->assign((n + 3) / 4, 0);
    for (int i = 0; i < n; ++i) {
      const int q = static_cast<int>(clamp_i(std::round(w[i] / s), -2, 1));
      const int code = (q + 2) & 3;
      (*out)[i / 4] |= static_cast<uint8_t>(code << ((i % 4) * 2));
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::Uint2: {
    float mn, mx;
    min_max(w, n, &mn, &mx);
    float span = mx - mn;
    if (span == 0) span = 1;
    out->assign(4 + (n + 3) / 4, 0);
    put_u32(*out, 0, f32_to_bits(mn));
    for (int i = 0; i < n; ++i) {
      const int q = static_cast<int>(clamp_i(std::round((w[i] - mn) / span * 3.0), 0, 3)) & 3;
      (*out)[4 + i / 4] |= static_cast<uint8_t>(q << ((i % 4) * 2));
    }
    *scale = span / 3.0f;
    return Status::Ok();
  }
  case DType::Ternary: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->assign((n + 3) / 4, 0);
    for (int i = 0; i < n; ++i) {
      const float r = w[i] / s;
      int code = 1;
      if (r > 0.5f) code = 2;
      else if (r < -0.5f) code = 0;
      (*out)[i / 4] |= static_cast<uint8_t>(code << ((i % 4) * 2));
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::Binary: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->assign((n + 7) / 8, 0);
    for (int i = 0; i < n; ++i) {
      if (w[i] >= 0) (*out)[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::Int: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    if (kHostInt64) {
      out->resize(n * 8);
      for (int i = 0; i < n; ++i) {
        const float qf = w[i] / s * 1e6f;
        const int64_t q = static_cast<int64_t>(std::round(static_cast<double>(qf)));
        put_u64(*out, i * 8, static_cast<uint64_t>(q));
      }
      *scale = s / 1e6f;
    } else {
      out->resize(n * 4);
      for (int i = 0; i < n; ++i) {
        const float qf = w[i] / s * 2147483647.0f;
        const int32_t q = static_cast<int32_t>(
            clamp_i(std::round(static_cast<double>(qf)), -2147483647.0, 2147483647.0));
        put_u32(*out, i * 4, static_cast<uint32_t>(q));
      }
      *scale = s / 2147483647.0f;
    }
    return Status::Ok();
  }
  case DType::Uint: {
    float mn, mx;
    min_max(w, n, &mn, &mx);
    float span = mx - mn;
    if (span == 0) span = 1;
    if (kHostInt64) {
      out->assign(4 + n * 8, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      for (int i = 0; i < n; ++i) {
        const float qf = (w[i] - mn) / span * 1e15f;
        const uint64_t q = static_cast<uint64_t>(clamp_i(std::round(static_cast<double>(qf)), 0, 1e15));
        put_u64(*out, 4 + i * 8, q);
      }
      *scale = span / 1e15f;
    } else {
      out->assign(4 + n * 4, 0);
      put_u32(*out, 0, f32_to_bits(mn));
      const double maxu = 4294967295.0;
      for (int i = 0; i < n; ++i) {
        const uint32_t q = static_cast<uint32_t>(clamp_i(std::round((w[i] - mn) / span * maxu), 0, maxu));
        put_u32(*out, 4 + i * 4, q);
      }
      *scale = span / static_cast<float>(maxu);
    }
    return Status::Ok();
  }
  case DType::Uintptr: {
    float mn, mx;
    min_max(w, n, &mn, &mx);
    float span = mx - mn;
    if (span == 0) span = 1;
    out->assign(4 + n * 8, 0);
    put_u32(*out, 0, f32_to_bits(mn));
    for (int i = 0; i < n; ++i) {
      const float qf = (w[i] - mn) / span * 1e15f;
      const uint64_t q = static_cast<uint64_t>(clamp_i(std::round(static_cast<double>(qf)), 0, 1e15));
      put_u64(*out, 4 + i * 8, q);
    }
    *scale = span / 1e15f;
    return Status::Ok();
  }
  case DType::Complex64: {
    out->assign(n * 8, 0);
    for (int i = 0; i < n; ++i) {
      put_u32(*out, i * 8, f32_to_bits(w[i]));
      put_u32(*out, i * 8 + 4, 0);
    }
    return Status::Ok();
  }
  case DType::Complex128: {
    out->assign(n * 16, 0);
    for (int i = 0; i < n; ++i) {
      put_u64(*out, i * 16, f64_to_bits(static_cast<double>(w[i])));
      put_u64(*out, i * 16 + 8, 0);
    }
    return Status::Ok();
  }
  case DType::NF4: {
    float s = abs_max(w, n);
    if (s == 0) s = 1;
    out->assign((n + 1) / 2, 0);
    for (int i = 0; i < n; ++i) {
      const float r = w[i] / s;
      int best = 0;
      float best_err = std::numeric_limits<float>::max();
      for (int c = 0; c < 16; ++c) {
        const float e = std::fabs(r - kNf4[c]);
        if (e < best_err) {
          best_err = e;
          best = c;
        }
      }
      if (i % 2 == 0) (*out)[i / 2] = static_cast<uint8_t>(best & 0xF);
      else (*out)[i / 2] |= static_cast<uint8_t>((best & 0xF) << 4);
    }
    *scale = s;
    return Status::Ok();
  }
  case DType::FP6:
  case DType::Int6:
    return pack_signed_nbit(w, n, 6, out, scale);
  case DType::Uint6:
    return pack_unsigned_nbit(w, n, 6, out, scale);
  case DType::Int5:
    return pack_signed_nbit(w, n, 5, out, scale);
  case DType::Uint5:
    return pack_unsigned_nbit(w, n, 5, out, scale);
  case DType::Int3:
    return pack_signed_nbit(w, n, 3, out, scale);
  case DType::Uint3:
    return pack_unsigned_nbit(w, n, 3, out, scale);
  case DType::Float32:
    return Status::Err(StatusCode::InvalidArgument, "pack_native: use f32 buffer for Float32");
  }
  return Status::Gap(std::string("pack_native ") + std::string(dtype_name(dt)));
}

Status weight_at(const WeightStore& s, int i, int n, float* out) {
  float scale = s.scale;
  if (scale == 0) scale = 1;
  const uint8_t* raw = s.native.data();
  const int raw_n = static_cast<int>(s.native.size());
  switch (s.dtype) {
  case DType::Float64: {
    if (raw_n < n * 8) return Status::Err(StatusCode::InvalidArgument, "float64 short");
    *out = static_cast<float>(bits_to_f64(get_u64(raw + i * 8)));
    return Status::Ok();
  }
  case DType::Float16: {
    if (raw_n < n * 2) return Status::Err(StatusCode::InvalidArgument, "float16 short");
    *out = float16_to_float32(get_u16(raw + i * 2));
    return Status::Ok();
  }
  case DType::BFloat16: {
    if (raw_n < n * 2) return Status::Err(StatusCode::InvalidArgument, "bfloat16 short");
    *out = bfloat16_to_float32(get_u16(raw + i * 2));
    return Status::Ok();
  }
  case DType::FP8E4M3: {
    if (raw_n < n) return Status::Err(StatusCode::InvalidArgument, "fp8e4m3 short");
    *out = fp8e4m3_to_float32(raw[i]);
    return Status::Ok();
  }
  case DType::FP8E5M2: {
    if (raw_n < n) return Status::Err(StatusCode::InvalidArgument, "fp8e5m2 short");
    *out = fp8e5m2_to_float32(raw[i]);
    return Status::Ok();
  }
  case DType::FP4: {
    if (raw_n < (n + 1) / 2) return Status::Err(StatusCode::InvalidArgument, "fp4 short");
    uint8_t code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
    *out = fp4_to_float32(code);
    return Status::Ok();
  }
  case DType::Int64: {
    if (raw_n < n * 8) return Status::Err(StatusCode::InvalidArgument, "int64 short");
    *out = static_cast<float>(static_cast<int64_t>(get_u64(raw + i * 8))) * scale;
    return Status::Ok();
  }
  case DType::Int32: {
    if (raw_n < n * 4) return Status::Err(StatusCode::InvalidArgument, "int32 short");
    *out = static_cast<float>(static_cast<int32_t>(get_u32(raw + i * 4))) * scale;
    return Status::Ok();
  }
  case DType::Int16: {
    if (raw_n < n * 2) return Status::Err(StatusCode::InvalidArgument, "int16 short");
    *out = static_cast<float>(static_cast<int16_t>(get_u16(raw + i * 2))) * scale;
    return Status::Ok();
  }
  case DType::Int8: {
    if (raw_n < n) return Status::Err(StatusCode::InvalidArgument, "int8 short");
    *out = static_cast<float>(static_cast<int8_t>(raw[i])) * scale;
    return Status::Ok();
  }
  case DType::Uint8: {
    if (raw_n < 4 + n) return Status::Err(StatusCode::InvalidArgument, "uint8 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(raw[4 + i]) * scale + min_v;
    return Status::Ok();
  }
  case DType::Uint16: {
    if (raw_n < 4 + n * 2) return Status::Err(StatusCode::InvalidArgument, "uint16 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u16(raw + 4 + i * 2)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Uint32: {
    if (raw_n < 4 + n * 4) return Status::Err(StatusCode::InvalidArgument, "uint32 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u32(raw + 4 + i * 4)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Uint64: {
    if (raw_n < 4 + n * 8) return Status::Err(StatusCode::InvalidArgument, "uint64 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Int4: {
    if (raw_n < (n + 1) / 2) return Status::Err(StatusCode::InvalidArgument, "int4 short");
    int8_t q = (i % 2 == 0) ? static_cast<int8_t>(raw[i / 2] & 0xF) : static_cast<int8_t>(raw[i / 2] >> 4);
    if (q > 7) q = static_cast<int8_t>(q - 16);
    *out = static_cast<float>(q) * scale;
    return Status::Ok();
  }
  case DType::Uint4: {
    if (raw_n < 4 + (n + 1) / 2) return Status::Err(StatusCode::InvalidArgument, "uint4 short");
    const float min_v = bits_to_f32(get_u32(raw));
    const uint8_t* body = raw + 4;
    int q = (i % 2 == 0) ? (body[i / 2] & 0xF) : (body[i / 2] >> 4);
    *out = static_cast<float>(q) * scale + min_v;
    return Status::Ok();
  }
  case DType::Int2: {
    if (raw_n < (n + 3) / 4) return Status::Err(StatusCode::InvalidArgument, "int2 short");
    const int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
    *out = static_cast<float>(code - 2) * scale;
    return Status::Ok();
  }
  case DType::Uint2: {
    if (raw_n < 4 + (n + 3) / 4) return Status::Err(StatusCode::InvalidArgument, "uint2 short");
    const float min_v = bits_to_f32(get_u32(raw));
    const uint8_t* body = raw + 4;
    const int code = (body[i / 4] >> ((i % 4) * 2)) & 3;
    *out = static_cast<float>(code) * scale + min_v;
    return Status::Ok();
  }
  case DType::Ternary: {
    if (raw_n < (n + 3) / 4) return Status::Err(StatusCode::InvalidArgument, "ternary short");
    const int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
    if (code == 0) *out = -scale;
    else if (code == 2) *out = scale;
    else *out = 0;
    return Status::Ok();
  }
  case DType::Binary: {
    if (raw_n < (n + 7) / 8) return Status::Err(StatusCode::InvalidArgument, "binary short");
    *out = (raw[i / 8] & (1u << (i % 8))) ? scale : -scale;
    return Status::Ok();
  }
  case DType::Int: {
    if (kHostInt64) {
      if (raw_n < n * 8) return Status::Err(StatusCode::InvalidArgument, "int short");
      *out = static_cast<float>(static_cast<int64_t>(get_u64(raw + i * 8))) * scale;
    } else {
      if (raw_n < n * 4) return Status::Err(StatusCode::InvalidArgument, "int short");
      *out = static_cast<float>(static_cast<int32_t>(get_u32(raw + i * 4))) * scale;
    }
    return Status::Ok();
  }
  case DType::Uint: {
    if (kHostInt64) {
      if (raw_n < 4 + n * 8) return Status::Err(StatusCode::InvalidArgument, "uint short");
      const float min_v = bits_to_f32(get_u32(raw));
      *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    } else {
      if (raw_n < 4 + n * 4) return Status::Err(StatusCode::InvalidArgument, "uint short");
      const float min_v = bits_to_f32(get_u32(raw));
      *out = static_cast<float>(get_u32(raw + 4 + i * 4)) * scale + min_v;
    }
    return Status::Ok();
  }
  case DType::Uintptr: {
    if (raw_n < 4 + n * 8) return Status::Err(StatusCode::InvalidArgument, "uintptr short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Complex64: {
    if (raw_n < n * 8) return Status::Err(StatusCode::InvalidArgument, "complex64 short");
    *out = bits_to_f32(get_u32(raw + i * 8));
    return Status::Ok();
  }
  case DType::Complex128: {
    if (raw_n < n * 16) return Status::Err(StatusCode::InvalidArgument, "complex128 short");
    *out = static_cast<float>(bits_to_f64(get_u64(raw + i * 16)));
    return Status::Ok();
  }
  case DType::NF4: {
    if (raw_n < (n + 1) / 2) return Status::Err(StatusCode::InvalidArgument, "nf4 short");
    int code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
    *out = kNf4[code] * scale;
    return Status::Ok();
  }
  case DType::FP6:
  case DType::Int6: {
    const int need = (n * 6 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "int6/fp6 short");
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 6), 6)) * scale;
    return Status::Ok();
  }
  case DType::Uint6: {
    const int need = 4 + (n * 6 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "uint6 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 6)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Int5: {
    const int need = (n * 5 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "int5 short");
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 5), 5)) * scale;
    return Status::Ok();
  }
  case DType::Uint5: {
    const int need = 4 + (n * 5 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "uint5 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 5)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Int3: {
    const int need = (n * 3 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "int3 short");
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 3), 3)) * scale;
    return Status::Ok();
  }
  case DType::Uint3: {
    const int need = 4 + (n * 3 + 7) / 8;
    if (raw_n < need) return Status::Err(StatusCode::InvalidArgument, "uint3 short");
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 3)) * scale + min_v;
    return Status::Ok();
  }
  case DType::Float32:
    return Status::Err(StatusCode::InvalidArgument, "weight_at: Float32 uses f32 buffer");
  }
  return Status::Gap(std::string("weight_at ") + std::string(dtype_name(s.dtype)));
}

// FormatNone in-place write keeping Scale / unsigned min fixed (Go setWeightAt).

} // namespace

Status WeightStore::encode_format_none(DType dt, const float* scratch, int n) {
  if (!scratch || n != rows * cols) {
    return Status::Err(StatusCode::InvalidArgument, "encode: bad scratch");
  }
  format = QuantFormat::None;
  dtype = dt;
  if (dt == DType::Float32) {
    f32.assign(scratch, scratch + n);
    native.clear();
    scale = 1;
    return Status::Ok();
  }
  f32.clear();
  return pack_native(dt, scratch, n, &native, &scale);
}

Status WeightStore::decode_row_f64(int row, double* dst) const {
  if (!dst || row < 0 || row >= rows) {
    return Status::Err(StatusCode::InvalidArgument, "decode_row shape");
  }
  if (format != QuantFormat::None) {
    return Status::Gap("decode_row packed formats");
  }
  const int off = row * cols;
  const int n = rows * cols;
  if (dtype == DType::Float32) {
    if (static_cast<int>(f32.size()) < n) {
      return Status::Err(StatusCode::InvalidArgument, "f32 short");
    }
    for (int c = 0; c < cols; ++c) {
      dst[c] = static_cast<double>(f32[off + c]);
    }
    return Status::Ok();
  }
  if (dtype == DType::Float64) {
    if (static_cast<int>(native.size()) < n * 8) {
      return Status::Err(StatusCode::InvalidArgument, "float64 native short");
    }
    for (int c = 0; c < cols; ++c) {
      dst[c] = bits_to_f64(get_u64(native.data() + (off + c) * 8));
    }
    return Status::Ok();
  }
  for (int c = 0; c < cols; ++c) {
    float v = 0;
    if (auto st = weight_at(*this, off + c, n, &v); !st) {
      return st;
    }
    dst[c] = static_cast<double>(v);
  }
  return Status::Ok();
}

Status WeightStore::apply_sgd(const double* dW, double lr) {
  if (!dW || rows <= 0 || cols <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "weights: invalid SGD arguments");
  }
  if (dtype != DType::Float32 || format != QuantFormat::None) {
    return Status::Err(StatusCode::Unsupported,
                       "weights: Metal SGD supports Float32 FormatNone only; CPU execution is forbidden");
  }
  const auto n = static_cast<std::size_t>(rows) * cols;
  if (f32.size() != n) return Status::Err(StatusCode::InvalidArgument, "weights: f32 size");
  return metal::sgd_f32_f64_grad(f32.data(), dW, n, lr);
}

Status WeightStore::apply_bias_sgd(const double* dB, int n, double lr) {
  if (bias.empty()) return Status::Ok();
  return Status::Err(StatusCode::Unsupported,
                     "weights: Metal float64 bias SGD not implemented; CPU execution is forbidden");
}

} // namespace welvet
