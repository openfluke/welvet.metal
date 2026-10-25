namespace native_gpu {
enum class DType : int {
  Float64 = 0,
  Float32 = 1,
  Float16 = 2,
  BFloat16 = 3,
  FP8E4M3 = 4,
  FP8E5M2 = 5,
  Int64 = 6,
  Int32 = 7,
  Int16 = 8,
  Int8 = 9,
  Uint64 = 10,
  Uint32 = 11,
  Uint16 = 12,
  Uint8 = 13,
  Int4 = 14,
  Uint4 = 15,
  FP4 = 16,
  Int2 = 17,
  Uint2 = 18,
  Ternary = 19,
  Binary = 20,
  Int = 21,
  Uint = 22,
  Uintptr = 23,
  Complex64 = 24,
  Complex128 = 25,
  NF4 = 26,
  FP6 = 27,
  Int6 = 28,
  Uint6 = 29,
  Int5 = 30,
  Uint5 = 31,
  Int3 = 32,
  Uint3 = 33,
};
struct Raw {
  device uchar *ptr;
  int count;
  device uchar *data() const { return ptr; }
  int size() const { return count; }
  device uchar &operator[](uint i) { return ptr[i]; }
};
struct WeightStore {
  DType dtype;
  Raw native;
  float scale;
};
constant bool kHostInt64 = true;
inline ulong pack_float(float v) {
  uint u = as_type<uint>(v);
  uint sign = u >> 31, e = (u >> 23) & 255, m = u & 0x7fffff;
  if (e == 255)
    return (ulong(sign) << 63) | (0x7fful << 52) | (ulong(m) << 29);
  if (e == 0) {
    if (!m)
      return ulong(sign) << 63;
    int exp;
    float f = frexp(abs(v), exp);
    return (ulong(sign) << 63) | (ulong(exp - 1 + 1023) << 52) |
           (ulong(ldexp(f, 53)) & 0xffffffffffffful);
  }
  return (ulong(sign) << 63) | (ulong(int(e) - 127 + 1023) << 52) | (ulong(m) << 29);
}
inline float2 unpack_sd(ulong bits) {
  bool neg = bits >> 63;
  int e = int((bits >> 52) & 2047);
  ulong m = bits & 0xffffffffffffful;
  if (e == 2047)
    return float2(neg ? -INFINITY : INFINITY, 0);
  if (e == 0 && !m)
    return float2(0);
  if (e)
    m |= 1ul << 52;
  int exp = (e ? e - 1023 : 1 - 1023) - 52;
  float hi = ldexp(float(m >> 29), exp + 29), lo = ldexp(float(m & 0x1ffffffful), exp);
  float2 v = sd_add(float2(hi, 0), float2(lo, 0));
  return neg ? -v : v;
}
inline ulong pack_sd(float2 value) {
  value = sd_add(float2(value.x, 0), float2(value.y, 0));
  if (value.x == 0)
    return pack_float(value.y);
  if (!isfinite(value.x))
    return pack_float(value.x);
  bool neg = value.x < 0;
  float hi = neg ? -value.x : value.x, lo = neg ? -value.y : value.y;
  int e;
  float frac = frexp(hi, e);
  long mant = long(ldexp(frac, 53)) + long(ldexp(lo, 53 - e));
  if (mant >= (1l << 53)) {
    mant >>= 1;
    ++e;
  }
  if (mant < (1l << 52)) {
    mant <<= 1;
    --e;
  }
  return (ulong(neg) << 63) | (ulong(e - 1 + 1023) << 52) | (ulong(mant) & 0xffffffffffffful);
}
inline float bits_to_f32(uint x) { return as_type<float>(x); }
inline uint f32_to_bits(float x) { return as_type<uint>(x); }
inline float bits_to_f64(ulong x) { return sd_to_float(unpack_sd(x)); }
inline ulong f64_to_bits(float x) { return pack_float(x); }
inline ulong quant_u64(float v, float min_v, float scale) {
  float q = round((v - min_v) / scale);
  if (q <= 0)
    return 0;
  if (q >= 1e15f)
    return 1000000000000000ul;
  return ulong(q);
}
float clamp_i(float v, float lo, float hi) {
  if (v < lo)
    return lo;
  if (v > hi)
    return hi;
  return v;
}
void put_u16(thread Raw &b, uint off, ushort v) {
  b[off] = static_cast<uchar>(v & 0xff);
  b[off + 1] = static_cast<uchar>((v >> 8) & 0xff);
}
void put_u32(thread Raw &b, uint off, uint v) {
  for (int i = 0; i < 4; ++i)
    b[off + i] = static_cast<uchar>((v >> (8 * i)) & 0xff);
}
void put_u64(thread Raw &b, uint off, ulong v) {
  for (int i = 0; i < 8; ++i)
    b[off + i] = static_cast<uchar>((v >> (8 * i)) & 0xff);
}
ushort get_u16(device const uchar *p) { return ushort(p[0]) | (ushort(p[1]) << 8); }
uint get_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}
ulong get_u64(device const uchar *p) {
  ulong v = 0;
  for (int i = 0; i < 8; ++i)
    v |= ulong(p[i]) << (8 * i);
  return v;
}
ulong get_bit_code(device const uchar *raw, int i, int bits) {
  const int bit_off = i * bits;
  ulong c = 0;
  for (int k = 0; k < bits; ++k) {
    if (raw[(bit_off + k) / 8] & (1u << ((bit_off + k) % 8))) {
      c |= ulong(1) << k;
    }
  }
  return c;
}
long sign_extend(ulong code, int bits) {
  const ulong sign_bit = ulong(1) << (bits - 1);
  const ulong mask = (ulong(1) << bits) - 1;
  code &= mask;
  if (code & sign_bit) {
    return static_cast<long>(code) - static_cast<long>(ulong(1) << bits);
  }
  return static_cast<long>(code);
}
constant float kNf4[16] = {
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
inline uint f32bits(float f) { return as_type<uint>(f); }
inline float bitsf32(uint u) { return as_type<float>(u); }
ushort float32_to_float16(float f) {
  const uint b = f32bits(f);
  const ushort sign = static_cast<ushort>((b >> 16) & 0x8000);
  int expo = static_cast<int>((b >> 23) & 0xff) - 127 + 15;
  const uint mant = b & 0x7fffff;
  if (expo <= 0) {
    return sign;
  }
  if (expo >= 31) {
    return static_cast<ushort>(sign | 0x7c00);
  }
  return static_cast<ushort>(sign | (expo << 10) | (mant >> 13));
}

float float16_to_float32(ushort h) {
  const uint sign = static_cast<uint>(h & 0x8000) << 16;
  const uint expo = (h >> 10) & 0x1f;
  const uint mant = h & 0x3ff;
  if (expo == 0) {
    if (mant == 0) {
      return bitsf32(sign);
    }
    return bitsf32(sign | f32bits(static_cast<float>(mant) / 1024.0f / 16384.0f));
  }
  if (expo == 31) {
    return bitsf32(sign | 0x7f800000u | (mant << 13));
  }
  return bitsf32(sign | ((expo - 15 + 127) << 23) | (mant << 13));
}

ushort float32_to_bfloat16(float f) { return static_cast<ushort>(f32bits(f) >> 16); }

float bfloat16_to_float32(ushort h) { return bitsf32(static_cast<uint>(h) << 16); }

uchar float32_to_fp8e4m3(float f) {
  if (f == 0.0f || isnan(f)) {
    return 0;
  }
  uchar sign = 0;
  if (f < 0) {
    sign = 0x80;
    f = -f;
  }
  if (f > 448.0f) {
    return static_cast<uchar>(sign | 0x7e);
  }
  const uint bits = f32bits(f);
  int exp = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint mant = bits & 0x7fffff;
  int e = exp + 7;
  if (e <= 0) {
    if (e < -2) {
      return sign;
    }
    mant |= 0x800000;
    const unsigned shift = static_cast<unsigned>(1 - e + 20);
    const uint m = mant >> shift;
    return static_cast<uchar>(sign | (m & 0x7));
  }
  if (e >= 15) {
    return static_cast<uchar>(sign | 0x7e);
  }
  uint m = (mant >> 20) & 0x7;
  if (mant & 0x80000) {
    m++;
    if (m == 8) {
      m = 0;
      e++;
      if (e >= 15) {
        return static_cast<uchar>(sign | 0x7e);
      }
    }
  }
  return static_cast<uchar>(sign | (e << 3) | m);
}

float fp8e4m3_to_float32(uchar b) {
  uint sign = 0;
  if (b & 0x80) {
    sign = 0x80000000u;
  }
  const int e = (b >> 3) & 0xf;
  const int m = b & 0x7;
  if (e == 0) {
    if (m == 0) {
      return bitsf32(sign);
    }
    return bitsf32(sign) * static_cast<float>(m) / 8.0f / 64.0f;
  }
  if (e == 15 && m == 7) {
    return bitsf32(sign | f32bits(448.0f));
  }
  const uint fexp = static_cast<uint>(e - 7 + 127);
  return bitsf32(sign | (fexp << 23) | (static_cast<uint>(m) << 20));
}

uchar float32_to_fp8e5m2(float f) {
  if (isnan(f)) {
    return 0x7e;
  }
  uchar sign = 0;
  if (f < 0 || signbit(f)) {
    sign = 0x80;
    if (f < 0) {
      f = -f;
    }
  }
  if (isinf(f)) {
    return static_cast<uchar>(sign | 0x7c);
  }
  if (f == 0.0f) {
    return sign;
  }
  const uint bits = f32bits(f);
  int exp = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint mant = bits & 0x7fffff;
  int e = exp + 15;
  if (e <= 0) {
    return sign;
  }
  if (e >= 31) {
    return static_cast<uchar>(sign | 0x7c);
  }
  uint m = (mant >> 21) & 0x3;
  if (mant & 0x100000) {
    m++;
    if (m == 4) {
      m = 0;
      e++;
      if (e >= 31) {
        return static_cast<uchar>(sign | 0x7c);
      }
    }
  }
  return static_cast<uchar>(sign | (e << 2) | m);
}

float fp8e5m2_to_float32(uchar b) {
  uint sign = 0;
  if (b & 0x80) {
    sign = 0x80000000u;
  }
  const int e = (b >> 2) & 0x1f;
  const int m = b & 0x3;
  if (e == 0) {
    if (m == 0) {
      return bitsf32(sign);
    }
    return bitsf32(sign) * static_cast<float>(m) / 4.0f / 16384.0f;
  }
  if (e == 31) {
    if (m == 0) {
      return bitsf32(sign | 0x7f800000u);
    }
    return bitsf32(sign | 0x7fc00000u);
  }
  const uint fexp = static_cast<uint>(e - 15 + 127);
  return bitsf32(sign | (fexp << 23) | (static_cast<uint>(m) << 21));
}

constant float kFp4[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};

uchar float32_to_fp4(float f) {
  uchar best = 0;
  float best_err = FLT_MAX;
  for (int c = 0; c < 16; ++c) {
    const float e = fabs(f - kFp4[c]);
    if (e < best_err) {
      best_err = e;
      best = static_cast<uchar>(c);
    }
  }
  return best;
}

float fp4_to_float32(uchar code) { return kFp4[code & 0xf]; }

bool weight_at(thread const WeightStore &s, int i, int n, thread float *out) {
  float scale = s.scale;
  if (scale == 0)
    scale = 1;
  device const uchar *raw = s.native.data();
  const int raw_n = static_cast<int>(s.native.size());
  switch (s.dtype) {
  case DType::Float64: {
    if (raw_n < n * 8)
      return false;
    *out = static_cast<float>(bits_to_f64(get_u64(raw + i * 8)));
    return true;
  }
  case DType::Float16: {
    if (raw_n < n * 2)
      return false;
    *out = float16_to_float32(get_u16(raw + i * 2));
    return true;
  }
  case DType::BFloat16: {
    if (raw_n < n * 2)
      return false;
    *out = bfloat16_to_float32(get_u16(raw + i * 2));
    return true;
  }
  case DType::FP8E4M3: {
    if (raw_n < n)
      return false;
    *out = fp8e4m3_to_float32(raw[i]);
    return true;
  }
  case DType::FP8E5M2: {
    if (raw_n < n)
      return false;
    *out = fp8e5m2_to_float32(raw[i]);
    return true;
  }
  case DType::FP4: {
    if (raw_n < (n + 1) / 2)
      return false;
    uchar code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
    *out = fp4_to_float32(code);
    return true;
  }
  case DType::Int64: {
    if (raw_n < n * 8)
      return false;
    *out = static_cast<float>(static_cast<long>(get_u64(raw + i * 8))) * scale;
    return true;
  }
  case DType::Int32: {
    if (raw_n < n * 4)
      return false;
    *out = static_cast<float>(static_cast<int>(get_u32(raw + i * 4))) * scale;
    return true;
  }
  case DType::Int16: {
    if (raw_n < n * 2)
      return false;
    *out = static_cast<float>(static_cast<short>(get_u16(raw + i * 2))) * scale;
    return true;
  }
  case DType::Int8: {
    if (raw_n < n)
      return false;
    *out = static_cast<float>(static_cast<char>(raw[i])) * scale;
    return true;
  }
  case DType::Uint8: {
    if (raw_n < 4 + n)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(raw[4 + i]) * scale + min_v;
    return true;
  }
  case DType::Uint16: {
    if (raw_n < 4 + n * 2)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u16(raw + 4 + i * 2)) * scale + min_v;
    return true;
  }
  case DType::Uint32: {
    if (raw_n < 4 + n * 4)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u32(raw + 4 + i * 4)) * scale + min_v;
    return true;
  }
  case DType::Uint64: {
    if (raw_n < 4 + n * 8)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    return true;
  }
  case DType::Int4: {
    if (raw_n < (n + 1) / 2)
      return false;
    char q =
        (i % 2 == 0) ? static_cast<char>(raw[i / 2] & 0xF) : static_cast<char>(raw[i / 2] >> 4);
    if (q > 7)
      q = static_cast<char>(q - 16);
    *out = static_cast<float>(q) * scale;
    return true;
  }
  case DType::Uint4: {
    if (raw_n < 4 + (n + 1) / 2)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    device const uchar *body = raw + 4;
    int q = (i % 2 == 0) ? (body[i / 2] & 0xF) : (body[i / 2] >> 4);
    *out = static_cast<float>(q) * scale + min_v;
    return true;
  }
  case DType::Int2: {
    if (raw_n < (n + 3) / 4)
      return false;
    const int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
    *out = static_cast<float>(code - 2) * scale;
    return true;
  }
  case DType::Uint2: {
    if (raw_n < 4 + (n + 3) / 4)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    device const uchar *body = raw + 4;
    const int code = (body[i / 4] >> ((i % 4) * 2)) & 3;
    *out = static_cast<float>(code) * scale + min_v;
    return true;
  }
  case DType::Ternary: {
    if (raw_n < (n + 3) / 4)
      return false;
    const int code = (raw[i / 4] >> ((i % 4) * 2)) & 3;
    if (code == 0)
      *out = -scale;
    else if (code == 2)
      *out = scale;
    else
      *out = 0;
    return true;
  }
  case DType::Binary: {
    if (raw_n < (n + 7) / 8)
      return false;
    *out = (raw[i / 8] & (1u << (i % 8))) ? scale : -scale;
    return true;
  }
  case DType::Int: {
    if (kHostInt64) {
      if (raw_n < n * 8)
        return false;
      *out = static_cast<float>(static_cast<long>(get_u64(raw + i * 8))) * scale;
    } else {
      if (raw_n < n * 4)
        return false;
      *out = static_cast<float>(static_cast<int>(get_u32(raw + i * 4))) * scale;
    }
    return true;
  }
  case DType::Uint: {
    if (kHostInt64) {
      if (raw_n < 4 + n * 8)
        return false;
      const float min_v = bits_to_f32(get_u32(raw));
      *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    } else {
      if (raw_n < 4 + n * 4)
        return false;
      const float min_v = bits_to_f32(get_u32(raw));
      *out = static_cast<float>(get_u32(raw + 4 + i * 4)) * scale + min_v;
    }
    return true;
  }
  case DType::Uintptr: {
    if (raw_n < 4 + n * 8)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_u64(raw + 4 + i * 8)) * scale + min_v;
    return true;
  }
  case DType::Complex64: {
    if (raw_n < n * 8)
      return false;
    *out = bits_to_f32(get_u32(raw + i * 8));
    return true;
  }
  case DType::Complex128: {
    if (raw_n < n * 16)
      return false;
    *out = static_cast<float>(bits_to_f64(get_u64(raw + i * 16)));
    return true;
  }
  case DType::NF4: {
    if (raw_n < (n + 1) / 2)
      return false;
    int code = (i % 2 == 0) ? (raw[i / 2] & 0xF) : (raw[i / 2] >> 4);
    *out = kNf4[code] * scale;
    return true;
  }
  case DType::FP6:
  case DType::Int6: {
    const int need = (n * 6 + 7) / 8;
    if (raw_n < need)
      return false;
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 6), 6)) * scale;
    return true;
  }
  case DType::Uint6: {
    const int need = 4 + (n * 6 + 7) / 8;
    if (raw_n < need)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 6)) * scale + min_v;
    return true;
  }
  case DType::Int5: {
    const int need = (n * 5 + 7) / 8;
    if (raw_n < need)
      return false;
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 5), 5)) * scale;
    return true;
  }
  case DType::Uint5: {
    const int need = 4 + (n * 5 + 7) / 8;
    if (raw_n < need)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 5)) * scale + min_v;
    return true;
  }
  case DType::Int3: {
    const int need = (n * 3 + 7) / 8;
    if (raw_n < need)
      return false;
    *out = static_cast<float>(sign_extend(get_bit_code(raw, i, 3), 3)) * scale;
    return true;
  }
  case DType::Uint3: {
    const int need = 4 + (n * 3 + 7) / 8;
    if (raw_n < need)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    *out = static_cast<float>(get_bit_code(raw + 4, i, 3)) * scale + min_v;
    return true;
  }
  case DType::Float32:
    return false;
  }
  return false;
}
void put_nibble(device uchar *raw, int i, uchar code) {
  code &= 0xF;
  if (i % 2 == 0) {
    raw[i / 2] = static_cast<uchar>((raw[i / 2] & 0xF0) | code);
  } else {
    raw[i / 2] = static_cast<uchar>((raw[i / 2] & 0x0F) | (code << 4));
  }
}
void put_2bit(device uchar *raw, int i, uchar code) {
  code &= 3;
  const unsigned shift = static_cast<unsigned>(i % 4) * 2;
  raw[i / 4] = static_cast<uchar>((raw[i / 4] & ~(3u << shift)) | (code << shift));
}
void set_bit_code(device uchar *raw, int i, int bits, ulong code) {
  const int bit_off = i * bits;
  for (int k = 0; k < bits; ++k) {
    const int byte_idx = (bit_off + k) / 8;
    const unsigned bit = static_cast<unsigned>((bit_off + k) % 8);
    if ((code >> static_cast<unsigned>(k)) & 1) {
      raw[byte_idx] |= static_cast<uchar>(1u << bit);
    } else {
      raw[byte_idx] = static_cast<uchar>(raw[byte_idx] & ~(1u << bit));
    }
  }
}
bool set_signed_nbit(device uchar *raw, int i, int bits, float scale, float v) {
  const int max_q = (1 << (bits - 1)) - 1;
  const int min_q = -(1 << (bits - 1));
  int q = static_cast<int>(round(static_cast<float>(v / scale)));
  if (q > max_q)
    q = max_q;
  if (q < min_q)
    q = min_q;
  set_bit_code(raw, i, bits, static_cast<ulong>(static_cast<unsigned>(q) & ((1u << bits) - 1)));
  return true;
}
bool set_unsigned_nbit(device uchar *raw, int i, int bits, float scale, float v) {
  const int max_q = (1 << bits) - 1;
  const float min_v = bits_to_f32(get_u32(raw));
  int q = static_cast<int>(
      clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, static_cast<float>(max_q)));
  set_bit_code(raw + 4, i, bits, static_cast<ulong>(q));
  return true;
}
bool set_weight_at(thread WeightStore &s, int i, int n, float v) {
  float scale = s.scale;
  if (scale == 0)
    scale = 1;
  device uchar *raw = s.native.data();
  const int raw_n = static_cast<int>(s.native.size());
  switch (s.dtype) {
  case DType::Float16: {
    if (raw_n < n * 2)
      return false;
    put_u16(s.native, static_cast<uint>(i * 2), float32_to_float16(v));
    return true;
  }
  case DType::BFloat16: {
    if (raw_n < n * 2)
      return false;
    put_u16(s.native, static_cast<uint>(i * 2), float32_to_bfloat16(v));
    return true;
  }
  case DType::FP8E4M3: {
    if (raw_n < n)
      return false;
    raw[i] = float32_to_fp8e4m3(v);
    return true;
  }
  case DType::FP8E5M2: {
    if (raw_n < n)
      return false;
    raw[i] = float32_to_fp8e5m2(v);
    return true;
  }
  case DType::FP4: {
    if (raw_n < (n + 1) / 2)
      return false;
    put_nibble(raw, i, float32_to_fp4(v));
    return true;
  }
  case DType::Int64: {
    if (raw_n < n * 8)
      return false;
    const long q = static_cast<long>(round(static_cast<float>(v / scale)));
    put_u64(s.native, static_cast<uint>(i * 8), static_cast<ulong>(q));
    return true;
  }
  case DType::Int32: {
    if (raw_n < n * 4)
      return false;
    const int q = static_cast<int>(
        clamp_i(round(static_cast<float>(v / scale)), -2147483648.0, 2147483647.0));
    put_u32(s.native, static_cast<uint>(i * 4), static_cast<uint>(q));
    return true;
  }
  case DType::Int16: {
    if (raw_n < n * 2)
      return false;
    const short q =
        static_cast<short>(clamp_i(round(static_cast<float>(v / scale)), -32768.0, 32767.0));
    put_u16(s.native, static_cast<uint>(i * 2), static_cast<ushort>(q));
    return true;
  }
  case DType::Int8: {
    if (raw_n < n)
      return false;
    raw[i] = static_cast<uchar>(
        static_cast<char>(clamp_i(round(static_cast<float>(v / scale)), -128.0, 127.0)));
    return true;
  }
  case DType::Uint8: {
    if (raw_n < 4 + n)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    raw[4 + i] =
        static_cast<uchar>(clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 255.0));
    return true;
  }
  case DType::Uint16: {
    if (raw_n < 4 + n * 2)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const ushort q =
        static_cast<ushort>(clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 65535.0));
    put_u16(s.native, static_cast<uint>(4 + i * 2), q);
    return true;
  }
  case DType::Uint32: {
    if (raw_n < 4 + n * 4)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const uint q = static_cast<uint>(
        clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 4294967295.0));
    put_u32(s.native, static_cast<uint>(4 + i * 4), q);
    return true;
  }
  case DType::Uint64: {
    if (raw_n < 4 + n * 8)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const ulong q = quant_u64(v, min_v, scale);
    put_u64(s.native, static_cast<uint>(4 + i * 8), q);
    return true;
  }
  case DType::Int4: {
    if (raw_n < (n + 1) / 2)
      return false;
    const int q = static_cast<int>(clamp_i(round(static_cast<float>(v / scale)), -8.0, 7.0)) & 0xF;
    put_nibble(raw, i, static_cast<uchar>(q));
    return true;
  }
  case DType::Uint4: {
    if (raw_n < 4 + (n + 1) / 2)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const int q =
        static_cast<int>(clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 15.0)) & 0xF;
    put_nibble(raw + 4, i, static_cast<uchar>(q));
    return true;
  }
  case DType::Int2: {
    if (raw_n < (n + 3) / 4)
      return false;
    const int q = static_cast<int>(clamp_i(round(static_cast<float>(v / scale)), -2.0, 1.0));
    put_2bit(raw, i, static_cast<uchar>((q + 2) & 3));
    return true;
  }
  case DType::Uint2: {
    if (raw_n < 4 + (n + 3) / 4)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const int q =
        static_cast<int>(clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 3.0)) & 3;
    put_2bit(raw + 4, i, static_cast<uchar>(q));
    return true;
  }
  case DType::Ternary: {
    if (raw_n < (n + 3) / 4)
      return false;
    const float r = v / scale;
    uchar code = 1;
    if (r > 0.5f)
      code = 2;
    else if (r < -0.5f)
      code = 0;
    put_2bit(raw, i, code);
    return true;
  }
  case DType::Binary: {
    if (raw_n < (n + 7) / 8)
      return false;
    const unsigned bit = static_cast<unsigned>(i % 8);
    if (v >= 0) {
      raw[i / 8] |= static_cast<uchar>(1u << bit);
    } else {
      raw[i / 8] = static_cast<uchar>(raw[i / 8] & ~(1u << bit));
    }
    return true;
  }
  case DType::Int: {
    if (kHostInt64) {
      if (raw_n < n * 8)
        return false;
      const long q = static_cast<long>(round(static_cast<float>(v / scale)));
      put_u64(s.native, static_cast<uint>(i * 8), static_cast<ulong>(q));
    } else {
      if (raw_n < n * 4)
        return false;
      const int q = static_cast<int>(
          clamp_i(round(static_cast<float>(v / scale)), -2147483648.0, 2147483647.0));
      put_u32(s.native, static_cast<uint>(i * 4), static_cast<uint>(q));
    }
    return true;
  }
  case DType::Uint: {
    if (kHostInt64) {
      if (raw_n < 4 + n * 8)
        return false;
      const float min_v = bits_to_f32(get_u32(raw));
      const ulong q = quant_u64(v, min_v, scale);
      put_u64(s.native, static_cast<uint>(4 + i * 8), q);
    } else {
      if (raw_n < 4 + n * 4)
        return false;
      const float min_v = bits_to_f32(get_u32(raw));
      const uint q = static_cast<uint>(
          clamp_i(round(static_cast<float>((v - min_v) / scale)), 0.0, 4294967295.0));
      put_u32(s.native, static_cast<uint>(4 + i * 4), q);
    }
    return true;
  }
  case DType::Uintptr: {
    if (raw_n < 4 + n * 8)
      return false;
    const float min_v = bits_to_f32(get_u32(raw));
    const ulong q = quant_u64(v, min_v, scale);
    put_u64(s.native, static_cast<uint>(4 + i * 8), q);
    return true;
  }
  case DType::Complex64: {
    if (raw_n < n * 8)
      return false;
    put_u32(s.native, static_cast<uint>(i * 8), f32_to_bits(v));
    return true;
  }
  case DType::Complex128: {
    if (raw_n < n * 16)
      return false;
    put_u64(s.native, static_cast<uint>(i * 16), f64_to_bits(static_cast<float>(v)));
    return true;
  }
  case DType::NF4: {
    if (raw_n < (n + 1) / 2)
      return false;
    const float r = v / scale;
    int best = 0;
    float best_err = FLT_MAX;
    for (int c = 0; c < 16; ++c) {
      const float e = fabs(r - kNf4[c]);
      if (e < best_err) {
        best_err = e;
        best = c;
      }
    }
    put_nibble(raw, i, static_cast<uchar>(best & 0xF));
    return true;
  }
  case DType::FP6:
  case DType::Int6: {
    const int need = (n * 6 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_signed_nbit(raw, i, 6, scale, v);
  }
  case DType::Int5: {
    const int need = (n * 5 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_signed_nbit(raw, i, 5, scale, v);
  }
  case DType::Int3: {
    const int need = (n * 3 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_signed_nbit(raw, i, 3, scale, v);
  }
  case DType::Uint6: {
    const int need = 4 + (n * 6 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_unsigned_nbit(raw, i, 6, scale, v);
  }
  case DType::Uint5: {
    const int need = 4 + (n * 5 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_unsigned_nbit(raw, i, 5, scale, v);
  }
  case DType::Uint3: {
    const int need = 4 + (n * 3 + 7) / 8;
    if (raw_n < need)
      return false;
    return set_unsigned_nbit(raw, i, 3, scale, v);
  }
  case DType::Float64:
  case DType::Float32:
    return false;
  }
  return false;
}

} // namespace native_gpu
kernel void op_native_sgd(OP_ARGS) {
  if (tid)
    return;
  using namespace native_gpu;
  int n = c[1];
  d[0] = 0;
  WeightStore s = {DType(c[0]), {(device uchar *)a, c[2]}, r[0]};
  device float2 *grad = (device float2 *)b;
  float2 lr = float2(r[1], r[2]);
  if (s.dtype == DType::Float64) {
    for (int i = 0; i < n; ++i) {
      ulong raw = get_u64(s.native.ptr + i * 8);
      float2 value = sd_add(unpack_sd(raw), -sd_mul(lr, grad[i]));
      put_u64(s.native, i * 8, pack_sd(value));
    }
    return;
  }
  float v0 = 0;
  if (!weight_at(s, 0, n, &v0) || !set_weight_at(s, 0, n, v0)) {
    d[0] = 1;
    return;
  }
  for (int i = 0; i < n; ++i) {
    float old = 0;
    if (!weight_at(s, i, n, &old)) {
      d[0] = 1;
      return;
    }
    float value = sd_to_float(sd_add(float2(old, 0), -sd_mul(lr, grad[i])));
    if (!set_weight_at(s, i, n, value)) {
      d[0] = 1;
      return;
    }
  }
}
kernel void op_bias_sgd(OP_ARGS) {
  if (tid >= uint(c[0]))
    return;
  using namespace native_gpu;
  device uchar *raw = (device uchar *)a;
  device float2 *grad = (device float2 *)b;
  float2 value = sd_add(unpack_sd(get_u64(raw + tid * 8)), -sd_mul(float2(r[0], r[1]), grad[tid]));
  Raw out = {raw, c[0] * 8};
  put_u64(out, tid * 8, pack_sd(value));
}
