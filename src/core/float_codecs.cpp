#include "welvet/float_codecs.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace welvet {
namespace {

uint32_t f32bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
float bitsf32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

} // namespace

uint16_t float32_to_float16(float f) {
  const uint32_t b = f32bits(f);
  const uint16_t sign = static_cast<uint16_t>((b >> 16) & 0x8000);
  int expo = static_cast<int>((b >> 23) & 0xff) - 127 + 15;
  const uint32_t mant = b & 0x7fffff;
  if (expo <= 0) {
    return sign;
  }
  if (expo >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (expo << 10) | (mant >> 13));
}

float float16_to_float32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  const uint32_t expo = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x3ff;
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

uint16_t float32_to_bfloat16(float f) { return static_cast<uint16_t>(f32bits(f) >> 16); }

float bfloat16_to_float32(uint16_t h) { return bitsf32(static_cast<uint32_t>(h) << 16); }

uint8_t float32_to_fp8e4m3(float f) {
  if (f == 0.0f || std::isnan(f)) {
    return 0;
  }
  uint8_t sign = 0;
  if (f < 0) {
    sign = 0x80;
    f = -f;
  }
  if (f > 448.0f) {
    return static_cast<uint8_t>(sign | 0x7e);
  }
  const uint32_t bits = f32bits(f);
  int exp = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint32_t mant = bits & 0x7fffff;
  int e = exp + 7;
  if (e <= 0) {
    if (e < -2) {
      return sign;
    }
    mant |= 0x800000;
    const unsigned shift = static_cast<unsigned>(1 - e + 20);
    const uint32_t m = mant >> shift;
    return static_cast<uint8_t>(sign | (m & 0x7));
  }
  if (e >= 15) {
    return static_cast<uint8_t>(sign | 0x7e);
  }
  uint32_t m = (mant >> 20) & 0x7;
  if (mant & 0x80000) {
    m++;
    if (m == 8) {
      m = 0;
      e++;
      if (e >= 15) {
        return static_cast<uint8_t>(sign | 0x7e);
      }
    }
  }
  return static_cast<uint8_t>(sign | (e << 3) | m);
}

float fp8e4m3_to_float32(uint8_t b) {
  uint32_t sign = 0;
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
  const uint32_t fexp = static_cast<uint32_t>(e - 7 + 127);
  return bitsf32(sign | (fexp << 23) | (static_cast<uint32_t>(m) << 20));
}

uint8_t float32_to_fp8e5m2(float f) {
  if (std::isnan(f)) {
    return 0x7e;
  }
  uint8_t sign = 0;
  if (f < 0 || std::signbit(f)) {
    sign = 0x80;
    if (f < 0) {
      f = -f;
    }
  }
  if (std::isinf(f)) {
    return static_cast<uint8_t>(sign | 0x7c);
  }
  if (f == 0.0f) {
    return sign;
  }
  const uint32_t bits = f32bits(f);
  int exp = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint32_t mant = bits & 0x7fffff;
  int e = exp + 15;
  if (e <= 0) {
    return sign;
  }
  if (e >= 31) {
    return static_cast<uint8_t>(sign | 0x7c);
  }
  uint32_t m = (mant >> 21) & 0x3;
  if (mant & 0x100000) {
    m++;
    if (m == 4) {
      m = 0;
      e++;
      if (e >= 31) {
        return static_cast<uint8_t>(sign | 0x7c);
      }
    }
  }
  return static_cast<uint8_t>(sign | (e << 2) | m);
}

float fp8e5m2_to_float32(uint8_t b) {
  uint32_t sign = 0;
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
  const uint32_t fexp = static_cast<uint32_t>(e - 15 + 127);
  return bitsf32(sign | (fexp << 23) | (static_cast<uint32_t>(m) << 21));
}

namespace {
constexpr float kFp4[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
}

uint8_t float32_to_fp4(float f) {
  uint8_t best = 0;
  float best_err = std::numeric_limits<float>::max();
  for (int c = 0; c < 16; ++c) {
    const float e = std::fabs(f - kFp4[c]);
    if (e < best_err) {
      best_err = e;
      best = static_cast<uint8_t>(c);
    }
  }
  return best;
}

float fp4_to_float32(uint8_t code) { return kFp4[code & 0xf]; }

} // namespace welvet
