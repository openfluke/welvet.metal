#pragma once

// Sequence-mixing contract — mirrors welvet/layers/seqmix (Kind tags + docs only).
// Concrete math lives in mha / mamba / gdn; this package does not Forward.

#include <cstdint>
#include <string>

namespace welvet::seqmix {

enum class Kind : int32_t {
  Attention = 0,  // MHA / GQA / …
  SSM = 1,        // Mamba / S6
  LinearAttn = 2, // GDN-style / Performer / …
  ConvMix = 3,    // Hyena / future
};

inline const char* kind_name(Kind k) {
  switch (k) {
    case Kind::Attention:
      return "attention";
    case Kind::SSM:
      return "ssm";
    case Kind::LinearAttn:
      return "linear_attn";
    case Kind::ConvMix:
      return "conv_mix";
    default:
      return nullptr;
  }
}

inline std::string to_string(Kind k) {
  if (const char* s = kind_name(k)) return s;
  return "Kind(" + std::to_string(static_cast<int>(k)) + ")";
}

// Documented shared I/O: Forward(l, x[B,T,D]) → (pre, post[B,T,D]).
struct Contract {
  Kind kind = Kind::Attention;
  int d_model = 0;
  int max_t = 0;
};

inline constexpr Kind mamba_kind() { return Kind::SSM; }
inline constexpr Kind gdn_kind() { return Kind::LinearAttn; }
inline constexpr Kind mha_mixer_kind() { return Kind::Attention; }

} // namespace welvet::seqmix
