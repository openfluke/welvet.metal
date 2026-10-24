#pragma once

#include <cstdint>
#include <string_view>

namespace welvet {

// Mirrors welvet/core/dtype.go — 34 storage types (IDs 0–33).
enum class DType : int32_t {
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

constexpr int kNumDTypes = 34;

std::string_view dtype_name(DType dt);
bool dtype_implemented_forward(DType dt); // true only for implemented storage paths

} // namespace welvet
