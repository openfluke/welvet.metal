#include "welvet/dtype.hpp"

namespace welvet {

std::string_view dtype_name(DType dt) {
  switch (dt) {
  case DType::Float64: return "float64";
  case DType::Float32: return "float32";
  case DType::Float16: return "float16";
  case DType::BFloat16: return "bfloat16";
  case DType::FP8E4M3: return "fp8e4m3";
  case DType::FP8E5M2: return "fp8e5m2";
  case DType::Int64: return "int64";
  case DType::Int32: return "int32";
  case DType::Int16: return "int16";
  case DType::Int8: return "int8";
  case DType::Uint64: return "uint64";
  case DType::Uint32: return "uint32";
  case DType::Uint16: return "uint16";
  case DType::Uint8: return "uint8";
  case DType::Int4: return "int4";
  case DType::Uint4: return "uint4";
  case DType::FP4: return "fp4";
  case DType::Int2: return "int2";
  case DType::Uint2: return "uint2";
  case DType::Ternary: return "ternary";
  case DType::Binary: return "binary";
  case DType::Int: return "int";
  case DType::Uint: return "uint";
  case DType::Uintptr: return "uintptr";
  case DType::Complex64: return "complex64";
  case DType::Complex128: return "complex128";
  case DType::NF4: return "nf4";
  case DType::FP6: return "fp6";
  case DType::Int6: return "int6";
  case DType::Uint6: return "uint6";
  case DType::Int5: return "int5";
  case DType::Uint5: return "uint5";
  case DType::Int3: return "int3";
  case DType::Uint3: return "uint3";
  }
  return "unknown";
}

bool dtype_implemented_forward(DType dt) {
  // FormatNone dense forward: all 34 storage dtypes (encode → MatVec f64-acc).
  const int id = static_cast<int>(dt);
  return id >= 0 && id < kNumDTypes;
}

} // namespace welvet
