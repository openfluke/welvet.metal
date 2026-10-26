#pragma once
#include "welvet/status.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>
#include <type_traits>
namespace welvet::metal {
struct Arg {
  const void *input;
  void *output;
  std::size_t bytes;
};
inline Arg read(const std::vector<float> &v) {
  return {v.data(), nullptr, v.size() * sizeof(float)};
}
inline Arg write(std::vector<float> &v) { return {v.data(), v.data(), v.size() * sizeof(float)}; }
inline Arg read(const std::vector<int32_t> &v) {
  return {v.data(), nullptr, v.size() * sizeof(int32_t)};
}
template<class T> inline Arg read(const std::vector<T>& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  return {v.data(),nullptr,v.size()*sizeof(T)};
}
template<class T> inline Arg write(std::vector<T>& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  return {v.data(),v.data(),v.size()*sizeof(T)};
}
Status run(const char *kernel, std::initializer_list<Arg> args,
           std::initializer_list<int32_t> ints = {}, std::initializer_list<float> reals = {},
           std::size_t threads = 1);
} // namespace welvet::metal
