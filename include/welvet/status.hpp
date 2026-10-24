#pragma once

#include <string>
#include <string_view>

namespace welvet {

enum class StatusCode {
  Ok = 0,
  InvalidArgument = 1,
  Unsupported = 2,
  Gap = 3, // planned rebuild item — see gaps::mark / GAPS.md
  Internal = 4,
};

struct Status {
  StatusCode code = StatusCode::Ok;
  std::string message;

  static Status Ok() { return {}; }
  static Status Err(StatusCode c, std::string_view msg) {
    return Status{c, std::string(msg)};
  }
  static Status Gap(std::string_view what) {
    return Err(StatusCode::Gap, std::string("gap? ") + std::string(what));
  }

  explicit operator bool() const { return code == StatusCode::Ok; }
  bool ok() const { return code == StatusCode::Ok; }
};

} // namespace welvet
