#pragma once

#include <string>
#include <string_view>

#include "welvet/status.hpp"

namespace welvet::metal {

// Load / compile the Welvet Metal shader library (M0: runtime source compile).
Status library_ensure_loaded();
bool library_ready();
std::string library_source_path();
Status library_reload(std::string_view optional_override_path);

} // namespace welvet::metal
