#pragma once

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace welvet::gaps {

struct Entry {
  const char* id;
  const char* area;
  const char* detail;
  bool implemented;
};

// Full Metal rebuild runway inventory. Unimplemented rows are "gap?".
const std::vector<Entry>& all();
std::size_t count_gaps();
std::size_t count_done();
void print_census(std::ostream& os);

std::string mark(std::string_view what);

} // namespace welvet::gaps
