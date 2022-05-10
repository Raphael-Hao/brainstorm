/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace brt {
namespace jit {

std::vector<uint> to_uint_vector(const std::string& s, char delimiter = ',') {
  std::vector<uint> uint_vector;
  std::string token;
  std::istringstream vector_stream(s);
  while (getline(vector_stream, token, delimiter)) {
    uint_vector.push_back(std::stoul(token));
  }
  return uint_vector;
}

long capture_with_default(const std::string& s, const std::regex& tag_regex, long default_val) {
  std::smatch match;
  if (std::regex_search(s, match, tag_regex)) {
    return std::stol(match[1]);
  } else {
    return default_val;
  }
}

std::string capture_with_default(const std::string& s, const std::regex& tag_regex,
                                 std::string default_val) {
  std::smatch match;
  if (std::regex_search(s, match, tag_regex)) {
    return match[1];
  } else {
    return default_val;
  }
}

}  // namespace jit

}  // namespace brt
