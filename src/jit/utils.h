/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#ifndef BRT_JIT_UTILS_H_
#define BRT_JIT_UTILS_H_

#include <algorithm>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace brt {
namespace jit {

std::vector<uint> ToUintVector(const std::string& s, char delimiter = ',') {
  std::vector<uint> uint_vector;
  std::string token;
  std::istringstream vector_stream(s);
  while (getline(vector_stream, token, delimiter)) {
    uint_vector.push_back(std::stoul(token));
  }
  return uint_vector;
}

std::vector<int> ToIntVector(const std::string& s, char delimiter = ',') {
  std::vector<int> int_vector;
  std::string token;
  std::istringstream vector_stream(s);
  while (getline(vector_stream, token, delimiter)) {
    int_vector.push_back(std::stoi(token));
  }
  return int_vector;
}

long CaptureWithDefault(const std::string& s, const std::regex& tag_regex, long default_val) {
  std::smatch match;
  if (std::regex_search(s, match, tag_regex)) {
    return std::stol(match[1]);
  } else {
    return default_val;
  }
}

std::string CaptureWithDefault(const std::string& s, const std::regex& tag_regex,
                                 std::string default_val) {
  std::smatch match;
  if (std::regex_search(s, match, tag_regex)) {
    return match[1];
  } else {
    return default_val;
  }
}

template <typename T>
std::vector<int> SortIndice(const std::vector<T>& V) {
  std::vector<int> indice(V.size());
  std::iota(indice.begin(), indice.end(), 0);
  std::stable_sort(indice.begin(), indice.end(), [&V](int i1, int i2) { return V[i1] < V[i2]; });
  return indice;
}

}  // namespace jit
}  // namespace brt

#endif //BRT_JIT_UTILS_H_
