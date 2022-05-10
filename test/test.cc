/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

// extract to string
#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

std::vector<uint> to_uint_vector(const std::string& s, char delimiter = ',') {
  std::string s_without_sp = s;
  // printf("s_without_sp: %s\n", s_without_sp.c_str());
  // if (delimiter != ' ') {
  //   s_without_sp = std::regex_replace(s, std::regex(R"(\s+)"), "");
  // }
  // printf("s_without_sp: %s\n", s_without_sp.c_str());
  std::vector<uint> tokens;
  std::string token;
  std::istringstream token_stream(s_without_sp);
  while (getline(token_stream, token, delimiter)) {
    tokens.push_back(std::stoul(token));
  }
  return tokens;
}

int main() {

  std::string name = "1,    2, 3, 5";
  auto tokens = to_uint_vector(name);
  for (auto i : tokens) {
    std::cout << i << std::endl;
  }

  return 0;
}