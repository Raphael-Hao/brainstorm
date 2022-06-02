#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <unordered_map>

const std::unordered_map<std::string, size_t> size_of_str_type{
    {"char", sizeof(char)},         {"float", sizeof(float)},      {"double", sizeof(double)},
    {"int8_t", sizeof(int8_t)},     {"int16_t", sizeof(int16_t)},  {"int32_t", sizeof(int32_t)},
    {"int64_t", sizeof(int64_t)},   {"uint8_t", sizeof(uint8_t)},  {"uint16_t", sizeof(uint16_t)},
    {"uint32_t", sizeof(uint32_t)}, {"uint64_t", sizeof(uint64_t)}};

int main(int argc, char const* argv[]) {
  for (auto type_size : size_of_str_type) {
    std::cout << type_size.first << ": " << type_size.second << std::endl;
  }
  return 0;
}
