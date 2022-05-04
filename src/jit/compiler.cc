/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include "./compiler.h"
namespace brt {
namespace jit {

CUDACompiler& CUDACompiler::get_compiler() {
  static CUDACompiler instance;
  return instance;
}
}  // namespace jit
}  // namespace brt
