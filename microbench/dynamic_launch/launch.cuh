#pragma once

#include <cuda_runtime.h>
#include <string>

#define LAUNCH_CHECK(launch_flag, kFlag) \
  if (*launch_flag != kFlag) {           \
    return;                              \
  }

#define LAUNCH_CHECK_ASM(launch_flag, kFlag) \
  if (*launch_flag != kFlag) {               \
    asm("trap;");                            \
  }
