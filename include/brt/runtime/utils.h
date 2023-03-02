/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_RUNTIME_UTILS_H
#define BRT_RUNTIME_UTILS_H

#include <iostream>

#define CHECK(x)                                                                               \
  if (!(x)) {                                                                                  \
    std::cerr << "CHECK failed: " << #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::abort();                                                                                   \
  }

#define CHECK_EQ(x, y)                                                                  \
  if ((x) != (y)) {                                                                     \
    std::cerr << "CHECK_EQ failed: " << #x << " != " << #y << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                                 \
    std::abort();                                                                            \
  }

#define CHECK_NE(x, y)                                                                  \
  if ((x) == (y)) {                                                                     \
    std::cerr << "CHECK_NE failed: " << #x << " == " << #y << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                                 \
    std::abort();                                                                            \
  }

#define CHECK_LT(x, y)                                                                  \
  if ((x) >= (y)) {                                                                     \
    std::cerr << "CHECK_LT failed: " << #x << " >= " << #y << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                                 \
    std::abort();                                                                            \
  }

#define CHECK_GT(x, y)                                                                  \
  if ((x) <= (y)) {                                                                     \
    std::cerr << "CHECK_GT failed: " << #x << " <= " << #y << " at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                                 \
    std::abort();                                                                            \
  }

#define CHECK_LE(x, y)                                                                             \
  if ((x) > (y)) {                                                                                 \
    std::cerr << "CHECK_LE failed: " << #x << " > " << #y << " at " << __FILE__ << ":" << __LINE__ \
              << std::endl;                                                                        \
    std::abort();                                                                                       \
  }

#define CHECK_GE(x, y)                                                                             \
  if ((x) < (y)) {                                                                                 \
    std::cerr << "CHECK_GE failed: " << #x << " < " << #y << " at " << __FILE__ << ":" << __LINE__ \
              << std::endl;                                                                        \
    std::abort();                                                                                       \
  }

#endif
