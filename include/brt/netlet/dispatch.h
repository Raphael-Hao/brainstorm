/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#pragma once
#include <brt/common/cuda_utils.h>

namespace brt {
namespace netlet {
class Dispatcher {
 public:
  virtual ~Dispatcher() = default;

  virtual void dispatch(const std::string& message) = 0;
};

class HeteroDispatcher : Dispatcher {
 public:
  virtual ~HeteroDispatcher() = default;

  virtual void dispatch(const std::string& message) = 0;
};

class HomoDispatcher : Dispatcher {
 public:
  virtual ~HomoDispatcher() = default;

  virtual void dispatch(const std::string& message) = 0;
};

class DispatcherFactory {
 public:
  ~DispatcherFactory() = default;

  static std::shared_ptr<Dispatcher> create();
};
}  // namespace netlet
}  // namespace brt