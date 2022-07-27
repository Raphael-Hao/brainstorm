/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#ifndef BRT_RUNTIME_DISPATCH_H_
#define BRT_RUNTIME_DISPATCH_H_

#include <brt/runtime/cuda_utils.h>

#include <memory>

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

#endif  // BRT_RUNTIME_DISPATCH_H_
