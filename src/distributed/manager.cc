/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */

#include "./manager.h"

namespace brt {
namespace distributed {
NCCLManager& NCCLManager::get_manager() {
  static NCCLManager manager;
  return manager;
}
}  // namespace distributed
}  // namespace brt
