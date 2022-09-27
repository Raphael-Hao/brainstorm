# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import os

fabric_types = os.environ.get("BRT_CAPTURE_FABRIC_TYPE")
fabric_types = fabric_types.split(":") if fabric_types else []
print(fabric_types)