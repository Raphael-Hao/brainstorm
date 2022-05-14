execute_process(COMMAND python "-c" "import torch.utils; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
# set(Torch_DIR "${TORCH_CMAKE_PREFIX_PATH}/Torch")
find_package(Torch CONFIG REQUIRED)