
add_executable(batch_matmul thor/batch_matmul_1024_512.cu)

target_include_directories(batch_matmul PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(batch_matmul ${BRT_LINKER_LIBS})