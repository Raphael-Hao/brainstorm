
# adding executable for all tests

file(GLOB HFUSION_SOURCE CONFIGURE_DEPENDS "*.cu")
foreach(file_name ${HFUSION_SOURCE})
  get_filename_component(file_name_no_ext ${file_name} NAME_WE)
  message(STATUS "Build " "${file_name_no_ext}")
  add_executable(${file_name_no_ext} ${file_name})
  target_include_directories(${file_name_no_ext} PUBLIC "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(${file_name_no_ext} PUBLIC ${BRT_LINKER_LIBS})
endforeach(file_name )
