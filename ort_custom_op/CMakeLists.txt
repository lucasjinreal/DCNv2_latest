cmake_minimum_required(VERSION 3.10)
project (customop)
add_definitions(-std=c++11)


set(TEST_SOURCE dcn_op_test.cc)
set(HEADER dcn_v2_op.h dcn_v2_im2col_cpu.h)
set(SOURCE dcn_v2_op.h dcn_v2_im2col_cpu.cpp)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories(/data/xingchen/software/eigen-3.3.9/)
include_directories(/root/Microsoft.ML.OnnxRuntime.1.7.0/build/native/include)

#Include path to header files for Custom Op Test
#include_directories(<PATH_TO_ONNXRUNTIME_TEST_UTIL_INCLUDE_DIR>)

#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS /root/Microsoft.ML.OnnxRuntime.1.7.0/runtimes/linux-x64/native)
target_link_libraries(customop PUBLIC ${ONNXRUNTIME_LIBRARY})
