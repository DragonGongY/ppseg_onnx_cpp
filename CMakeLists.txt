cmake_minimum_required(VERSION 3.10.0)
project(ppseg VERSION 0.1.0 LANGUAGES C CXX)

set(ONNX_ROOT_DIR "/home/xxx/workspace/thirdparts/onnxruntime-linux-x64-1.16.3")
find_package(OpenCV REQUIRED)

add_executable(ppseg main.cpp ppseg.cpp)

target_include_directories(ppseg PRIVATE ${ONNX_ROOT_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppseg PRIVATE ${ONNX_ROOT_DIR}/lib/libonnxruntime.so ${OpenCV_LIBS})
