# 模块：Runtime Graph CMake 构建入口
#
# 职责：
#   构建 runtime_graph 静态库、_graph_fixed Python 扩展、C++ 单测和 Python
#   单测；runtime_graph 不依赖 torch/ATen/GPU/OCL。
#
# 边界：
#   - 本文件只负责编译与测试注册，不下载依赖；
#   - Python 扩展需要 Python 3.10+、pybind11 和 C++17 编译器；
#   - 不修改源码、不运行模型推理。
#
# 依赖：
#   CMake 3.20+、C++17 编译器、Python 3.10+、pybind11。
#
# 用法：
#   cmake -S Graph_Fixed -B Build/graph_fixed -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build Build/graph_fixed
#   ctest --test-dir Build/graph_fixed --output-on-failure

cmake_minimum_required(VERSION 3.20)

project(MobileNetGraphFixed LANGUAGES CXX)

# 非 MSVC 工具链的归档器覆盖。
#
# 背景：使用 Zig 等包装编译器时，CMake 探测阶段可能把 CMAKE_AR 置为
# CMAKE_AR-NOTFOUND。调用方可显式传入 -DGRAPH_FIXED_AR=... 和
# -DGRAPH_FIXED_RANLIB=...，在生成构建规则前覆盖缓存变量。
if(GRAPH_FIXED_AR)
  set(CMAKE_AR "${GRAPH_FIXED_AR}" CACHE FILEPATH "Archiver" FORCE)
  set(CMAKE_CXX_COMPILER_AR "${GRAPH_FIXED_AR}" CACHE FILEPATH
      "Compiler archiver" FORCE)
endif()
if(GRAPH_FIXED_RANLIB)
  set(CMAKE_RANLIB "${GRAPH_FIXED_RANLIB}" CACHE FILEPATH "Ranlib" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB "${GRAPH_FIXED_RANLIB}" CACHE FILEPATH
      "Compiler ranlib" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GRAPH_FIXED_BUILD_PYTHON "Build the _graph_fixed Python extension" ON)
option(GRAPH_FIXED_BUILD_TESTS "Build C++ and Python tests" ON)

# 仓库根目录：Graph_Fixed 的上一级。
get_filename_component(MOBILENET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Runtime Graph 静态库：不链接 torch，只依赖 Runtime/common/status。
add_library(
  graph_fixed STATIC
  src/attr.cc
  src/graph.cc
  src/graph_env.cc
  src/op.cc
  src/op_env.cc
  src/op_type.cc
  src/shape.cc
  src/storage.cc
  src/tensor.cc
  src/types.cc
  "${MOBILENET_ROOT}/Runtime/common/status.cc"
)
target_include_directories(
  graph_fixed PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${MOBILENET_ROOT}"
)

# 静态库会被链接进 Python MODULE；Linux/macOS 上必须使用位置无关代码，
# Windows 上该属性不产生额外编译选项。
set_target_properties(graph_fixed PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(graph_fixed PRIVATE /W4)
else()
  target_compile_options(graph_fixed PRIVATE -Wall -Wextra)
endif()

# Python 扩展：输出 _graph_fixed.pyd / _graph_fixed.so。
if(GRAPH_FIXED_BUILD_PYTHON)
  find_package(Python3 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE GRAPH_FIXED_PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(APPEND CMAKE_PREFIX_PATH "${GRAPH_FIXED_PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
  endif()

  add_library(graph_fixed_python MODULE python/graph_fixed_python.cc)
  set_target_properties(
    graph_fixed_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_graph_fixed"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
  )
  if(WIN32)
    set_target_properties(graph_fixed_python PROPERTIES SUFFIX ".pyd")
  endif()
  target_link_libraries(
    graph_fixed_python PRIVATE graph_fixed pybind11::module
  )
endif()

# 测试：C++ 单测与 Python 单测。
if(GRAPH_FIXED_BUILD_TESTS)
  enable_testing()

  add_executable(graph_fixed_test tests/graph_fixed_test.cc)
  target_link_libraries(graph_fixed_test PRIVATE graph_fixed)
  add_test(NAME graph_fixed_unit COMMAND graph_fixed_test)
  set_tests_properties(
    graph_fixed_unit PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  if(TARGET graph_fixed_python)
    add_test(
      NAME graph_fixed_python
      COMMAND "${CMAKE_COMMAND}" -E env
              "GRAPH_FIXED_PYTHON_DIR=$<TARGET_FILE_DIR:graph_fixed_python>"
              "${Python3_EXECUTABLE}" -I
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graph_fixed.py"
    )
    set_tests_properties(
      graph_fixed_python PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
endif()
