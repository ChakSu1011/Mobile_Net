# 模块：Graph 模块 CMake 构建入口
#
# 职责：
#   构建 graph_common、graph_ir、graph_runtime、graph_python 四个目标，并注册
#   C++/Python 测试。graph_runtime 目标不得链接 LibTorch；graph_ir 与
#   graph_python 才允许依赖 torch。
#
# 边界：
#   - 本文件只负责编译与测试注册，不下载依赖；
#   - 正式构建需要与本机 Python ``torch`` 同版本、同 ABI 的 LibTorch；
#     无 LibTorch 的测试环境可使用 MOBILENET_GRAPH_HOST_TORCH_BRIDGE；
#   - 不修改源码、不运行模型推理。
#
# 依赖：
#   CMake 3.20+、C++17 编译器、LibTorch（或 torch Python 头文件）、pybind11、
#   Python 3.10+。
#
# 用法：
#   cmake -S Graph -B Build/cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build Build/cmake --config Release
#   ctest --test-dir Build/cmake -C Release --output-on-failure

cmake_minimum_required(VERSION 3.20)

project(MobileNetGraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MOBILENET_GRAPH_BUILD_TESTS "Build Graph unit and integration tests" ON)
option(MOBILENET_GRAPH_BUILD_PYTHON "Build the _graph Python extension" ON)
option(
  MOBILENET_GRAPH_HOST_TORCH_BRIDGE
  "Build the Python extension against Python torch headers without linking LibTorch"
  OFF
)

# 仓库根目录：Graph 的上一级。
get_filename_component(MOBILENET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# torch 依赖封装。
#
# 正式模式通过 find_package(Torch) 获取同版本 LibTorch；桥接模式只取
# Python torch 包中的头文件，用于无 MSVC/LibTorch 的测试环境。
add_library(mobilenet_torch INTERFACE)
if(MOBILENET_GRAPH_HOST_TORCH_BRIDGE)
  # 桥接模式不经过 TorchConfig，需自行声明“无 cmake_macros.h”。
  target_compile_definitions(
    mobilenet_torch INTERFACE C10_USING_CUSTOM_GENERATED_MACROS
  )
  find_package(Python3 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'include')"
    OUTPUT_VARIABLE MOBILENET_TORCH_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  target_include_directories(
    mobilenet_torch INTERFACE
    "${MOBILENET_TORCH_INCLUDE_DIR}"
    "${MOBILENET_TORCH_INCLUDE_DIR}/torch/csrc/api/include"
  )
else()
  # 优先使用调用方提供的 LibTorch；未提供时回退到本机 Python torch 的 CMake
  # 配置目录，保证图构建与 Python 侧使用同一份 torch。
  find_package(Torch QUIET)
  if(NOT Torch_FOUND)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE MOBILENET_TORCH_CMAKE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(APPEND CMAKE_PREFIX_PATH "${MOBILENET_TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)
  endif()
  target_link_libraries(mobilenet_torch INTERFACE torch)
endif()

# 运行时公共类型：Status、DataType、Kernel 描述。
#
# graph_common 依赖这些符号，但 graph_runtime 不依赖 torch，因此单独成库。
add_library(
  graph_base STATIC
  "${MOBILENET_ROOT}/Runtime/common/data_types.cc"
  "${MOBILENET_ROOT}/Runtime/common/status.cc"
  "${MOBILENET_ROOT}/Runtime/kernel/kernel_types.cc"
)
target_include_directories(graph_base PUBLIC "${MOBILENET_ROOT}")

# Graph 公共层：与 torch 无关的枚举、维度、符号形状、量化描述与 V1~V16 规则。
add_library(
  graph_common STATIC
  cpp/common/src/graph_common.cc
  cpp/common/src/validation_common.cc
)
target_include_directories(graph_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/include")
target_link_libraries(graph_common PUBLIC graph_base)

# RuntimeGraph：仅依赖 TensorDesc 与 Backend/Kernel 抽象，不链接 torch。
add_library(
  graph_runtime STATIC
  cpp/runtime/src/runtime_graph.cc
  cpp/runtime/src/runtime_graph_builder.cc
  cpp/runtime/src/runtime_graph_forward.cc
  cpp/runtime/src/runtime_graph_validator.cc
)
target_include_directories(graph_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/runtime/include")
target_link_libraries(graph_runtime PUBLIC graph_common)

# Graph IR：使用 at::Tensor 表达属性，链接 LibTorch/torch 头文件。
add_library(
  graph_ir STATIC
  cpp/ir/src/fusion_rule.cc
  cpp/ir/src/graph.cc
  cpp/ir/src/graph_builder.cc
  cpp/ir/src/graph_validator.cc
  cpp/ir/src/tensor_desc.cc
)
target_include_directories(graph_ir PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp/ir/include")
target_link_libraries(graph_ir PUBLIC graph_common mobilenet_torch)

# Python 绑定模块：输出 _graph.pyd / _graph.so。
if(MOBILENET_GRAPH_BUILD_PYTHON)
  if(NOT MOBILENET_GRAPH_HOST_TORCH_BRIDGE)
    find_package(Python3 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  endif()

  set(MOBILENET_GRAPH_PYTHON_SOURCES
      cpp/python/graph_python.cc
      cpp/python/qwen3_fixture.cc)
  if(MOBILENET_GRAPH_HOST_TORCH_BRIDGE)
    list(APPEND MOBILENET_GRAPH_PYTHON_SOURCES cpp/python/torch_shims.cc)
  endif()

  add_library(graph_python MODULE ${MOBILENET_GRAPH_PYTHON_SOURCES})
  set_target_properties(
    graph_python PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_graph"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
  )
  target_include_directories(graph_python PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp/python")
  target_link_libraries(graph_python PRIVATE graph_ir graph_runtime graph_common Python3::Module)

  if(MOBILENET_GRAPH_HOST_TORCH_BRIDGE)
    # 桥接模式不链接 LibTorch；THPVariable_Wrap 在运行时从 torch_python.dll 解析。
    target_compile_definitions(graph_python PRIVATE MOBILENET_GRAPH_HOST_TORCH_BRIDGE)
  else()
    find_library(
      MOBILENET_TORCH_PYTHON_LIBRARY
      NAMES torch_python
      PATHS "${TORCH_INSTALL_PREFIX}/lib"
    )
    if(MOBILENET_TORCH_PYTHON_LIBRARY)
      target_link_libraries(graph_python PRIVATE "${MOBILENET_TORCH_PYTHON_LIBRARY}")
    endif()
  endif()
  if(WIN32)
    set_target_properties(graph_python PROPERTIES SUFFIX ".pyd")
  endif()
endif()

# 测试：C++ RuntimeGraph 单测与 Python 单元/集成测试。
if(MOBILENET_GRAPH_BUILD_TESTS)
  enable_testing()

  add_executable(
    test_runtime_graph
    "${MOBILENET_ROOT}/Tests/unit/graph/test_runtime_graph.cc"
  )
  target_include_directories(
    test_runtime_graph PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/runtime/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/include"
  )
  target_link_libraries(test_runtime_graph PRIVATE graph_runtime graph_common)

  add_test(NAME graph_runtime_unit COMMAND test_runtime_graph)
  set_tests_properties(
    graph_runtime_unit PROPERTIES
    WORKING_DIRECTORY "${MOBILENET_ROOT}"
  )

  if(MOBILENET_GRAPH_BUILD_PYTHON)
    add_test(
      NAME graph_ir_python
      COMMAND "${CMAKE_COMMAND}" -E env
              "MOBILENET_GRAPH_PYTHON_DIR=$<TARGET_FILE_DIR:graph_python>"
              "${Python3_EXECUTABLE}" -I
              "${MOBILENET_ROOT}/Tests/unit/graph/test_graph_ir.py"
    )
    add_test(
      NAME graph_qwen3_cross_language
      COMMAND "${CMAKE_COMMAND}" -E env
              "MOBILENET_GRAPH_PYTHON_DIR=$<TARGET_FILE_DIR:graph_python>"
              "${Python3_EXECUTABLE}" -I
              "${MOBILENET_ROOT}/Tests/integration/graph/test_qwen3_cross_language.py"
    )
    set_tests_properties(
      graph_ir_python graph_qwen3_cross_language PROPERTIES
      WORKING_DIRECTORY "${MOBILENET_ROOT}"
    )
  endif()
endif()
