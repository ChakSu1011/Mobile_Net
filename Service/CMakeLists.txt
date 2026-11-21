# 模块：Service 构建入口
#
# 职责：
#   构建阶段二服务层：Tokenizer 与 Session 模块，以及它们的测试。
#
# 边界：
#   - 复用 Runtime 与 Graph 静态库；
#   - 关闭 Runtime/Backend 自带的耗时端到端测试；
#   - 不下载依赖、不生成模型包。
#
# 用法：
#   cmake -S Service -B Service/build -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
#   cmake --build Service/build
#   ctest --test-dir Service/build --output-on-failure

cmake_minimum_required(VERSION 3.20)

project(MobileNetService LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_filename_component(MOBILENET_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

find_package(Threads REQUIRED)

# Zig 编译器包装器不会自动提供归档器；为 C/CXX 静态库补齐。
if(CMAKE_CXX_COMPILER MATCHES "zig_cxx" OR
   CMAKE_C_COMPILER MATCHES "zig_cc")
  set(CMAKE_AR "${MOBILENET_ROOT}/Graph/tools/zig_ar.cmd"
      CACHE FILEPATH "Archiver" FORCE)
  set(CMAKE_RANLIB "${MOBILENET_ROOT}/Graph/tools/zig_ranlib.cmd"
      CACHE FILEPATH "Ranlib" FORCE)
  set(CMAKE_C_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH
      "C compiler archiver" FORCE)
  set(CMAKE_C_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH
      "C compiler ranlib" FORCE)
  set(CMAKE_CXX_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH
      "C++ compiler archiver" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH
      "C++ compiler ranlib" FORCE)
endif()

# Runtime 以子目录方式构建；关闭其自带的重型端到端测试与后端测试。
set(MOBILENET_BUILD_E2E_TESTS OFF CACHE BOOL
    "Disable Runtime end-to-end tests for Service builds" FORCE)
set(MOBILENET_BACKEND_BUILD_TESTS OFF CACHE BOOL
    "Disable backend tests for Service builds" FORCE)

if(NOT TARGET mobilenet_runtime)
  add_subdirectory("${MOBILENET_ROOT}/Runtime"
                   "${CMAKE_BINARY_DIR}/runtime")
endif()

enable_testing()

add_subdirectory(tokenizer "${CMAKE_BINARY_DIR}/tokenizer")
add_subdirectory(session "${CMAKE_BINARY_DIR}/session")
