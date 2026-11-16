# 模块：Backend CMake 构建入口
#
# 职责：
#   构建 Backend/Kernel 公共契约与框架源码（runtime_backend 静态库），并
#   提供公共头编译/注册表可执行检查。后续 P1/P4 在该目标上追加 mock/ocl
#   后端 object library。
#
# 边界：
#   - 只编译 Runtime/backend 下已存在的框架源码；不包含 Loader/Engine；
#   - 复用 Graph 静态库（关闭 Python 与 Graph 自身测试）；
#   - 不下载依赖、不执行设备代码；
#   - 不修改源码、不生成模型包。
#
# 依赖：
#   CMake 3.20+、C++17 编译器、Graph 源目录。
#
# 用法：
#   cmake -S Runtime/backend -B Runtime/backend/build -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd \
#     -DCMAKE_AR=d:/Mobile_Net/Graph/tools/zig_ar.cmd \
#     -DCMAKE_RANLIB=d:/Mobile_Net/Graph/tools/zig_ranlib.cmd
#   cmake --build Runtime/backend/build
#   ctest --test-dir Runtime/backend/build --output-on-failure

cmake_minimum_required(VERSION 3.20)

# 独立构建时声明工程；被 Runtime/CMakeLists.txt 作为子目录加入时复用父工程。
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MobileNetBackend LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MOBILENET_BACKEND_BUILD_TESTS "Build backend tests" ON)

# 非 MSVC 工具链的归档器覆盖。
#
# 背景：使用 Zig 等包装编译器时，CMake 探测阶段可能把 CMAKE_AR 置为
# CMAKE_AR-NOTFOUND。调用方可显式传入 -DBACKEND_AR=... 与
# -DBACKEND_RANLIB=...，在生成构建规则前覆盖缓存变量。
if(BACKEND_AR)
  set(CMAKE_AR "${BACKEND_AR}" CACHE FILEPATH "Archiver" FORCE)
  set(CMAKE_CXX_COMPILER_AR "${BACKEND_AR}" CACHE FILEPATH
      "Compiler archiver" FORCE)
endif()
if(BACKEND_RANLIB)
  set(CMAKE_RANLIB "${BACKEND_RANLIB}" CACHE FILEPATH "Ranlib" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB "${BACKEND_RANLIB}" CACHE FILEPATH
      "Compiler ranlib" FORCE)
endif()

# 仓库根目录：Runtime/backend 的上两级。
get_filename_component(MOBILENET_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# Runtime 后续会由 Runtime/CMakeLists.txt 链接 Graph 静态库；此处允许
# 独立构建，同时在已被上层加入时复用既有 graph 目标。
if(NOT TARGET graph)
  set(GRAPH_BUILD_PYTHON OFF CACHE BOOL
      "Disable Graph Python binding for Runtime backend builds" FORCE)
  set(GRAPH_BUILD_TESTS OFF CACHE BOOL
      "Disable Graph tests for Runtime backend builds" FORCE)
  add_subdirectory("${MOBILENET_ROOT}/Graph"
                   "${CMAKE_BINARY_DIR}/graph")
endif()

# Backend 框架静态库：公共契约 + 注册表 + 执行型 OpEnv + 边界类型映射。
add_library(
  runtime_backend
  STATIC
  backend_registry.cc
  exec_op_env.cc
  kernel_types.cc
)
target_include_directories(runtime_backend PUBLIC "${MOBILENET_ROOT}")
target_link_libraries(runtime_backend PUBLIC graph)

if(MSVC)
  target_compile_options(runtime_backend PRIVATE /W4)
else()
  target_compile_options(runtime_backend PRIVATE -Wall -Wextra)
endif()

if(MOBILENET_BACKEND_BUILD_TESTS)
  enable_testing()

  # 公共头编译检查 + 注册表/ABI 行为检查。
  add_executable(backend_header_check tests/header_check.cc)
  target_link_libraries(backend_header_check PRIVATE runtime_backend)
  if(MSVC)
    target_compile_options(backend_header_check PRIVATE /W4)
  else()
    target_compile_options(backend_header_check PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME backend_header_check COMMAND backend_header_check)
endif()
