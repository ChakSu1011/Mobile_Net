# 模块：Runtime CMake 构建入口
#
# 职责：
#   构建 Runtime 公共库（common/loader/scheduler/kvcache/destroy/executor/
#   engine）、Backend/Mock 插件与 Mock 端到端测试。
#
# 边界：
#   - 复用 Graph 静态库与 Runtime/backend 公共契约；
#   - 构建 Mock 与 OCL 后端，不包含 CUDA/Vulkan 产品后端；
#   - 不生成模型包、不下载依赖。
#
# 用法：
#   cmake -S Runtime -B Runtime/build -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=d:/Mobile_Net/Graph/tools/zig_cxx.cmd
#   cmake --build Runtime/build
#   ctest --test-dir Runtime/build --output-on-failure

cmake_minimum_required(VERSION 3.20)

get_filename_component(MOBILENET_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Zig 编译器包装器不会自动提供归档器；在 project() 之前补齐，避免
# CMAKE_AR-NOTFOUND 写入编译器信息。
if(CMAKE_CXX_COMPILER MATCHES "zig_cxx")
  if(NOT CMAKE_AR OR CMAKE_AR MATCHES "NOTFOUND")
    set(CMAKE_AR "${MOBILENET_ROOT}/Graph/tools/zig_ar.cmd"
        CACHE FILEPATH "Archiver" FORCE)
  endif()
  if(NOT CMAKE_RANLIB OR CMAKE_RANLIB MATCHES "NOTFOUND")
    set(CMAKE_RANLIB "${MOBILENET_ROOT}/Graph/tools/zig_ranlib.cmd"
        CACHE FILEPATH "Ranlib" FORCE)
  endif()
  if(NOT CMAKE_CXX_COMPILER_AR OR CMAKE_CXX_COMPILER_AR MATCHES "NOTFOUND")
    set(CMAKE_CXX_COMPILER_AR "${CMAKE_AR}"
        CACHE FILEPATH "Compiler archiver" FORCE)
  endif()
  if(NOT CMAKE_CXX_COMPILER_RANLIB OR
     CMAKE_CXX_COMPILER_RANLIB MATCHES "NOTFOUND")
    set(CMAKE_CXX_COMPILER_RANLIB "${CMAKE_RANLIB}"
        CACHE FILEPATH "Compiler ranlib" FORCE)
  endif()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MobileNetRuntime LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Graph 静态库（关闭 Python 绑定与 Graph 自身测试）。
if(NOT TARGET graph)
  set(GRAPH_BUILD_PYTHON OFF CACHE BOOL
      "Disable Graph Python binding for Runtime builds" FORCE)
  set(GRAPH_BUILD_TESTS OFF CACHE BOOL
      "Disable Graph tests for Runtime builds" FORCE)
  add_subdirectory("${MOBILENET_ROOT}/Graph"
                   "${CMAKE_BINARY_DIR}/graph")
endif()

# Backend 公共契约与 Mock 后端。
add_subdirectory(backend "${CMAKE_BINARY_DIR}/backend")

# Runtime 公共库：不重复编译 Graph/status 与 backend/data_types。
add_library(
  mobilenet_runtime
  STATIC
  common/resource_stack.cc
  common/json.cc
  common/sha256.cc
  common/file_util.cc
  loader/manifest.cc
  loader/ir_dump.cc
  loader/loader.cc
  scheduler/scheduler.cc
  kvcache/kv_cache_manager.cc
  destroy/destroyer.cc
  executor/executor.cc
  engine/engine.cc
)

target_include_directories(
  mobilenet_runtime PUBLIC "${MOBILENET_ROOT}")
target_link_libraries(
  mobilenet_runtime PUBLIC graph runtime_backend Threads::Threads)

if(MSVC)
  target_compile_options(mobilenet_runtime PRIVATE /W4)
else()
  target_compile_options(mobilenet_runtime PRIVATE -Wall -Wextra)
endif()

# 端到端测试：Qwen3-0.6B 单步前向与 PyTorch token 对齐（Mock/OCL）。
option(MOBILENET_BUILD_E2E_TESTS "Build mock/ocl end-to-end tests" ON)
if(MOBILENET_BUILD_E2E_TESTS)
  enable_testing()
  add_executable(mock_e2e_test tests/mock_e2e_test.cc)
  target_link_libraries(
    mock_e2e_test PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,mobilenet_runtime>"
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
    graph Threads::Threads)
  if(MSVC)
    target_compile_options(mock_e2e_test PRIVATE /W4)
  else()
    target_compile_options(mock_e2e_test PRIVATE -Wall -Wextra)
  endif()
  add_test(
    NAME mock_e2e_qwen3
    COMMAND mock_e2e_test
      "${MOBILENET_ROOT}/Models/Qwen3/artifacts/qwen3_0_6b" 0)

  # OCL 端到端测试：同一进程内用相同 prompt 分别跑 OCL 与 Mock，
  # 比较生成 token 与 PyTorch 参考值；无 OpenCL 设备时返回 77。
  if(TARGET backend_ocl AND TARGET backend_mock)
    add_executable(ocl_e2e_test backend/ocl/tests/ocl_e2e_test.cc)
    target_link_libraries(
      ocl_e2e_test PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,mobilenet_runtime>"
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
      graph Threads::Threads)
    if(MSVC)
      target_compile_options(ocl_e2e_test PRIVATE /W4)
    else()
      target_compile_options(ocl_e2e_test PRIVATE -Wall -Wextra)
    endif()
    add_test(
      NAME ocl_e2e_qwen3
      COMMAND ocl_e2e_test
        "${MOBILENET_ROOT}/Models/Qwen3/artifacts/qwen3_0_6b" 0)
    set_tests_properties(ocl_e2e_qwen3 PROPERTIES SKIP_RETURN_CODE 77)

    # Paged Attention：prefill + 多步 decode 自回归生成。
    add_executable(ocl_kv_e2e_test
      backend/ocl/tests/ocl_kv_e2e_test.cc)
    target_link_libraries(
      ocl_kv_e2e_test PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,mobilenet_runtime>"
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
      graph Threads::Threads)
    if(MSVC)
      target_compile_options(ocl_kv_e2e_test PRIVATE /W4)
    else()
      target_compile_options(ocl_kv_e2e_test PRIVATE -Wall -Wextra)
    endif()
    add_test(
      NAME ocl_kv_e2e_qwen3
      COMMAND ocl_kv_e2e_test
        "${MOBILENET_ROOT}/Models/Qwen3/artifacts/qwen3_0_6b_kv")
    set_tests_properties(ocl_kv_e2e_qwen3 PROPERTIES SKIP_RETURN_CODE 77)
    add_test(
      NAME ocl_kv_mode_mismatch
      COMMAND ocl_kv_e2e_test
        "${MOBILENET_ROOT}/Models/Qwen3/artifacts/qwen3_0_6b_kv" mismatch)
    set_tests_properties(ocl_kv_mode_mismatch PROPERTIES SKIP_RETURN_CODE 77)

    # 交互式 token REPL：终端输入 token id，输出自回归 token。
    add_executable(ocl_token_repl
      backend/ocl/tests/ocl_token_repl.cc)
    target_link_libraries(
      ocl_token_repl PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,mobilenet_runtime>"
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>"
      graph Threads::Threads)
    if(MSVC)
      target_compile_options(ocl_token_repl PRIVATE /W4)
    else()
      target_compile_options(ocl_token_repl PRIVATE -Wall -Wextra)
    endif()
  endif()
endif()
