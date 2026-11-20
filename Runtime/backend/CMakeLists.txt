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
option(MOBILENET_BACKEND_MOCK "Build the mock CPU backend" ON)
option(MOBILENET_BACKEND_OCL "Build the OpenCL backend" ON)

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

# 后端以 object library 编入，防止静态注册被链接器裁剪。
if(MOBILENET_BACKEND_MOCK)
  add_subdirectory(mock)
endif()
if(MOBILENET_BACKEND_OCL)
  add_subdirectory(ocl)
endif()

# Backend 框架静态库：公共契约 + 注册表 + 执行型 OpEnv + 算子规格与 Kernel。
add_library(
  runtime_backend
  STATIC
  backend_registry.cc
  exec_op_env.cc
  kernel_types.cc
  # Graph 的 graph 目标只编译 status.cc；DataTypeSize 在 common/data_types.cc，
  # 独立构建 Backend 时需一并编译。RuntimeCore 建立 common 目标后由上层替换。
  "${MOBILENET_ROOT}/Runtime/common/data_types.cc"
  ops/attr_utils.cc
  ops/op_validation.cc
  ops/op_registry.cc
  ops/kernel_op_env.cc
  ops/io_op_env.cc
  ops/op_env_factory.cc
  ops/gather/gather_spec.cc
  ops/rms_norm/rms_norm_spec.cc
  ops/linear/linear_spec.cc
  ops/linear_swi_glu/linear_swi_glu_spec.cc
  ops/linear_per_head_rmsnorm_rope/per_head_rmsnorm_rope_spec.cc
  ops/mega_attention/mega_attention_spec.cc
  ops/add/add_spec.cc
  ops/sample/sample_spec.cc
)

if(MOBILENET_BACKEND_MOCK)
  target_sources(runtime_backend PRIVATE $<TARGET_OBJECTS:backend_mock>)
endif()
if(MOBILENET_BACKEND_OCL)
  target_sources(runtime_backend PRIVATE $<TARGET_OBJECTS:backend_ocl>)
endif()

target_include_directories(runtime_backend PUBLIC "${MOBILENET_ROOT}")
target_link_libraries(runtime_backend PUBLIC graph)
if(MOBILENET_BACKEND_OCL AND UNIX)
  # OclApi 使用 dlopen/dlsym；Android/Linux 需要 libdl。
  target_link_libraries(runtime_backend PUBLIC dl)
endif()

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

  if(MOBILENET_BACKEND_MOCK)
    # Mock 后端契约测试：Buffer/传输/ConvertParam/Kernel/生命周期。
    # 用 WHOLE_ARCHIVE 链接，确保 register_mock.cc.obj 的静态注册器不被
    # 链接器裁剪；产品构建同样需要对后端库使用 whole-archive 或显式锚点。
    add_executable(backend_contract_test tests/backend_contract_test.cc)
    target_link_libraries(
      backend_contract_test PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>")
    if(MSVC)
      target_compile_options(backend_contract_test PRIVATE /W4)
    else()
      target_compile_options(backend_contract_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME backend_contract_test COMMAND backend_contract_test)

    # OpEnv/Kernel 执行链测试：Graph → OpEnvFactory → Forward → Destroy。
    add_executable(op_env_test tests/op_env_test.cc)
    target_link_libraries(
      op_env_test PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>")
    if(MSVC)
      target_compile_options(op_env_test PRIVATE /W4)
    else()
      target_compile_options(op_env_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME op_env_test COMMAND op_env_test)
  endif()

  if(MOBILENET_BACKEND_OCL)
    # SHA-256 主机测试：无需 OpenCL 设备。
    add_executable(ocl_sha256_test ocl/tests/ocl_sha256_test.cc)
    target_link_libraries(ocl_sha256_test PRIVATE runtime_backend)
    if(MSVC)
      target_compile_options(ocl_sha256_test PRIVATE /W4)
    else()
      target_compile_options(ocl_sha256_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ocl_sha256_test COMMAND ocl_sha256_test)

    # OCL 契约测试：无设备时返回 77，CTest 记为 Skipped。
    add_executable(ocl_contract_test ocl/tests/ocl_contract_test.cc)
    target_include_directories(
      ocl_contract_test PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/ocl/third_party")
    target_link_libraries(
      ocl_contract_test PRIVATE
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,runtime_backend>")
    if(MSVC)
      target_compile_options(ocl_contract_test PRIVATE /W4)
    else()
      target_compile_options(ocl_contract_test PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ocl_contract_test COMMAND ocl_contract_test)
    set_tests_properties(ocl_contract_test PROPERTIES
                         SKIP_RETURN_CODE 77)
  endif()
endif()
