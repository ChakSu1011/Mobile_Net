# 模块：OCL 后端构建入口
#
# 职责：
#   以 object library 编译 OCL Backend、动态加载 API、Program 缓存、SHA-256
#   与自注册入口；由 Runtime/backend/CMakeLists.txt 合并进 runtime_backend。
#
# 边界：
#   - 只编译 ocl/ 源码；不定义测试；
#   - 不链接 OpenCL 库（运行时 dlopen/LoadLibrary）；
#   - 依赖父目录提供的 MOBILENET_ROOT 与 graph 目标。

add_library(
  backend_ocl
  OBJECT
  ocl_api.cc
  ocl_context.cc
  ocl_utils.cc
  ocl_sha256.cc
  ocl_buffer.cc
  ocl_program_cache.cc
  ocl_convert.cc
  ocl_backend.cc
  register_ocl.cc
)

target_include_directories(
  backend_ocl PRIVATE
  "${MOBILENET_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(backend_ocl PRIVATE graph)

if(UNIX)
  target_link_libraries(backend_ocl PRIVATE dl)
endif()

if(MSVC)
  target_compile_options(backend_ocl PRIVATE /W4)
else()
  target_compile_options(backend_ocl PRIVATE -Wall -Wextra)
endif()
