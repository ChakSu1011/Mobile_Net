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

# 把 kernels/<op>/*.cl 在配置期生成 C++ 原始字符串头，避免运行时读文件，
# 也避免在 .cc 中手写重复源码。
function(mobilenet_embed_ocl_source op file)
  file(READ "${file}" ocl_source_content)
  set(generated_header "${CMAKE_CURRENT_BINARY_DIR}/generated/${op}_source.h")
  file(WRITE "${generated_header}"
    "#pragma once\n"
    "namespace mobilenet {\n"
    "namespace ocl {\n"
    "inline constexpr char k${op}Source[] = R\"OCLSRC(\n"
    "${ocl_source_content}\n"
    ")OCLSRC\";\n"
    "}  // namespace ocl\n"
    "}  // namespace mobilenet\n")
endfunction()

mobilenet_embed_ocl_source(
  Gather "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gather/gather.cl")
mobilenet_embed_ocl_source(
  Add "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add/add.cl")
mobilenet_embed_ocl_source(
  Sample "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sample/sample.cl")
mobilenet_embed_ocl_source(
  RmsNorm "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm/rms_norm.cl")
mobilenet_embed_ocl_source(
  Linear "${CMAKE_CURRENT_SOURCE_DIR}/kernels/linear/linear.cl")
mobilenet_embed_ocl_source(
  LinearSwiGlu "${CMAKE_CURRENT_SOURCE_DIR}/kernels/linear_swi_glu/linear_swi_glu.cl")
mobilenet_embed_ocl_source(
  PerHeadRmsNormRope "${CMAKE_CURRENT_SOURCE_DIR}/kernels/linear_per_head_rmsnorm_rope/linear_per_head_rmsnorm_rope.cl")
mobilenet_embed_ocl_source(
  MegaAttention "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mega_attention/mega_attention.cl")

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
  ocl_kernel.cc
  ocl_kernel_factory.cc
  ocl_source_registry.cc
  ocl_backend.cc
  register_ocl.cc
  kernels/gather/gather_kernel.cc
  kernels/add/add_kernel.cc
  kernels/sample/sample_kernel.cc
  kernels/rms_norm/rms_norm_kernel.cc
  kernels/linear/linear_kernel.cc
  kernels/linear_swi_glu/linear_swi_glu_kernel.cc
  kernels/linear_per_head_rmsnorm_rope/linear_per_head_rmsnorm_rope_kernel.cc
  kernels/mega_attention/mega_attention_kernel.cc
)

target_include_directories(
  backend_ocl PRIVATE
  "${MOBILENET_ROOT}"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
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
