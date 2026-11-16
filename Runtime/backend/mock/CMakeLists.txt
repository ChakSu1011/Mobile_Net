# 模块：Mock 后端构建入口
#
# 职责：
#   以 object library 编译 Mock 后端、CPU 参考 Kernel 与自注册入口；由
#   Runtime/backend/CMakeLists.txt 合并进 runtime_backend，防止静态注册被
#   链接器裁剪。
#
# 边界：
#   - 只编译 mock/ 下的源码；不定义测试；
#   - 不初始化设备、不生成模型包；
#   - 依赖父目录提供的 MOBILENET_ROOT 与 graph 目标。
#
# 依赖：
#   CMake 3.20+、C++17 编译器、Graph 静态库。

add_library(
  backend_mock
  OBJECT
  mock_backend.cc
  mock_buffer.cc
  half_utils.cc
  register_mock.cc
  kernels/kernel_common.cc
  kernels/mock_kernel_factory.cc
  kernels/gather_kernel.cc
  kernels/rms_norm_kernel.cc
  kernels/linear_kernel.cc
  kernels/linear_swi_glu_kernel.cc
  kernels/per_head_rmsnorm_rope_kernel.cc
  kernels/mega_attention_kernel.cc
  kernels/add_kernel.cc
  kernels/sample_kernel.cc
)

target_include_directories(backend_mock PRIVATE "${MOBILENET_ROOT}")
target_link_libraries(backend_mock PRIVATE graph)

if(MSVC)
  target_compile_options(backend_mock PRIVATE /W4)
else()
  target_compile_options(backend_mock PRIVATE -Wall -Wextra)
endif()
