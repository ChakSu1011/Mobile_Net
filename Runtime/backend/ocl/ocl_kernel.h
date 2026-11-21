// 模块：OCL Kernel 基类
//
// 职责：
//   为各融合算子 OCL Kernel 提供统一生命周期：首次 Execute 前 Prepare
//   （编译 Program、创建 cl_kernel），之后 ExecuteImpl 只绑定参数/下发；
//   提供参数绑定、工作尺寸与视图校验工具。
//
// 边界：
//   - 不定义算子数学、不解析 attr 语义（由 OpSpec/派生类负责）；
//   - 不拥有 Context/ProgramCache；持有 cl_kernel 并在析构释放；
//   - 不调用 clFinish；同步由 Backend/Executor 负责。
//
// 依赖：
//   Runtime/backend/kernel.h、ocl_context.h、ocl_program_cache.h、
//   ocl_buffer.h、Runtime/common/status.h、Utils/cpp/types.h。

#ifndef MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_H_
#define MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_H_

#include <string>

#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_context.h"
#include "Runtime/backend/ocl/ocl_context.h"
#include "Runtime/backend/ocl/ocl_program_cache.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace ocl {

// OCL Kernel 基类。
class OclKernel : public Kernel {
 public:
  // 创建 Kernel 基类。
  //
  // 参数：
  //   context：非拥有 Context；
  //   programs：非拥有 Program 缓存；
  //   kernel_key：用于 Program 缓存键与错误定位。
  OclKernel(OclContext* context, OclProgramCache* programs,
            std::string kernel_key);

  ~OclKernel() override;

  OclKernel(const OclKernel&) = delete;
  OclKernel& operator=(const OclKernel&) = delete;
  OclKernel(OclKernel&&) = delete;
  OclKernel& operator=(OclKernel&&) = delete;

  // 执行一次节点计算。
  //
  // 首次调用先执行 Prepare；成功后再调用 ExecuteImpl。
  // 参数：
  //   context：节点执行上下文。
  // 返回：
  //   成功返回 kOk；Prepare/ExecuteImpl 的错误码原样返回。
  Status Execute(const runtime_graph::KernelContext& context) final;

 protected:
  // 加载期/首次执行前的准备。
  virtual Status Prepare(
      const runtime_graph::KernelContext& context) = 0;

  // 执行实际的计算与下发。
  virtual Status ExecuteImpl(
      const runtime_graph::KernelContext& context) = 0;

  // 编译/获取 Program 并创建 cl_kernel（按 kernel_key 查源码）。
  Status Compile(const char* entry, const std::string& options);

  // 显式源码版本的 Compile。
  Status Compile(const char* source, const char* entry,
                 const std::string& options);

  // 按 kernel_key 的源码创建额外 cl_kernel（用于一个算子多个入口）。
  //
  // 参数：
  //   entry：入口函数名；
  //   options：编译选项；
  //   kernel：输出参数，成功时写入非空句柄。
  // 返回：
  //   成功返回 kOk；源码缺失/编译失败返回对应错误。
  Status CompileExtra(const char* entry, const std::string& options,
                      cl_kernel* kernel);

  // 绑定 cl_mem 参数（指定 kernel）。
  Status SetMemArg(cl_kernel kernel, U32 index,
                   const runtime_graph::TensorView& view);

  // 绑定 cl_mem 参数（使用主 kernel）。
  Status SetMemArg(U32 index, const runtime_graph::TensorView& view);

  // 绑定 int 参数（指定 kernel）。
  Status SetIntArg(cl_kernel kernel, U32 index, I32 value);

  // 绑定 int 参数（使用主 kernel）。
  Status SetIntArg(U32 index, I32 value);

  // 绑定 64 位整数参数（指定 kernel）。
  Status SetLongArg(cl_kernel kernel, U32 index, I64 value);

  // 绑定 float 参数（指定 kernel）。
  Status SetFloatArg(cl_kernel kernel, U32 index, float value);

  // 绑定 float 参数（使用主 kernel）。
  Status SetFloatArg(U32 index, float value);

  // 下发一维 NDRange 并 flush（指定 kernel）。
  Status Enqueue1D(cl_kernel kernel, Usize global, Usize local);

  // 下发一维 NDRange 并 flush（使用主 kernel）。
  Status Enqueue1D(Usize global, Usize local);

  // 释放额外 kernel；幂等。
  void ReleaseExtraKernel(cl_kernel* kernel);

  // 校验 TensorView 的设备/精度/布局/字节数。
  Status ValidateView(const runtime_graph::TensorView& view, DataType dtype,
                      const char* role) const;

  // 把字节 offset 转为元素 offset（检查整除与 int32 范围）。
  Status ElementOffset(const runtime_graph::TensorView& view, DataType dtype,
                       I32* offset) const;

  // 计算 tensor 元素数（按 shape）。
  static I64 ElementCount(const runtime_graph::TensorView& view);

  // 计算本步有效行数。
  //
  // 功能：
  //   Executor 执行（`packed_tokens=true`）时，把按静态 shape 计算的行数
  //   截断到本步 packed token 总数，跳过 padding 行；直接调用 Kernel 的
  //   测试/基准（`packed_tokens=false`）保持全量计算。
  //
  // 参数：
  //   context：节点执行上下文；
  //   static_rows：按静态 shape 计算的行数，必须为正。
  // 返回：
  //   有效行数，范围为 [1, static_rows]。
  I64 EffectiveRows(const runtime_graph::KernelContext& context,
                    I64 static_rows) const;

  // 非拥有 Context/ProgramCache。
  OclContext* context_ = nullptr;
  OclProgramCache* programs_ = nullptr;

  // 缓存键与 cl_kernel。
  std::string kernel_key_;
  cl_kernel kernel_ = nullptr;

 private:
  // 释放 cl_kernel；幂等。
  void ReleaseKernel();

  // 是否已完成 Prepare。
  Bool prepared_ = false;
};

}  // namespace ocl
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_BACKEND_OCL_OCL_KERNEL_H_
