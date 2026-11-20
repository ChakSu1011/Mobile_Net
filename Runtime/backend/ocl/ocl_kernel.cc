// 模块：OCL Kernel 基类实现
//
// 职责：
//   实现 Prepare/Execute 两段式生命周期、参数绑定、工作尺寸、视图校验。
//
// 边界：
//   不实现算子数学；不调用 clFinish；不拥有 Context/ProgramCache。
//
// 依赖：
//   Runtime/backend/ocl/ocl_kernel.h、ocl_source_registry.h、
//   ocl_utils.h、ocl_buffer.h、Runtime/common/data_types.h。

#include "Runtime/backend/ocl/ocl_kernel.h"

#include <limits>
#include <string>
#include <utility>

#include "Runtime/backend/ocl/ocl_buffer.h"
#include "Runtime/backend/ocl/ocl_source_registry.h"
#include "Runtime/backend/ocl/ocl_utils.h"
#include "Runtime/common/data_types.h"

namespace mobilenet {
namespace ocl {

OclKernel::OclKernel(OclContext* context, OclProgramCache* programs,
                     std::string kernel_key)
    : context_(context),
      programs_(programs),
      kernel_key_(std::move(kernel_key)) {}

OclKernel::~OclKernel() { ReleaseKernel(); }

Status OclKernel::Execute(const runtime_graph::KernelContext& context) {
  if (!prepared_) {
    const Status status = Prepare(context);
    if (!status.ok()) {
      return status;
    }
    prepared_ = true;
  }
  return ExecuteImpl(context);
}

Status OclKernel::Compile(const char* entry,
                          const std::string& options) {
  const char* source = GetOclSource(kernel_key_.c_str());
  if (source == nullptr) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "OclKernel::Compile: no source for kernel_key=" +
                          kernel_key_);
  }
  return Compile(source, entry, options);
}

Status OclKernel::Compile(const char* source, const char* entry,
                          const std::string& options) {
  if (kernel_ != nullptr) {
    return Status::Ok();
  }
  if (context_ == nullptr || programs_ == nullptr || source == nullptr ||
      entry == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::Compile: missing context/program/entry");
  }
  cl_program program = nullptr;
  Status status = programs_->GetOrBuild(kernel_key_, source, options,
                                        &program);
  if (!status.ok()) {
    return status;
  }
  cl_int error = CL_SUCCESS;
  kernel_ = context_->functions()->clCreateKernel(program, entry, &error);
  if (error != CL_SUCCESS || kernel_ == nullptr) {
    kernel_ = nullptr;
    return CheckCl(error, "CreateKernel", entry);
  }
  return Status::Ok();
}

Status OclKernel::CompileExtra(const char* entry,
                               const std::string& options,
                               cl_kernel* kernel) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::CompileExtra: kernel output is null");
  }
  const char* source = GetOclSource(kernel_key_.c_str());
  if (source == nullptr) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "OclKernel::CompileExtra: no source for kernel_key=" +
                          kernel_key_);
  }
  cl_program program = nullptr;
  Status status = programs_->GetOrBuild(kernel_key_, source, options,
                                        &program);
  if (!status.ok()) {
    return status;
  }
  cl_int error = CL_SUCCESS;
  *kernel = context_->functions()->clCreateKernel(program, entry, &error);
  if (error != CL_SUCCESS || *kernel == nullptr) {
    *kernel = nullptr;
    return CheckCl(error, "CreateKernel", entry);
  }
  return Status::Ok();
}

Status OclKernel::SetMemArg(cl_kernel kernel, U32 index,
                            const runtime_graph::TensorView& view) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::SetMemArg: kernel is not compiled");
  }
  const auto* buffer = static_cast<const OclBuffer*>(view.data);
  if (buffer == nullptr || buffer->mem() == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::SetMemArg: null OclBuffer");
  }
  cl_mem mem = buffer->mem();
  const cl_int error = context_->functions()->clSetKernelArg(
      kernel, index, sizeof(cl_mem), &mem);
  return CheckCl(error, "SetKernelArg", "mem arg");
}

Status OclKernel::SetMemArg(U32 index,
                            const runtime_graph::TensorView& view) {
  return SetMemArg(kernel_, index, view);
}

Status OclKernel::SetIntArg(cl_kernel kernel, U32 index, I32 value) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::SetIntArg: kernel is not compiled");
  }
  const cl_int error = context_->functions()->clSetKernelArg(
      kernel, index, sizeof(I32), &value);
  return CheckCl(error, "SetKernelArg", "int arg");
}

Status OclKernel::SetIntArg(U32 index, I32 value) {
  return SetIntArg(kernel_, index, value);
}

Status OclKernel::SetLongArg(cl_kernel kernel, U32 index, I64 value) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::SetLongArg: kernel is not compiled");
  }
  const cl_int error = context_->functions()->clSetKernelArg(
      kernel, index, sizeof(I64), &value);
  return CheckCl(error, "SetKernelArg", "long arg");
}

Status OclKernel::SetFloatArg(cl_kernel kernel, U32 index, float value) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::SetFloatArg: kernel is not compiled");
  }
  const cl_int error = context_->functions()->clSetKernelArg(
      kernel, index, sizeof(float), &value);
  return CheckCl(error, "SetKernelArg", "float arg");
}

Status OclKernel::SetFloatArg(U32 index, float value) {
  return SetFloatArg(kernel_, index, value);
}

Status OclKernel::Enqueue1D(cl_kernel kernel, Usize global, Usize local) {
  if (kernel == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "OclKernel::Enqueue1D: kernel is not compiled");
  }
  if (global == 0 || local == 0 || global % local != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::Enqueue1D: invalid global/local size");
  }
  const size_t global_size[1] = {global};
  const size_t local_size[1] = {local};
  cl_int error = context_->functions()->clEnqueueNDRangeKernel(
      context_->queue(), kernel, 1, nullptr, global_size, local_size, 0,
      nullptr, nullptr);
  if (error != CL_SUCCESS) {
    return CheckCl(error, "EnqueueNDRangeKernel", kernel_key_);
  }
  error = context_->functions()->clFlush(context_->queue());
  return CheckCl(error, "Flush", kernel_key_);
}

Status OclKernel::Enqueue1D(Usize global, Usize local) {
  return Enqueue1D(kernel_, global, local);
}

void OclKernel::ReleaseExtraKernel(cl_kernel* kernel) {
  if (kernel != nullptr && *kernel != nullptr && context_ != nullptr &&
      context_->functions() != nullptr &&
      context_->functions()->clReleaseKernel != nullptr) {
    (void)context_->functions()->clReleaseKernel(*kernel);
    *kernel = nullptr;
  }
}

Status OclKernel::ValidateView(const runtime_graph::TensorView& view,
                               DataType dtype, const char* role) const {
  if (view.data == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel: null tensor data for " +
                          std::string(role != nullptr ? role : "?"));
  }
  if (view.dtype != dtype) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "OclKernel: dtype mismatch for " +
                          std::string(role != nullptr ? role : "?"));
  }
  if (view.layout != runtime_graph::Layout::kRowMajor ||
      view.device != runtime_graph::DeviceType::kDevice) {
    return MakeStatus(ErrorCode::kUnsupportedLayout,
                      "OclKernel: layout/device mismatch for " +
                          std::string(role != nullptr ? role : "?"));
  }
  const U32 element_size = DataTypeSize(dtype);
  if (element_size == 0 || view.bytes == 0 ||
      view.offset % element_size != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel: invalid bytes/offset for " +
                          std::string(role != nullptr ? role : "?"));
  }
  const I64 elements = view.shape.GetShapeSize();
  if (elements <= 0 ||
      static_cast<U64>(elements) * element_size != view.bytes) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel: shape/bytes mismatch for " +
                          std::string(role != nullptr ? role : "?"));
  }
  return Status::Ok();
}

Status OclKernel::ElementOffset(const runtime_graph::TensorView& view,
                                DataType dtype, I32* offset) const {
  if (offset == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::ElementOffset: null output");
  }
  const U32 element_size = DataTypeSize(dtype);
  if (element_size == 0 || view.offset % element_size != 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::ElementOffset: unaligned byte offset");
  }
  const U64 element_offset = view.offset / element_size;
  if (element_offset >
      static_cast<U64>(std::numeric_limits<I32>::max())) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "OclKernel::ElementOffset: offset exceeds int32");
  }
  *offset = static_cast<I32>(element_offset);
  return Status::Ok();
}

I64 OclKernel::ElementCount(const runtime_graph::TensorView& view) {
  return view.shape.GetShapeSize();
}

I64 OclKernel::EffectiveRows(const runtime_graph::KernelContext& context,
                             I64 static_rows) const {
  if (static_rows <= 0 || !context.packed_tokens || context.batch != 1 ||
      context.token_count <= 0) {
    return static_rows;
  }
  const I64 tokens = static_cast<I64>(context.token_count);
  return tokens < static_rows ? tokens : static_rows;
}

void OclKernel::ReleaseKernel() {
  if (kernel_ != nullptr && context_ != nullptr &&
      context_->functions() != nullptr &&
      context_->functions()->clReleaseKernel != nullptr) {
    (void)context_->functions()->clReleaseKernel(kernel_);
  }
  kernel_ = nullptr;
}

}  // namespace ocl
}  // namespace mobilenet
