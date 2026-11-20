// 模块：Input/Output 边界 OpEnv 实现
//
// 职责：
//   实现设备视图绑定与 H2D/D2H 传输；主机 Storage 每步从 attr 读取。
//
// 边界：
//   不做布局/精度转换；不拥有 Buffer；不修改图。
//
// 依赖：
//   Runtime/backend/ops/io_op_env.h、Runtime/backend/backend.h、
//   Runtime/backend/kernel_types.h、Graph/include/attr.h、storage.h、
//   tensor.h。

#include "Runtime/backend/ops/io_op_env.h"

#include <string>
#include <variant>

#include "Graph/include/attr.h"
#include "Graph/include/op.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/kernel_types.h"
#include "Utils/cpp/types.h"

namespace mobilenet {

IoOpEnv::IoOpEnv(runtime_graph::Op* op, Backend* backend)
    : ExecOpEnv(op), backend_(backend) {}

Status IoOpEnv::Init(runtime_graph::GraphEnv* graph_env) {
  Status status = ExecOpEnv::Init(graph_env);
  if (!status.ok()) {
    return status;
  }
  if (backend_ == nullptr || op_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "IoOpEnv::Init: missing backend or op");
  }

  const runtime_graph::Tensor* tensor = nullptr;
  if (op_->Type() == runtime_graph::OpType::kInput) {
    is_input_ = true;
    if (op_->DstTensors().size() != 1) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "op=" + op_->Name() +
                            ": input must have exactly one tensor");
    }
    tensor = op_->DstTensors()[0];
  } else if (op_->Type() == runtime_graph::OpType::kOutput) {
    is_input_ = false;
    if (op_->SrcTensors().size() != 1) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "op=" + op_->Name() +
                            ": output must have exactly one tensor");
    }
    tensor = op_->SrcTensors()[0];
  } else {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "op=" + op_->Name() +
                          ": not an Input/Output op");
  }

  const runtime_graph::Storage& storage = tensor->GetStorage();
  if (storage.data_ptr == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op_->Name() +
            ": device storage is not bound by Loader");
  }
  const DataType dtype = DataTypeFromPrecision(storage.precision);
  if (dtype == DataType::kUnknown) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        "op=" + op_->Name() + ": unsupported device precision");
  }
  device_view_.data = storage.data_ptr;
  device_view_.offset = storage.offset;
  device_view_.bytes = storage.byte_size;
  device_view_.shape = storage.shape;
  device_view_.dtype = dtype;
  device_view_.layout = storage.layout;
  device_view_.device = storage.device_type;
  return Status::Ok();
}

Status IoOpEnv::Forward() {
  const auto it = op_->Attrs().find("storage");
  if (it == op_->Attrs().end()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op_->Name() +
                          ": attr 'storage' is missing");
  }
  const auto* host = std::get_if<runtime_graph::Storage>(&it->second);
  if (host == nullptr || host->data_ptr == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op_->Name() +
            ": host storage is not bound for this step");
  }
  if (host->device_type != runtime_graph::DeviceType::kHost) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op_->Name() +
                          ": host storage must be device_type=kHost");
  }
  if (host->byte_size != device_view_.bytes) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op_->Name() +
                          ": host/device byte_size mismatch");
  }
  if (DataTypeFromPrecision(host->precision) != device_view_.dtype) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "op=" + op_->Name() +
                          ": host/device precision mismatch");
  }

  auto* buffer = static_cast<Buffer*>(device_view_.data);
  if (buffer == nullptr) {
    return MakeStatus(ErrorCode::kInternal,
                      "op=" + op_->Name() + ": null device buffer");
  }
  Stream* stream =
      step_context_ != nullptr ? step_context_->stream : nullptr;
  Status status = Status::Ok();
  // storage 可能绑定在 arena 的某个切片上，偏移必须交给后端传输接口；
  // OpenCL 的 cl_mem 是句柄，不能通过指针加法携带偏移。
  if (is_input_) {
    status = backend_->UploadAt(buffer->DevicePtr(), device_view_.offset,
                                host->data_ptr, host->byte_size, stream);
  } else {
    status = backend_->DownloadAt(host->data_ptr, buffer->DevicePtr(),
                                  device_view_.offset, host->byte_size,
                                  stream);
  }
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op_->Name());
  }
  return Status::Ok();
}

}  // namespace mobilenet
