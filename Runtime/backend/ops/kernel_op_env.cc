// 模块：通用计算 OpEnv 实现
//
// 职责：
//   实现算子契约校验、Kernel 创建、张量视图缓存与 KernelContext 组装。
//
// 边界：
//   不做算子数学、不做 Buffer 分配、不做 H2D/D2H；不拥有 Op/Backend。
//
// 依赖：
//   Runtime/backend/ops/kernel_op_env.h、kernel_types.h、
//   Graph/include/graph_env.h、storage.h、tensor.h。

#include "Runtime/backend/ops/kernel_op_env.h"

#include <string>

#include "Graph/include/graph_env.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/kernel.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/backend/ops/attr_utils.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace {

// workspace 共享存储键前缀。
constexpr C8 kWorkspacePrefix[] = "workspace.";

}  // namespace

// 保存 op/backend/spec 借用指针。
KernelOpEnv::KernelOpEnv(runtime_graph::Op* op, Backend* backend,
                         const OpSpec* spec)
    : ExecOpEnv(op), backend_(backend), spec_(spec) {}

// 加载期初始化：校验 spec、缓存输入输出 TensorView、创建 Kernel、登记 workspace。
Status KernelOpEnv::Init(runtime_graph::GraphEnv* graph_env) {
  Status status = ExecOpEnv::Init(graph_env);
  if (!status.ok()) {
    return status;
  }
  if (backend_ == nullptr || spec_ == nullptr || op_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "KernelOpEnv::Init: missing backend/op/spec");
  }
  status = spec_->validate(*op_);
  if (!status.ok()) {
    return status;
  }

  inputs_.clear();
  inputs_.reserve(op_->SrcTensors().size());
  for (Usize i = 0; i < op_->SrcTensors().size(); ++i) {
    runtime_graph::TensorView view;
    status = BuildView(*op_->SrcTensors()[i], "input", &view);
    if (!status.ok()) {
      return status;
    }
    inputs_.push_back(view);
  }
  outputs_.clear();
  outputs_.reserve(op_->DstTensors().size());
  for (Usize i = 0; i < op_->DstTensors().size(); ++i) {
    runtime_graph::TensorView view;
    status = BuildView(*op_->DstTensors()[i], "output", &view);
    if (!status.ok()) {
      return status;
    }
    outputs_.push_back(view);
  }

  KernelDesc desc;
  status = spec_->build_kernel_desc(*op_, &desc);
  if (!status.ok()) {
    return status;
  }
  status = backend_->CreateKernel(desc, &kernel_);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op_->Name());
  }

  if (spec_->need_workspace) {
    if (graph_env_ == nullptr) {
      return MakeStatus(
          ErrorCode::kStateError,
          "op=" + op_->Name() +
              ": workspace requested but GraphEnv is null");
    }
    runtime_graph::Storage workspace;
    workspace.precision = runtime_graph::Precision::kUInt8;
    workspace.layout = runtime_graph::Layout::kRowMajor;
    workspace.device_type = runtime_graph::DeviceType::kDevice;
    workspace.byte_size = spec_->workspace_bytes;
    workspace.reusable = false;
    status = graph_env_->AddSharedStorage(
        std::string(kWorkspacePrefix) + op_->Name(), workspace);
    if (!status.ok()) {
      return PrefixStatus(status, "op=" + op_->Name());
    }
  }
  return Status::Ok();
}

// 执行节点：填充 KernelContext（含 AttentionContext/元数据），调用 Kernel::Execute。
Status KernelOpEnv::Forward() {
  if (step_context_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "op=" + op_->Name() +
                          ": StepContext is not set");
  }
  if (kernel_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "op=" + op_->Name() + ": Kernel is not created");
  }

  runtime_graph::KernelContext context;
  context.inputs = inputs_.data();
  context.input_count = static_cast<U32>(inputs_.size());
  context.outputs = outputs_.data();
  context.output_count = static_cast<U32>(outputs_.size());
  context.attrs = &op_->Attrs();
  context.backend = step_context_->backend != nullptr
                        ? step_context_->backend
                        : backend_;
  context.stream = step_context_->stream;
  context.batch = step_context_->batch;
  context.token_count = step_context_->packed_tokens;
  context.seq_len = step_context_->packed_tokens;
  // Executor 按 packed 布局写入 token；Kernel 可据此跳过静态 padding 行。
  context.packed_tokens = true;
  if (!outputs_.empty() && outputs_[0].shape.max_dim >= 3) {
    context.seq_len =
        static_cast<I32>(outputs_[0].shape.dims[1]);
  }

  runtime_graph::AttentionContext attention;
  if (op_->Type() == runtime_graph::OpType::kMegaAttention) {
    Bool use_kv_cache = false;
    if (HasAttr(op_->Attrs(), "use_kv_cache")) {
      const Status attr_status =
          GetBoolAttr(op_->Attrs(), "use_kv_cache", &use_kv_cache);
      if (!attr_status.ok()) {
        return PrefixStatus(attr_status, "op=" + op_->Name());
      }
    }
    attention.batch = step_context_->batch;
    attention.token_count = step_context_->packed_tokens;
    attention.kv_block_size =
        step_context_->kv_view != nullptr
            ? step_context_->kv_view->block_size
            : 0;
    attention.kv = step_context_->kv_view;
    if (use_kv_cache) {
      if (step_context_->kv_view == nullptr) {
        return MakeStatus(
            ErrorCode::kStateError,
            "op=" + op_->Name() +
                ": use_kv_cache=true requires KvCacheView");
      }
      I64 layer_index = -1;
      const Status layer_status =
          GetIntAttr(op_->Attrs(), "kv_layer_index", &layer_index);
      if (!layer_status.ok()) {
        return PrefixStatus(layer_status, "op=" + op_->Name());
      }
      if (layer_index < 0 ||
          static_cast<Usize>(layer_index) >=
              step_context_->kv_view->layers.size()) {
        return MakeStatus(
            ErrorCode::kInvalidArgument,
            "op=" + op_->Name() +
                ": kv_layer_index out of range");
      }
      attention.layer_index = static_cast<I32>(layer_index);
      attention.meta = step_context_->attention_meta;
    }
    context.attention = &attention;
  }

  // Sample 始终只处理 output_slot 行；Linear 在 attr sample_rows=true 时
  // 同样只计算这些行（LM Head 优化）。
  Bool use_sample_rows =
      op_->Type() == runtime_graph::OpType::kSample;
  if (op_->Type() == runtime_graph::OpType::kLinear &&
      HasAttr(op_->Attrs(), "sample_rows")) {
    Bool sample_rows = false;
    const Status attr_status =
        GetBoolAttr(op_->Attrs(), "sample_rows", &sample_rows);
    if (!attr_status.ok()) {
      return PrefixStatus(attr_status, "op=" + op_->Name());
    }
    use_sample_rows = sample_rows;
  }
  if (use_sample_rows) {
    context.sample_rows_enabled = step_context_->sample_rows_enabled;
    context.sample_rows = step_context_->sample_rows;
    context.sample_row_count = step_context_->sample_row_count;
    context.sample_rows_view = step_context_->sample_rows_view;
  }

  if (spec_->need_workspace) {
    if (graph_env_ == nullptr) {
      return MakeStatus(ErrorCode::kStateError,
                        "op=" + op_->Name() + ": GraphEnv is null");
    }
    const runtime_graph::Storage* workspace =
        graph_env_->GetSharedStorage(std::string(kWorkspacePrefix) +
                                     op_->Name());
    if (workspace == nullptr || workspace->data_ptr == nullptr) {
      return MakeStatus(
          ErrorCode::kStateError,
          "op=" + op_->Name() + ": workspace is not bound");
    }
    workspace_.data = workspace->data_ptr;
    workspace_.offset = workspace->offset;
    workspace_.bytes = workspace->byte_size;
    workspace_.dtype = DataTypeFromPrecision(workspace->precision);
    workspace_.layout = workspace->layout;
    workspace_.device = workspace->device_type;
    workspace_.shape = workspace->shape;
    context.workspace = workspace_;
  }

  const Status status = kernel_->Execute(context);
  if (!status.ok()) {
    return PrefixStatus(status, "op=" + op_->Name());
  }
  return Status::Ok();
}

// 返回 spec 是否声明 workspace。
Bool KernelOpEnv::NeedTempData() const {
  return spec_ != nullptr && spec_->need_workspace;
}

// 由 Storage 构造 Kernel 侧 TensorView；校验绑定/精度/布局。
Status KernelOpEnv::BuildView(const runtime_graph::Tensor& tensor,
                              const C8* role,
                              runtime_graph::TensorView* view) const {
  if (view == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "KernelOpEnv::BuildView: view must not be null");
  }
  const runtime_graph::Storage& storage = tensor.GetStorage();
  if (storage.data_ptr == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op_->Name() + " " + role + ": storage is not bound");
  }
  const DataType dtype = DataTypeFromPrecision(storage.precision);
  if (dtype == DataType::kUnknown) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        "op=" + op_->Name() + " " + role + ": unsupported precision");
  }
  view->data = storage.data_ptr;
  view->offset = storage.offset;
  view->bytes = storage.byte_size;
  view->shape = storage.shape;
  view->dtype = dtype;
  view->layout = storage.layout;
  view->device = storage.device_type;
  return Status::Ok();
}

}  // namespace mobilenet
