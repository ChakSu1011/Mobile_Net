// 模块：RuntimeGraph 执行实现
//
// 职责：
//   实现 Backend/Arena/Kernel 绑定、V17 执行前校验、Forward（H2D/计算/D2H）
//   与 Release（逆拓扑序销毁 Kernel）。
//
// 边界：
//   - 不拥有 Backend、Kernel、Arena、I/O 缓冲区；
//   - Forward 稳态不分配堆内存（KernelContext 使用栈上数组）；
//   - 不实现任何具体算子 Kernel。

#include <cstdint>
#include <string>
#include <utility>

#include "graph/runtime_graph.h"
#include "graph/runtime_validator.h"

namespace mobilenet {
namespace graph {
namespace {

// 添加问题。
void AddIssue(ValidationReport* report, ErrorCode code, std::string message,
              std::string node_name = "", std::string edge_name = "",
              std::string binding_name = "") {
  report->AddIssue(ValidationIssue{code, std::move(message),
                                   std::move(node_name), std::move(edge_name),
                                   std::move(binding_name), ""});
}

// 计算设备地址：arena 基址 + edge.memory_offset + tensor.offset。
Status ResolveEdgePtr(const std::map<int32_t, ArenaBinding>& arenas,
                      const RuntimeEdge& edge, void** pointer) {
  if (pointer == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "edge pointer output must not be null");
  }
  if (edge.arena_id < 0) {
    return Status::Error(ErrorCode::kStateError,
                         "edge has no arena: " + edge.name);
  }
  const auto iterator = arenas.find(edge.arena_id);
  if (iterator == arenas.end()) {
    return Status::Error(ErrorCode::kStateError,
                         "edge arena is not bound: " + edge.name);
  }
  const ArenaBinding& arena = iterator->second;
  if (arena.base == nullptr || arena.bytes == 0) {
    return Status::Error(ErrorCode::kBufferError,
                         "edge arena is empty: " + edge.name);
  }
  if (edge.memory_offset > arena.bytes ||
      edge.tensor.offset > arena.bytes - edge.memory_offset ||
      edge.tensor.size_bytes >
          arena.bytes - edge.memory_offset - edge.tensor.offset) {
    return Status::Error(ErrorCode::kBufferError,
                         "edge memory range exceeds arena: " + edge.name);
  }
  *pointer =
      static_cast<char*>(arena.base) + edge.memory_offset + edge.tensor.offset;
  return Status::Ok();
}

// 检查 IoBinding 与 host TensorDesc 一致。
Status CheckHostBinding(const IoBinding& binding, const TensorDesc& host_tensor,
                        std::string_view binding_name) {
  if (binding.host_ptr == nullptr) {
    return Status::Error(
        ErrorCode::kInvalidArgument,
        "I/O binding host_ptr is null: " + std::string(binding_name));
  }
  if (binding.bytes < host_tensor.size_bytes) {
    return Status::Error(
        ErrorCode::kBufferError,
        "I/O binding is too small: " + std::string(binding_name));
  }
  if (binding.dtype != host_tensor.dtype) {
    return Status::Error(
        ErrorCode::kUnsupportedDtype,
        "I/O binding dtype mismatch: " + std::string(binding_name));
  }
  if (binding.shape.size() != host_tensor.shape.size()) {
    return Status::Error(
        ErrorCode::kInvalidArgument,
        "I/O binding rank mismatch: " + std::string(binding_name));
  }
  for (size_t index = 0; index < binding.shape.size(); ++index) {
    if (binding.shape[index].value != host_tensor.shape[index].value) {
      return Status::Error(
          ErrorCode::kInvalidArgument,
          "I/O binding shape mismatch: " + std::string(binding_name));
    }
  }
  return Status::Ok();
}

}  // namespace

Status RuntimeGraph::BindBackend(Backend* backend, ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (backend == nullptr) {
    if (report != nullptr) {
      AddIssue(report, ErrorCode::kInvalidArgument, "backend must not be null");
    }
    return Status::Error(ErrorCode::kInvalidArgument,
                         "backend must not be null");
  }
  if (!finalized_) {
    if (report != nullptr) {
      AddIssue(report, ErrorCode::kStateError, "graph is not finalized");
    }
    return Status::Error(ErrorCode::kStateError, "graph is not finalized");
  }
  if (released_) {
    if (report != nullptr) {
      AddIssue(report, ErrorCode::kStateError, "graph has been released");
    }
    return Status::Error(ErrorCode::kStateError, "graph has been released");
  }
  if (backend_ != nullptr) {
    if (report != nullptr) {
      AddIssue(report, ErrorCode::kStateError, "backend is already bound");
    }
    return Status::Error(ErrorCode::kStateError, "backend is already bound");
  }
  backend_ = backend;
  return Status::Ok();
}

Status RuntimeGraph::BindArena(int32_t arena_id, void* base, uint64_t bytes,
                               ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (!finalized_ || released_) {
    return Status::Error(ErrorCode::kStateError,
                         "graph is not in a bindable state");
  }
  if (arena_id < 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "arena_id must be non-negative");
  }
  if (base == nullptr || bytes == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "arena base must be non-null and bytes positive");
  }
  if (arenas_.find(arena_id) != arenas_.end()) {
    return Status::Error(ErrorCode::kStateError,
                         "arena is already bound: " + std::to_string(arena_id));
  }
  arenas_.emplace(arena_id, ArenaBinding{base, bytes});
  return Status::Ok();
}

Status RuntimeGraph::BindArenas(const std::map<int32_t, ArenaBinding>& arenas,
                                ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (!finalized_ || released_) {
    return Status::Error(ErrorCode::kStateError,
                         "graph is not in a bindable state");
  }
  for (const auto& arena : arenas) {
    if (arena.first < 0 || arena.second.base == nullptr ||
        arena.second.bytes == 0) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "invalid arena binding");
    }
    if (arenas_.find(arena.first) != arenas_.end()) {
      return Status::Error(
          ErrorCode::kStateError,
          "arena is already bound: " + std::to_string(arena.first));
    }
  }
  arenas_.insert(arenas.begin(), arenas.end());
  return Status::Ok();
}

Status RuntimeGraph::BindKernel(NodeId node_id, Kernel* kernel,
                                ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (!finalized_ || released_) {
    return Status::Error(ErrorCode::kStateError,
                         "graph is not in a bindable state");
  }
  if (node_id >= nodes_.size()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "node id is out of range");
  }
  const RuntimeNode& node = nodes_[node_id];
  if (!IsComputeOp(node.op_type)) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "cannot bind kernel to I/O node: " + node.name);
  }
  if (kernel == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "kernel must not be null: " + node.name);
  }
  if (node_kernels_.size() != nodes_.size()) {
    return Status::Error(ErrorCode::kInternal,
                         "kernel table size does not match node count");
  }
  if (node_kernels_[node_id] != nullptr) {
    return Status::Error(ErrorCode::kStateError,
                         "kernel is already bound: " + node.name);
  }
  node_kernels_[node_id] = kernel;
  return Status::Ok();
}

Status RuntimeGraph::ValidateExecution(ForwardContext* context,
                                       ValidationReport* report) const {
  return RuntimeGraphValidator::ValidateExecution(*this, context, report);
}

Status RuntimeGraph::BuildKernelContext(NodeId node_id, ForwardContext* context,
                                        KernelIo* io,
                                        KernelContext* kernel_context) const {
  if (io == nullptr || kernel_context == nullptr || context == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "kernel context arguments must not be null");
  }
  if (node_id >= nodes_.size()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "node id is out of range");
  }
  const RuntimeNode& node = nodes_[node_id];
  if (node.input_edges.size() > kMaxNodePorts ||
      node.output_edges.size() > kMaxNodePorts) {
    return Status::Error(ErrorCode::kInternal,
                         "node has too many ports: " + node.name);
  }
  for (size_t index = 0; index < node.input_edges.size(); ++index) {
    const EdgeId edge_id = node.input_edges[index];
    if (edge_id >= edges_.size()) {
      return Status::Error(ErrorCode::kInternal,
                           "node input edge is out of range: " + node.name);
    }
    const RuntimeEdge& edge = edges_[edge_id];
    void* pointer = nullptr;
    const Status status = ResolveEdgePtr(arenas_, edge, &pointer);
    if (!status.ok()) {
      return status;
    }
    io->input_tensors[index] = &edge.tensor;
    io->input_ptrs[index] = pointer;
  }
  for (size_t index = 0; index < node.output_edges.size(); ++index) {
    const EdgeId edge_id = node.output_edges[index];
    if (edge_id >= edges_.size()) {
      return Status::Error(ErrorCode::kInternal,
                           "node output edge is out of range: " + node.name);
    }
    const RuntimeEdge& edge = edges_[edge_id];
    void* pointer = nullptr;
    const Status status = ResolveEdgePtr(arenas_, edge, &pointer);
    if (!status.ok()) {
      return status;
    }
    io->output_tensors[index] = &edge.tensor;
    io->output_ptrs[index] = pointer;
  }
  kernel_context->node = &node;
  kernel_context->input_tensors = io->input_tensors.data();
  kernel_context->output_tensors = io->output_tensors.data();
  kernel_context->num_inputs = static_cast<int32_t>(node.input_edges.size());
  kernel_context->num_outputs = static_cast<int32_t>(node.output_edges.size());
  kernel_context->input_ptrs = io->input_ptrs.data();
  kernel_context->output_ptrs = io->output_ptrs.data();
  kernel_context->attributes = &node.attributes;
  kernel_context->kv_cache = &context->kv_cache;
  kernel_context->attention = &context->attention;
  kernel_context->sampling = &context->sampling;
  kernel_context->workspace = context->workspace;
  kernel_context->workspace_bytes = context->workspace_bytes;
  kernel_context->stream = context->stream;
  return Status::Ok();
}

Status RuntimeGraph::Forward(ForwardContext* context,
                             ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (!finalized_ || released_) {
    return Status::Error(ErrorCode::kStateError,
                         "graph is not in a forwardable state");
  }
  if (backend_ == nullptr) {
    return Status::Error(ErrorCode::kStateError, "backend is not bound");
  }
  if (context == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "forward context must not be null");
  }
  if (context->backend != nullptr && context->backend != backend_) {
    return Status::Error(ErrorCode::kStateError,
                         "forward context backend mismatch");
  }

  for (NodeId node_id : topo_order_) {
    const RuntimeNode& node = nodes_[node_id];
    if (node.op_type == OpType::kInput) {
      if (!node.host_tensor.has_value()) {
        return Status::Error(ErrorCode::kInternal,
                             "input node has no host tensor: " + node.name);
      }
      const auto binding_iterator =
          context->host_inputs.find(node.binding_name);
      if (binding_iterator == context->host_inputs.end()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "input binding is missing: " + node.binding_name);
      }
      const Status binding_status = CheckHostBinding(
          binding_iterator->second, *node.host_tensor, node.binding_name);
      if (!binding_status.ok()) {
        return binding_status;
      }
      for (EdgeId edge_id : node.output_edges) {
        if (edge_id >= edges_.size()) {
          return Status::Error(ErrorCode::kInternal,
                               "input node edge is out of range");
        }
        const RuntimeEdge& edge = edges_[edge_id];
        void* device_pointer = nullptr;
        Status status = ResolveEdgePtr(arenas_, edge, &device_pointer);
        if (!status.ok()) {
          return status;
        }
        status =
            backend_->Upload(device_pointer, binding_iterator->second.host_ptr,
                             node.host_tensor->size_bytes, context->stream);
        if (!status.ok()) {
          return PrefixStatus(status, "Upload " + node.name);
        }
      }
      continue;
    }
    if (node.op_type == OpType::kOutput) {
      if (!node.host_tensor.has_value()) {
        return Status::Error(ErrorCode::kInternal,
                             "output node has no host tensor: " + node.name);
      }
      const auto binding_iterator =
          context->host_outputs.find(node.binding_name);
      if (binding_iterator == context->host_outputs.end()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "output binding is missing: " + node.binding_name);
      }
      const Status binding_status = CheckHostBinding(
          binding_iterator->second, *node.host_tensor, node.binding_name);
      if (!binding_status.ok()) {
        return binding_status;
      }
      if (node.input_edges.size() != 1) {
        return Status::Error(ErrorCode::kInternal,
                             "output node must have one input edge");
      }
      const EdgeId edge_id = node.input_edges.front();
      if (edge_id >= edges_.size()) {
        return Status::Error(ErrorCode::kInternal,
                             "output node edge is out of range");
      }
      const RuntimeEdge& edge = edges_[edge_id];
      void* device_pointer = nullptr;
      Status status = ResolveEdgePtr(arenas_, edge, &device_pointer);
      if (!status.ok()) {
        return status;
      }
      status =
          backend_->Download(binding_iterator->second.host_ptr, device_pointer,
                             node.host_tensor->size_bytes, context->stream);
      if (!status.ok()) {
        return PrefixStatus(status, "Download " + node.name);
      }
      continue;
    }

    if (node_id >= node_kernels_.size() || node_kernels_[node_id] == nullptr) {
      return Status::Error(ErrorCode::kStateError,
                           "compute node has no kernel: " + node.name);
    }
    KernelIo io;
    KernelContext kernel_context;
    const Status context_status =
        BuildKernelContext(node_id, context, &io, &kernel_context);
    if (!context_status.ok()) {
      return PrefixStatus(context_status, "KernelContext " + node.name);
    }
    const Status execute_status =
        node_kernels_[node_id]->Execute(kernel_context);
    if (!execute_status.ok()) {
      return PrefixStatus(execute_status, "Execute " + node.name);
    }
  }

  const Status sync_status = backend_->Synchronize(context->stream);
  if (!sync_status.ok()) {
    return PrefixStatus(sync_status, "Synchronize");
  }
  return Status::Ok();
}

Status RuntimeGraph::Release(ValidationReport* report) {
  if (report != nullptr) {
    report->Clear();
  }
  if (!finalized_) {
    return Status::Error(ErrorCode::kStateError, "graph is not finalized");
  }
  if (released_) {
    return Status::Ok();
  }
  Status first_error = Status::Ok();
  if (backend_ != nullptr) {
    for (auto iterator = topo_order_.rbegin(); iterator != topo_order_.rend();
         ++iterator) {
      const NodeId node_id = *iterator;
      if (node_id >= node_kernels_.size() ||
          node_kernels_[node_id] == nullptr) {
        continue;
      }
      Kernel* kernel = node_kernels_[node_id];
      node_kernels_[node_id] = nullptr;
      const Status status = backend_->DestroyKernel(kernel);
      if (!status.ok() && first_error.ok()) {
        first_error =
            PrefixStatus(status, "DestroyKernel " + nodes_[node_id].name);
      }
    }
  } else {
    for (Kernel*& kernel : node_kernels_) {
      if (kernel != nullptr) {
        if (first_error.ok()) {
          first_error = Status::Error(ErrorCode::kStateError,
                                      "cannot destroy kernel without backend");
        }
        kernel = nullptr;
      }
    }
  }
  arenas_.clear();
  backend_ = nullptr;
  released_ = true;
  if (!first_error.ok() && report != nullptr) {
    AddIssue(report, first_error.code(), first_error.message());
  }
  return first_error;
}

}  // namespace graph
}  // namespace mobilenet
