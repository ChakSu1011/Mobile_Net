// 模块：Runtime 默认 Loader 实现
//
// 职责：
//   实现 ILoader::Load：读取并校验 model.mgf 固定头与 manifest，重建 Graph
//   并校验 IR 哈希，按 BackendRegistry 初始化后端与能力校验，流式上传
//   weights.bin，分配 activation arena 与 I/O 缓冲，创建 GraphEnv/OpEnv，
//   最后把 Graph/Backend/ModelConfig 交给 Engine。
//
// 边界：
//   - 不调度请求、不管理 KV、不直接调用设备 API；
//   - 不写模型包、不缓存跨请求状态；
//   - 失败时 result 不修改；已创建资源全部登记到 ResourceStack 逆序释放；
//   - 除 OpEnvFactory/BackendRegistry 接口外不依赖具体后端实现。
//
// 依赖：
//   Runtime/loader/{loader,manifest,ir_dump}.h、
//   Runtime/common/{file_util,op_env_factory,resource_stack,sha256}.h、
//   Runtime/backend/{backend,backend_registry,kernel_types}.h、
//   Graph/include/*.h、C++17 标准库。

#include "Runtime/loader/loader.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "Graph/include/graph.h"
#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Graph/include/op_env.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
#include "Runtime/backend/kernel_types.h"
#include "Runtime/common/file_util.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/op_env_factory.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/runtime_config.h"
#include "Runtime/common/sha256.h"
#include "Runtime/common/status.h"
#include "Runtime/loader/ir_dump.h"
#include "Runtime/loader/manifest.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// 读取权重文件时的分块大小（8 MB），限制单次分配峰值。
constexpr U64 kWeightReadChunkBytes = 8ULL * 1024ULL * 1024ULL;

// metadata-only 模式下常量占位缓冲字节数；仅用于让 OpEnv 初始化通过，
// 绝对不允许执行推理。
constexpr U64 kMetadataOnlyDummyBytes = 64;

// 构造模型包错误。
//
// 参数：
//   detail：错误原因。
// 返回：
//   kInvalidModelFile 状态。
Status PackageError(const std::string& detail) {
  return MakeStatus(ErrorCode::kInvalidModelFile,
                    "model package: " + detail);
}

// 拼接目录与文件名。
//
// 参数：
//   directory：包目录；
//   name：文件名，不含分隔符。
// 返回：
//   使用 '/' 连接的路径。
std::string JoinPath(const std::string& directory,
                     const std::string& name) {
  if (!directory.empty() &&
      (directory.back() == '/' || directory.back() == '\\')) {
    return directory + name;
  }
  return directory + "/" + name;
}

// 判断是否为 2 的幂；0 返回 false。
//
// 参数：
//   value：待判断值。
// 返回：
//   true 表示 value 是 2 的幂。
Bool IsPowerOfTwo(U64 value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// 解析实际使用的缓冲对齐字节数。
//
// 功能：
//   取 manifest 计划对齐与后端最小对齐的较大值；两者都必须是 2 的幂。
//
// 参数：
//   plan_alignment：memory_plan.alignment；
//   min_alignment：BackendCapability.min_alignment，0 表示未声明；
//   alignment：输出参数，非空。
// 返回：
//   成功返回 kOk；非法对齐返回 kInvalidModelFile/kUnsupportedBackend。
Status ResolveAlignment(U64 plan_alignment, U64 min_alignment,
                        U64* alignment) {
  if (!IsPowerOfTwo(plan_alignment)) {
    return PackageError("memory_plan.alignment must be a power of two");
  }
  if (min_alignment == 0) {
    *alignment = plan_alignment;
    return Status::Ok();
  }
  if (!IsPowerOfTwo(min_alignment)) {
    return MakeStatus(
        ErrorCode::kUnsupportedBackend,
        "backend min_alignment is not a power of two");
  }
  *alignment = std::max(plan_alignment, min_alignment);
  return Status::Ok();
}

// 检查单缓冲大小是否超过后端能力。
//
// 参数：
//   bytes：缓冲字节数；
//   capability：后端能力；
//   label：错误信息中的缓冲名。
// 返回：
//   成功返回 kOk；超限返回 kBufferError。
Status CheckBufferLimit(U64 bytes, const BackendCapability& capability,
                        const std::string& label) {
  if (capability.max_buffer_bytes != 0 &&
      bytes > capability.max_buffer_bytes) {
    return MakeStatus(ErrorCode::kBufferError,
                      "buffer exceeds backend max_buffer_bytes: " + label);
  }
  return Status::Ok();
}

// 后端及其所有权状态。
//
// 说明：
//   加载阶段由 Loader 持有 owned；成功后移交给 LoadResult，raw 继续指向
//   Executor 持有的后端，供 ResourceStack 的缓冲释放闭包使用。
struct BackendState {
  // 加载期拥有的后端；成功后移出。
  std::unique_ptr<Backend> owned;

  // 非拥有裸指针；owned 为空时仍由 Executor 保证有效。
  Backend* raw = nullptr;
};

// 缓冲持有者；Storage.data_ptr 保存其 buffer.get() 的地址。
struct BufferHolder {
  // 设备缓冲句柄。
  std::unique_ptr<Buffer> buffer;
};

// 创建缓冲并登记到资源栈。
//
// 参数：
//   state：后端状态，非空；
//   desc：缓冲描述；
//   label：资源标签，用于日志与错误信息；
//   stack：资源栈，非空；
//   holder：输出参数，非空；成功时写入共享持有者。
// 返回：
//   成功返回 kOk；后端创建失败透传其错误码。
Status AllocateTrackedBuffer(BackendState* state, const BufferDesc& desc,
                             const std::string& label,
                             ResourceStack* stack,
                             std::shared_ptr<BufferHolder>* holder) {
  if (state == nullptr || state->raw == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "AllocateTrackedBuffer: backend is not ready");
  }
  auto created = std::make_shared<BufferHolder>();
  Status status = state->raw->CreateBuffer(desc, &created->buffer);
  if (!status.ok()) {
    return PrefixStatus(status, "buffer " + label);
  }
  stack->Push("buffer/" + label, [state, created]() -> Status {
    if (created->buffer == nullptr || state->raw == nullptr) {
      return Status::Ok();
    }
    const Status release =
        state->raw->DestroyBuffer(std::move(created->buffer));
    created->buffer.reset();
    return release;
  });
  *holder = std::move(created);
  return Status::Ok();
}

// 把 Storage 绑定到设备 Buffer。
//
// 参数：
//   tensor：目标张量，非空；
//   buffer：Buffer 对象地址，非空；
//   offset：缓冲内字节偏移；
//   bytes：有效字节数。
void BindDeviceStorage(runtime_graph::Tensor* tensor, Buffer* buffer,
                       U64 offset, U64 bytes) {
  runtime_graph::Storage& storage = tensor->GetStorage();
  storage.data_ptr = buffer;
  storage.offset = static_cast<Usize>(offset);
  storage.byte_size = static_cast<Usize>(bytes);
  storage.device_type = runtime_graph::DeviceType::kDevice;
}

// 把 GraphEnv 共享 Storage 绑定到设备 Buffer。
//
// 参数：
//   storage：共享 storage，非空；
//   buffer：Buffer 对象地址，非空。
void BindSharedStorage(runtime_graph::Storage* storage, Buffer* buffer) {
  storage->data_ptr = buffer;
  storage->offset = 0;
  storage->device_type = runtime_graph::DeviceType::kDevice;
}

// 校验 manifest 张量字节数与 shape/精度一致。
//
// 参数：
//   meta：张量元数据；
// 返回：
//   成功返回 kOk；精度不可映射、维度非法或字节数不一致返回
//   kInvalidModelFile。
Status ValidateTensorBytes(const TensorMeta& meta) {
  const DataType dtype = DataTypeFromPrecision(meta.precision);
  const U32 element_size = DataTypeSize(dtype);
  if (element_size == 0) {
    return PackageError("tensor has unsupported precision: " + meta.name);
  }
  U64 elements = 1;
  for (const I64 dim : meta.dims) {
    if (dim < 0) {
      return PackageError("tensor has negative dimension: " + meta.name);
    }
    const U64 value = static_cast<U64>(dim);
    if (value != 0 &&
        elements > std::numeric_limits<U64>::max() / value) {
      return PackageError("tensor element count overflows: " + meta.name);
    }
    elements *= value;
  }
  const U64 expected = elements * static_cast<U64>(element_size);
  if (meta.byte_size != expected) {
    return PackageError("tensor byte_size does not match shape: " +
                        meta.name);
  }
  return Status::Ok();
}

// 用 manifest 元数据填充张量的 Storage 描述（不含 data_ptr）。
//
// 参数：
//   tensor：目标张量，非空；
//   meta：张量元数据。
void FillStorageMetadata(runtime_graph::Tensor* tensor,
                         const TensorMeta& meta) {
  runtime_graph::Storage& storage = tensor->GetStorage();
  storage.shape.max_dim = static_cast<I64>(meta.dims.size());
  for (Usize index = 0; index < meta.dims.size(); ++index) {
    storage.shape.dims[index] = meta.dims[index];
    storage.shape.pitch[index] =
        meta.pitch.empty() ? meta.dims[index] : meta.pitch[index];
  }
  storage.precision = meta.precision;
  storage.layout = meta.layout;
  storage.byte_size = static_cast<Usize>(meta.byte_size);
  storage.device_type = runtime_graph::DeviceType::kUnknown;
}

// 按 manifest 顺序重建 Graph。
//
// 功能：
//   常量 → 输入 → 计算 op（sample 走 CreateSample）→ 输出 → SortOp；
//   每一步校验输出名称与 manifest 完全一致。
//
// 参数：
//   manifest：已校验 manifest；
//   graph：输出图，非空。
// 返回：
//   成功返回 kOk；Graph 接口失败或命名不一致返回 kInvalidModelFile。
Status RebuildGraph(const ModelManifest& manifest,
                    runtime_graph::Graph* graph) {
  if (graph == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "RebuildGraph: graph must not be null");
  }
  std::map<std::string, runtime_graph::Tensor*> tensors;

  // 常量：与 Python create_constant 的 Storage 构造保持一致。
  for (const std::string& name : manifest.graph.constants) {
    const auto meta_it = manifest.graph.tensors.find(name);
    if (meta_it == manifest.graph.tensors.end()) {
      return PackageError("constant tensor is missing: " + name);
    }
    const TensorMeta& meta = meta_it->second;
    if (meta.dims.size() > static_cast<Usize>(runtime_graph::kMaxShapeDim)) {
      return PackageError("constant rank exceeds kMaxShapeDim: " + name);
    }
    runtime_graph::Storage storage;
    storage.shape.max_dim = static_cast<I64>(meta.dims.size());
    for (Usize index = 0; index < meta.dims.size(); ++index) {
      storage.shape.dims[index] = meta.dims[index];
      storage.shape.pitch[index] =
          meta.pitch.empty() ? meta.dims[index] : meta.pitch[index];
    }
    storage.precision = meta.precision;
    storage.layout = meta.layout;
    storage.device_type = runtime_graph::DeviceType::kHost;
    storage.byte_size = static_cast<Usize>(meta.byte_size);
    storage.data_ptr = nullptr;
    storage.reusable = true;
    runtime_graph::Tensor* tensor = nullptr;
    const Status status = graph->CreateConstant(name, storage, &tensor);
    if (!status.ok()) {
      return PrefixStatus(PackageError("create constant failed: " + name),
                          status.message());
    }
    tensors.emplace(name, tensor);
  }

  // 输入。
  for (const std::string& name : manifest.graph.inputs) {
    const auto meta_it = manifest.graph.tensors.find(name);
    if (meta_it == manifest.graph.tensors.end()) {
      return PackageError("input tensor is missing: " + name);
    }
    runtime_graph::Tensor* tensor = nullptr;
    const Status status =
        graph->CreateInput(name, meta_it->second.precision,
                           meta_it->second.layout, &tensor);
    if (!status.ok()) {
      return PrefixStatus(PackageError("create input failed: " + name),
                          status.message());
    }
    tensors.emplace(name, tensor);
  }

  // 计算 op。
  for (const OpMeta& op : manifest.graph.ops) {
    std::vector<runtime_graph::Tensor*> sources;
    sources.reserve(op.inputs.size());
    for (const std::string& input : op.inputs) {
      const auto it = tensors.find(input);
      if (it == tensors.end()) {
        return PackageError("op " + op.name +
                            " references unknown tensor: " + input);
      }
      sources.push_back(it->second);
    }
    runtime_graph::OpType op_type = runtime_graph::OpType::kUnknown;
    Status status = OpTypeFromName(op.op_type, &op_type);
    if (!status.ok()) {
      return status;
    }
    std::vector<runtime_graph::Tensor*> outputs;
    if (op_type == runtime_graph::OpType::kSample) {
      if (sources.size() != 1) {
        return PackageError("sample op must have exactly one input: " +
                            op.name);
      }
      runtime_graph::Tensor* output = nullptr;
      status = graph->CreateSample(op.name, sources[0], op.attrs, &output);
      if (!status.ok()) {
        return PrefixStatus(PackageError("create sample failed: " + op.name),
                            status.message());
      }
      outputs.push_back(output);
    } else {
      status = graph->CreateOp(op.name, sources, op.outputs.size(), op_type,
                               op.attrs, &outputs);
      if (!status.ok()) {
        return PrefixStatus(PackageError("create op failed: " + op.name),
                            status.message());
      }
    }
    if (outputs.size() != op.outputs.size()) {
      return PackageError("op output count differs from manifest: " +
                          op.name);
    }
    for (Usize index = 0; index < outputs.size(); ++index) {
      if (outputs[index]->Name() != op.outputs[index]) {
        return PackageError("op output name differs from manifest: " +
                            op.name + " expected " + op.outputs[index] +
                            " got " + outputs[index]->Name());
      }
      tensors.emplace(op.outputs[index], outputs[index]);
    }
  }

  // 输出绑定。
  for (const IOBinding& binding : manifest.graph.outputs) {
    const auto it = tensors.find(binding.tensor);
    if (it == tensors.end()) {
      return PackageError("output references unknown tensor: " +
                          binding.tensor);
    }
    const Status status = graph->CreateOutput(binding.name, it->second);
    if (!status.ok()) {
      return PrefixStatus(PackageError("create output failed: " +
                                       binding.name),
                          status.message());
    }
  }

  const Status status = graph->SortOp();
  if (!status.ok()) {
    return PrefixStatus(PackageError("sort op failed"), status.message());
  }
  return Status::Ok();
}

// 从 Graph 公共查询接口建立 name → Tensor* 映射。
//
// 参数：
//   graph：已排序的图，非空；
//   index：输出参数，非空。
// 返回：
//   成功返回 kOk；同名 tensor 指针冲突返回 kInternal。
Status BuildTensorIndex(runtime_graph::Graph* graph,
                        std::map<std::string, runtime_graph::Tensor*>* index) {
  index->clear();
  const auto add = [index](runtime_graph::Tensor* tensor) -> Status {
    if (tensor == nullptr) {
      return MakeStatus(ErrorCode::kInternal,
                        "graph contains null tensor");
    }
    const auto inserted = index->emplace(tensor->Name(), tensor);
    if (!inserted.second && inserted.first->second != tensor) {
      return MakeStatus(ErrorCode::kInternal,
                        "graph has duplicate tensor name: " +
                            tensor->Name());
    }
    return Status::Ok();
  };
  for (runtime_graph::Tensor* tensor : graph->GetAllConstantTensor()) {
    Status status = add(tensor);
    if (!status.ok()) {
      return status;
    }
  }
  for (runtime_graph::Op* op : graph->GetAllInputOp()) {
    for (runtime_graph::Tensor* tensor : op->DstTensors()) {
      Status status = add(tensor);
      if (!status.ok()) {
        return status;
      }
    }
  }
  for (runtime_graph::Op* op : graph->GetOrderlyOPList()) {
    for (runtime_graph::Tensor* tensor : op->DstTensors()) {
      Status status = add(tensor);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

// 校验 manifest 张量与重建图完全对应。
//
// 参数：
//   manifest：manifest；
//   graph：重建图；
//   index：name → Tensor* 映射。
// 返回：
//   成功返回 kOk；缺项或多项返回 kInvalidModelFile。
Status ValidateGraphCoverage(
    const ModelManifest& manifest, runtime_graph::Graph* graph,
    const std::map<std::string, runtime_graph::Tensor*>& index) {
  const std::vector<runtime_graph::Tensor*>& constants =
      graph->GetAllConstantTensor();
  if (constants.size() != manifest.graph.constants.size()) {
    return PackageError("rebuilt constant count differs from manifest");
  }
  for (Usize i = 0; i < constants.size(); ++i) {
    if (constants[i]->Name() != manifest.graph.constants[i]) {
      return PackageError("rebuilt constant order differs from manifest");
    }
  }
  const std::vector<runtime_graph::Op*>& inputs = graph->GetAllInputOp();
  if (inputs.size() != manifest.graph.inputs.size()) {
    return PackageError("rebuilt input count differs from manifest");
  }
  for (Usize i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->Name() != manifest.graph.inputs[i]) {
      return PackageError("rebuilt input order differs from manifest");
    }
  }
  for (const auto& entry : manifest.graph.tensors) {
    const TensorMeta& meta = entry.second;
    if (meta.kind == TensorKind::kOutput) {
      continue;
    }
    const auto it = index.find(meta.name);
    if (it == index.end()) {
      return PackageError("manifest tensor is missing in graph: " +
                          meta.name);
    }
    if (meta.kind == TensorKind::kConstant && !it->second->IsConstant()) {
      return PackageError("manifest constant is not constant in graph: " +
                          meta.name);
    }
    if (meta.kind == TensorKind::kActivation && it->second->IsConstant()) {
      return PackageError("manifest activation is constant in graph: " +
                          meta.name);
    }
  }
  for (const auto& entry : index) {
    if (manifest.graph.tensors.find(entry.first) ==
        manifest.graph.tensors.end()) {
      return PackageError("graph tensor is missing in manifest: " +
                          entry.first);
    }
  }
  return Status::Ok();
}

// 校验后端能力是否覆盖模型需求。
//
// 参数：
//   model：模型配置；
//   capability：后端能力；
//   graph：重建图。
// 返回：
//   成功返回 kOk；算子/精度/传输能力缺失返回 kUnsupportedOp/
//   kUnsupportedDtype/kUnsupportedBackend。
Status ValidateCapability(const ModelConfig& model,
                          const BackendCapability& capability,
                          runtime_graph::Graph* graph) {
  const auto supports_dtype = [&capability](DataType dtype) {
    return std::find(capability.supported_dtypes.begin(),
                     capability.supported_dtypes.end(),
                     dtype) != capability.supported_dtypes.end();
  };
  if (!supports_dtype(model.dtype)) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "backend does not support target dtype: " +
                          std::string(DataTypeName(model.dtype)));
  }
  if (!capability.h2d || !capability.d2h) {
    return MakeStatus(
        ErrorCode::kUnsupportedBackend,
        "backend must support both host-to-device and device-to-host");
  }
  const auto supports_op = [&capability](runtime_graph::OpType op_type) {
    return std::find(capability.supported_ops.begin(),
                     capability.supported_ops.end(),
                     op_type) != capability.supported_ops.end();
  };
  for (runtime_graph::Op* op : graph->GetOrderlyOPList()) {
    if (op->Type() == runtime_graph::OpType::kInput ||
        op->Type() == runtime_graph::OpType::kOutput) {
      continue;
    }
    if (!supports_op(op->Type())) {
      return MakeStatus(ErrorCode::kUnsupportedOp,
                        "backend does not support op " +
                            std::string(runtime_graph::OpTypeToStr(
                                op->Type())) +
                            ": " + op->Name());
    }
  }
  // 图内浮点精度必须可被后端计算支持；整数/布尔张量只用于索引与采样。
  const auto check_float_tensor =
      [&supports_dtype](runtime_graph::Tensor* tensor,
                        const C8* role) -> Status {
    const DataType dtype =
        DataTypeFromPrecision(tensor->GetStorage().precision);
    if (dtype == DataType::kUnknown) {
      return MakeStatus(ErrorCode::kUnsupportedDtype,
                        std::string(role) +
                            " has unsupported precision: " +
                            tensor->Name());
    }
    if ((dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
         dtype == DataType::kFloat32 || dtype == DataType::kFloat64) &&
        !supports_dtype(dtype)) {
      return MakeStatus(ErrorCode::kUnsupportedDtype,
                        std::string("backend does not support ") + role +
                            " precision: " + tensor->Name());
    }
    return Status::Ok();
  };
  for (runtime_graph::Tensor* tensor : graph->GetAllConstantTensor()) {
    const Status status = check_float_tensor(tensor, "constant");
    if (!status.ok()) {
      return status;
    }
  }
  for (runtime_graph::Op* op : graph->GetOrderlyOPList()) {
    for (runtime_graph::Tensor* tensor : op->DstTensors()) {
      const Status status = check_float_tensor(tensor, "activation");
      if (!status.ok()) {
        return status;
      }
    }
  }
  return Status::Ok();
}

// 计算权重条目转换后的目标字节数。
//
// 参数：
//   entry：权重条目；
//   target_dtype：目标精度；
//   bytes：输出参数，非空。
// 返回：
//   成功返回 kOk；dtype 未知或字节数不整除返回 kInvalidModelFile。
Status WeightTargetBytes(const WeightEntry& entry, DataType target_dtype,
                         U64* bytes) {
  DataType source_dtype = DataType::kUnknown;
  Status status = DataTypeFromName(entry.dtype, &source_dtype);
  if (!status.ok()) {
    return status;
  }
  const U32 source_size = DataTypeSize(source_dtype);
  const U32 target_size = DataTypeSize(target_dtype);
  if (source_size == 0 || target_size == 0 ||
      entry.byte_size % source_size != 0) {
    return PackageError("weight dtype/byte_size is invalid: " + entry.name);
  }
  const U64 elements = entry.byte_size / source_size;
  if (elements != 0 &&
      target_size > std::numeric_limits<U64>::max() / elements) {
    return PackageError("weight target bytes overflow: " + entry.name);
  }
  *bytes = elements * static_cast<U64>(target_size);
  return Status::Ok();
}

// 默认 Loader 实现。
class DefaultLoader final : public ILoader {
 public:
  // 加载模型包；见 ILoader::Load。
  Status Load(const LoadConfig& config, ResourceStack* resource_stack,
              LoadResult* result) override;

 private:
  // 执行完整加载流程。
  //
  // 参数：
  //   config：包目录与运行参数；
  //   stack：资源栈，非空；
  //   result：输出参数，非空；失败时不修改。
  // 返回：
  //   kOk 或具体错误码。
  Status LoadPackage(const LoadConfig& config, ResourceStack* stack,
                     LoadResult* result) const;
};

Status DefaultLoader::Load(const LoadConfig& config,
                           ResourceStack* resource_stack,
                           LoadResult* result) {
  if (resource_stack == nullptr || result == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Loader::Load: resource_stack/result must not be null");
  }
  if (config.package_dir.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Loader::Load: package_dir must not be empty");
  }
  return LoadPackage(config, resource_stack, result);
}

Status DefaultLoader::LoadPackage(const LoadConfig& config,
                                  ResourceStack* stack,
                                  LoadResult* result) const {
  // 1. 固定头与 manifest。
  const std::string model_file = JoinPath(config.package_dir, "model.mgf");
  ModelPackageHeader header;
  Status status = ReadModelPackageHeader(model_file, &header);
  if (!status.ok()) {
    return status;
  }
  std::vector<U8> manifest_bytes;
  status = ReadFileRange(model_file, header.manifest_offset,
                         header.manifest_length, &manifest_bytes);
  if (!status.ok()) {
    return status;
  }
  const std::string manifest_sha =
      Sha256::Hex(manifest_bytes.data(), manifest_bytes.size());
  if (manifest_sha != Sha256BytesToHex(header.manifest_sha256)) {
    return PackageError("manifest SHA-256 mismatch: expected " +
                        Sha256BytesToHex(header.manifest_sha256) +
                        ", got " + manifest_sha);
  }
  const std::string manifest_text(manifest_bytes.begin(),
                                  manifest_bytes.end());
  ModelManifest manifest;
  status = ParseManifest(manifest_text, &manifest);
  if (!status.ok()) {
    return status;
  }
  ModelConfig model;
  status = ParseModelConfig(manifest, &model);
  if (!status.ok()) {
    return status;
  }

  // 2. 权重文件存在性与大小校验。
  const Bool verify_weights = config.runtime.verify_weights_sha256;
  const Bool metadata_only = config.runtime.allow_metadata_only;
  const std::string weights_file =
      JoinPath(config.package_dir, manifest.weights.file);
  Bool has_weight_file = false;
  if (header.has_weights) {
    U64 weights_size = 0;
    status = GetFileSize(weights_file, &weights_size);
    if (status.ok()) {
      has_weight_file = true;
      if (weights_size != header.weights_length ||
          weights_size != manifest.weights.total_bytes) {
        return PackageError(
            "weights file size mismatch: header=" +
            std::to_string(header.weights_length) + ", manifest=" +
            std::to_string(manifest.weights.total_bytes) + ", file=" +
            std::to_string(weights_size));
      }
    } else if (!metadata_only) {
      return PackageError("weights file is required: " + weights_file);
    }
  } else if (manifest.weights.total_bytes != 0) {
    return PackageError(
        "manifest declares weights but model header has no weights flag");
  }
  if (!has_weight_file && !metadata_only) {
    return PackageError(
        "weights.bin is missing and metadata-only loading is disabled");
  }

  // 3. 重建图并校验 IR 哈希（必须在绑定任何 data_ptr 之前）。
  auto graph = std::make_unique<runtime_graph::Graph>(manifest.graph.name);
  status = RebuildGraph(manifest, graph.get());
  if (!status.ok()) {
    return status;
  }
  const std::string ir = DumpIr(*graph);
  const std::string ir_sha = Sha256::Hex(ir.data(), ir.size());
  if (ir_sha != manifest.graph.ir_sha256) {
    return PackageError(
        "rebuilt graph IR SHA-256 mismatch: expected " +
        manifest.graph.ir_sha256 + ", got " + ir_sha);
  }
  if (manifest.graph.ir_line_count > 0 &&
      static_cast<I64>(CountIrLines(ir)) != manifest.graph.ir_line_count) {
    return PackageError("rebuilt graph IR line count mismatch: expected " +
                        std::to_string(manifest.graph.ir_line_count) +
                        ", got " + std::to_string(CountIrLines(ir)));
  }

  // 4. 使用 Graph 公共查询接口建立张量索引并核对 manifest。
  std::map<std::string, runtime_graph::Tensor*> tensor_index;
  status = BuildTensorIndex(graph.get(), &tensor_index);
  if (!status.ok()) {
    return status;
  }
  status = ValidateGraphCoverage(manifest, graph.get(), tensor_index);
  if (!status.ok()) {
    return status;
  }

  // 4.1 预填输入/activation 的 Storage 元数据（不绑定 data_ptr）。
  //     IR 哈希已在上一阶段完成，此处修改 Storage 不影响哈希校验。
  for (const auto& entry : manifest.graph.tensors) {
    const TensorMeta& meta = entry.second;
    if (meta.kind != TensorKind::kInput &&
        meta.kind != TensorKind::kActivation) {
      continue;
    }
    const auto tensor_it = tensor_index.find(meta.name);
    if (tensor_it == tensor_index.end()) {
      return PackageError("manifest tensor is missing in graph: " +
                          meta.name);
    }
    status = ValidateTensorBytes(meta);
    if (!status.ok()) {
      return status;
    }
    FillStorageMetadata(tensor_it->second, meta);
  }

  // 5. 后端创建、Init 与能力校验。
  if (config.runtime.backend.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Loader::Load: runtime.backend must not be empty");
  }
  U64 alignment = 0;
  status = ResolveAlignment(manifest.memory_plan.alignment, 0, &alignment);
  if (!status.ok()) {
    return status;
  }
  BackendConfig backend_config;
  backend_config.device_id = config.runtime.device_id;
  backend_config.queue_count = config.runtime.queue_count;
  std::unique_ptr<Backend> backend;
  status = BackendRegistry::Instance().Create(
      config.runtime.backend, backend_config, &backend);
  if (!status.ok()) {
    return status;
  }
  if (backend == nullptr) {
    return MakeStatus(
        ErrorCode::kUnsupportedBackend,
        "backend factory returned null: " + config.runtime.backend);
  }
  status = backend->Init(backend_config);
  if (!status.ok()) {
    return status;
  }
  auto backend_state = std::make_shared<BackendState>();
  backend_state->owned = std::move(backend);
  backend_state->raw = backend_state->owned.get();
  // 后端释放闭包最先登记，LIFO 时最后执行，保证缓冲先于后端释放。
  stack->Push("loader/backend", [backend_state]() -> Status {
    if (backend_state->owned == nullptr) {
      return Status::Ok();
    }
    const Status deinit = backend_state->owned->Deinit();
    backend_state->owned.reset();
    return deinit;
  });

  BackendCapability capability;
  status = backend_state->raw->Capability(&capability);
  if (!status.ok()) {
    return status;
  }
  status = ResolveAlignment(manifest.memory_plan.alignment,
                            capability.min_alignment, &alignment);
  if (!status.ok()) {
    return status;
  }
  status = ValidateCapability(model, capability, graph.get());
  if (!status.ok()) {
    return status;
  }

  // 6. 预算与单缓冲上限预检查。
  U64 planned_weight_bytes = 0;
  for (const WeightEntry& entry : manifest.weights.entries) {
    if (!entry.available) {
      continue;
    }
    U64 target_bytes = 0;
    status = WeightTargetBytes(entry, model.dtype, &target_bytes);
    if (!status.ok()) {
      return status;
    }
    status = CheckBufferLimit(target_bytes, capability, entry.name);
    if (!status.ok()) {
      return status;
    }
    if (planned_weight_bytes >
        std::numeric_limits<U64>::max() - target_bytes) {
      return PackageError("weights bytes overflow");
    }
    planned_weight_bytes += target_bytes;
  }
  status = CheckBufferLimit(manifest.memory_plan.arena_bytes, capability,
                            "activation_arena");
  if (!status.ok()) {
    return status;
  }
  U64 planned_io_bytes = 0;
  for (const std::string& name : manifest.graph.inputs) {
    const auto it = manifest.graph.tensors.find(name);
    if (it == manifest.graph.tensors.end()) {
      return PackageError("input tensor is missing: " + name);
    }
    status = ValidateTensorBytes(it->second);
    if (!status.ok()) {
      return status;
    }
    status = CheckBufferLimit(it->second.byte_size, capability,
                              "io/" + name);
    if (!status.ok()) {
      return status;
    }
    if (planned_io_bytes > std::numeric_limits<U64>::max() -
                               it->second.byte_size) {
      return PackageError("io bytes overflow");
    }
    planned_io_bytes += it->second.byte_size;
  }
  if (config.runtime.memory_budget_bytes != 0) {
    U64 total = planned_weight_bytes;
    if (total > std::numeric_limits<U64>::max() -
                    manifest.memory_plan.arena_bytes) {
      return PackageError("device memory plan overflows");
    }
    total += manifest.memory_plan.arena_bytes;
    if (total > std::numeric_limits<U64>::max() - planned_io_bytes) {
      return PackageError("device memory plan overflows");
    }
    total += planned_io_bytes;
    if (total > config.runtime.memory_budget_bytes) {
      return MakeStatus(ErrorCode::kDeviceOom,
                        "memory budget does not cover weights/arena/io");
    }
  }

  // 7. 权重流式读取与上传。
  U64 weights_bytes = 0;
  if (has_weight_file) {
    FileReader reader;
    status = reader.Open(weights_file);
    if (!status.ok()) {
      return status;
    }
    if (verify_weights) {
      Sha256 file_sha;
      std::vector<U8> file_chunk;
      while (true) {
        Usize read_bytes = 0;
        status = reader.ReadNext(kWeightReadChunkBytes, &file_chunk,
                                 &read_bytes);
        if (!status.ok()) {
          return status;
        }
        if (read_bytes == 0) {
          break;
        }
        file_sha.Update(file_chunk.data(), read_bytes);
      }
      const std::string expected =
          Sha256BytesToHex(header.weights_sha256);
      const std::string actual = file_sha.HexDigest();
      if (actual != expected) {
        return PackageError("weights SHA-256 mismatch: expected " +
                            expected + ", got " + actual);
      }
    }
    for (const WeightEntry& entry : manifest.weights.entries) {
      if (!entry.available) {
        continue;
      }
      const auto meta_it = manifest.graph.tensors.find(entry.name);
      const auto tensor_it = tensor_index.find(entry.name);
      if (meta_it == manifest.graph.tensors.end() ||
          tensor_it == tensor_index.end()) {
        return PackageError("weight entry has no constant tensor: " +
                            entry.name);
      }
      U64 target_bytes = 0;
      status = WeightTargetBytes(entry, model.dtype, &target_bytes);
      if (!status.ok()) {
        return status;
      }
      std::vector<U8> host_bytes;
      status = reader.ReadAt(entry.offset, entry.byte_size, &host_bytes);
      if (!status.ok()) {
        return status;
      }
      if (verify_weights && !entry.sha256.empty()) {
        const std::string actual =
            Sha256::Hex(host_bytes.data(), host_bytes.size());
        if (actual != entry.sha256) {
          return PackageError("weight SHA-256 mismatch: " + entry.name);
        }
      }
      DataType source_dtype = DataType::kUnknown;
      status = DataTypeFromName(entry.dtype, &source_dtype);
      if (!status.ok()) {
        return status;
      }
      ParamDesc desc;
      desc.name = entry.name;
      desc.host_ptr = host_bytes.data();
      desc.bytes = entry.byte_size;
      desc.source_dtype = source_dtype;
      desc.target_dtype = model.dtype;
      std::unique_ptr<Buffer> converted;
      status = backend_state->raw->ConvertParam(desc, &converted);
      if (!status.ok()) {
        return PrefixStatus(status, "weight " + entry.name);
      }
      if (converted == nullptr || converted->Bytes() != target_bytes) {
        return PackageError("converted weight size mismatch: " +
                            entry.name);
      }
      runtime_graph::Tensor* tensor = tensor_it->second;
      if (converted->Bytes() != tensor->GetStorage().byte_size) {
        return PackageError(
            "converted weight size does not match tensor storage: " +
            entry.name);
      }
      auto holder = std::make_shared<BufferHolder>();
      holder->buffer = std::move(converted);
      stack->Push("weights/" + entry.name,
                  [backend_state, holder]() -> Status {
                    if (holder->buffer == nullptr ||
                        backend_state->raw == nullptr) {
                      return Status::Ok();
                    }
                    const Status release = backend_state->raw->DestroyBuffer(
                        std::move(holder->buffer));
                    holder->buffer.reset();
                    return release;
                  });
      BindDeviceStorage(tensor, holder->buffer.get(), 0,
                        holder->buffer->Bytes());
      if (weights_bytes > std::numeric_limits<U64>::max() -
                              holder->buffer->Bytes()) {
        return PackageError("weights bytes overflow");
      }
      weights_bytes += holder->buffer->Bytes();
    }
  }

  // 8. metadata-only/缺失权重时用占位缓冲绑定常量，仅用于加载期检查。
  Bool need_dummy_weights = false;
  for (const std::string& name : manifest.graph.constants) {
    const auto it = tensor_index.find(name);
    if (it == tensor_index.end()) {
      return PackageError("constant tensor is missing: " + name);
    }
    if (it->second->GetStorage().data_ptr == nullptr) {
      need_dummy_weights = true;
      break;
    }
  }
  if (need_dummy_weights) {
    if (!metadata_only) {
      for (const WeightEntry& entry : manifest.weights.entries) {
        if (!entry.available) {
          return PackageError("weight is not available: " + entry.name);
        }
      }
      return PackageError(
          "constants are not fully bound by weights entries");
    }
    BufferDesc desc;
    desc.bytes = kMetadataOnlyDummyBytes;
    desc.alignment = alignment;
    desc.usage = BufferUsage::kPersistent;
    std::shared_ptr<BufferHolder> dummy;
    status = AllocateTrackedBuffer(backend_state.get(), desc,
                                   "metadata_only_weights", stack, &dummy);
    if (!status.ok()) {
      return status;
    }
    for (const std::string& name : manifest.graph.constants) {
      runtime_graph::Tensor* tensor = tensor_index[name];
      if (tensor->GetStorage().data_ptr == nullptr) {
        BindDeviceStorage(tensor, dummy->buffer.get(), 0,
                          tensor->GetStorage().byte_size);
      }
    }
  }

  // 9. 输入缓冲。
  U64 io_bytes = 0;
  for (runtime_graph::Op* op : graph->GetAllInputOp()) {
    if (op->DstTensors().size() != 1) {
      return PackageError("input op must have exactly one tensor: " +
                          op->Name());
    }
    runtime_graph::Tensor* tensor = op->DstTensors()[0];
    const auto meta_it = manifest.graph.tensors.find(tensor->Name());
    if (meta_it == manifest.graph.tensors.end()) {
      return PackageError("input tensor has no metadata: " +
                          tensor->Name());
    }
    const TensorMeta& meta = meta_it->second;
    BufferDesc desc;
    desc.bytes = meta.byte_size;
    desc.alignment = alignment;
    desc.usage = BufferUsage::kIo;
    std::shared_ptr<BufferHolder> holder;
    status = AllocateTrackedBuffer(backend_state.get(), desc,
                                   "io/" + tensor->Name(), stack, &holder);
    if (!status.ok()) {
      return status;
    }
    BindDeviceStorage(tensor, holder->buffer.get(), 0, meta.byte_size);
    io_bytes += meta.byte_size;
  }

  // 10. activation arena。
  const U64 arena_bytes = manifest.memory_plan.arena_bytes;
  std::shared_ptr<BufferHolder> arena;
  if (arena_bytes > 0) {
    BufferDesc desc;
    desc.bytes = arena_bytes;
    desc.alignment = alignment;
    desc.usage = BufferUsage::kActivation;
    status = AllocateTrackedBuffer(backend_state.get(), desc,
                                   "activation_arena", stack, &arena);
    if (!status.ok()) {
      return status;
    }
  }
  for (const MemoryPlanEntry& entry : manifest.memory_plan.entries) {
    const auto tensor_it = tensor_index.find(entry.name);
    if (tensor_it == tensor_index.end()) {
      return PackageError("memory entry has no tensor: " + entry.name);
    }
    runtime_graph::Tensor* tensor = tensor_it->second;
    if (tensor->IsConstant()) {
      return PackageError("memory entry points at a constant: " +
                          entry.name);
    }
    if (arena == nullptr) {
      return PackageError("activation arena is missing for: " + entry.name);
    }
    BindDeviceStorage(tensor, arena->buffer.get(), entry.offset,
                      entry.size);
  }

  // 11. GraphEnv 与逐 op OpEnv。
  auto graph_env = std::make_unique<runtime_graph::GraphEnv>();
  status = graph->SetGraphEnv(std::move(graph_env));
  if (!status.ok()) {
    return PrefixStatus(status, "Graph::SetGraphEnv");
  }
  runtime_graph::GraphEnv* env = graph->GetGraphEnv();
  std::unique_ptr<OpEnvFactory> factory = CreateDefaultOpEnvFactory();
  if (factory == nullptr) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "default OpEnvFactory is not available");
  }
  std::set<runtime_graph::Op*> created;
  const auto create_op_env = [&](runtime_graph::Op* op) -> Status {
    if (!created.insert(op).second) {
      return Status::Ok();
    }
    std::unique_ptr<runtime_graph::OpEnv> op_env;
    Status create_status =
        factory->Create(op, backend_state->raw, env, &op_env);
    if (!create_status.ok()) {
      return PrefixStatus(create_status, "op=" + op->Name());
    }
    create_status = op->SetOpEnv(std::move(op_env));
    if (!create_status.ok()) {
      return PrefixStatus(create_status, "op=" + op->Name());
    }
    return Status::Ok();
  };
  for (runtime_graph::Op* op : graph->GetAllInputOp()) {
    status = create_op_env(op);
    if (!status.ok()) {
      return status;
    }
  }
  for (runtime_graph::Op* op : graph->GetOrderlyOPList()) {
    status = create_op_env(op);
    if (!status.ok()) {
      return status;
    }
  }
  for (runtime_graph::Op* op : graph->GetAllOutputOp()) {
    status = create_op_env(op);
    if (!status.ok()) {
      return status;
    }
  }

  // 12. GraphEnv 中登记的 workspace 统一分配并回填 data_ptr。
  if (env != nullptr) {
    for (const auto& item : env->SharedStorages()) {
      if (item.second.data_ptr != nullptr || item.second.byte_size == 0) {
        continue;
      }
      status = CheckBufferLimit(item.second.byte_size, capability,
                                "workspace/" + item.first);
      if (!status.ok()) {
        return status;
      }
      BufferDesc desc;
      desc.bytes = item.second.byte_size;
      desc.alignment = alignment;
      desc.usage = BufferUsage::kWorkspace;
      std::shared_ptr<BufferHolder> holder;
      status = AllocateTrackedBuffer(backend_state.get(), desc,
                                     "workspace/" + item.first, stack,
                                     &holder);
      if (!status.ok()) {
        return status;
      }
      runtime_graph::Storage* storage = env->GetSharedStorage(item.first);
      if (storage == nullptr) {
        return MakeStatus(ErrorCode::kInternal,
                          "workspace storage disappeared: " + item.first);
      }
      BindSharedStorage(storage, holder->buffer.get());
    }
  }

  // 13. 移交所有权并填充结果。
  result->graph = std::move(graph);
  result->backend = std::move(backend_state->owned);
  result->model = model;
  result->weights_bytes = weights_bytes;
  result->arena_bytes = arena_bytes;
  result->io_bytes = io_bytes;
  return Status::Ok();
}

}  // namespace

std::unique_ptr<ILoader> CreateDefaultLoader() {
  return std::make_unique<DefaultLoader>();
}

}  // namespace runtime
}  // namespace mobilenet
