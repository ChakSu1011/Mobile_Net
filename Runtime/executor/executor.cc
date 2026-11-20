// 模块：Executor 默认实现（ExecBackend）
//
// 职责：
//   实现 IExecutor：Init 阶段接管 Backend，按 Graph 的 Input/Output op 及其
//   Tensor Storage 预分配主机 I/O 缓冲；Execute 阶段把 ExecutionPlan 打包成
//   token/position 输入，经 IoBinder 绑定主机 Storage，逐 op 驱动
//   OpEnv::Forward，同步后回读 sample 输出并填充 StepResult；Deinit 阶段
//   释放 Backend 与全部缓冲。
//
// 边界：
//   - 不做调度、KV 分配/释放、停止判定，也不构建 Response；
//   - 不创建或销毁 OpEnv，不拥有 Graph/IoBinder 的所有权；
//   - 不包含后端专有头文件，不写死模型形状、词表大小或输入名称；
//   - 仅允许单推理线程调用，内部不加锁；
//   - 公共头文件不改动：本文件只实现 executor.h 已声明的接口。
//
// 依赖：
//   Runtime/executor/executor.h、Runtime/backend/backend.h、
//   Runtime/backend/exec_op_env.h、Runtime/common/io_binder.h、
//   Graph/include/{graph,op,storage,tensor}.h、C++17 标准库。

#include "Runtime/executor/executor.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Graph/include/graph.h"
#include "Graph/include/op.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/backend/backend.h"
#include "Runtime/backend/exec_op_env.h"
#include "Runtime/common/io_binder.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {
namespace {

// I64 token 的字节数；token/position/sample 缓冲均按该单位换算容量。
constexpr Usize kI64Bytes = sizeof(I64);

// 单个 Input/Output op 的主机 I/O 缓冲槽。
//
// 功能：
//   保存 op 名称、主机 Storage 模板与预分配的主机字节缓冲；Init 时建立，
//   Execute 时直接复用，避免每步分配。
//
// 边界：
//   - storage 模板的 shape/precision/layout/byte_size 复制自设备 Storage，
//     device_type 固定 kHost、offset 固定 0、data_ptr 在绑定前必须重新指向
//     buffer；
//   - 只在 Executor 内部使用，不暴露到公共头。
struct HostIoSlot {
  // op 名称，作为 IoBinder::SetInput/SetOutput 的键。
  std::string name;

  // 主机 Storage 模板；data_ptr 为空表示尚未绑定。
  runtime_graph::Storage storage;

  // 预分配的主机字节缓冲，大小等于 storage.byte_size。
  std::vector<U8> buffer;
};

// 解析 Storage 的有效字节数。
//
// 功能：
//   优先使用 storage.byte_size；当其未设置（0）时，按 shape 元素数与精度
//   字节数推导，兼容只声明形状的图。
//
// 参数：
//   storage：设备侧 Storage；
//   role：日志用角色名，例如 "input.input_ids"；
//   bytes：输出参数，成功时写入正字节数。
//
// 返回：
//   成功返回 kOk；bytes 为空、shape 非法或字节数溢出返回 kInvalidArgument；
//   精度无法计算字节数返回 kUnsupportedDtype。
Status ResolveByteSize(const runtime_graph::Storage& storage,
                       const std::string& role, Usize* bytes) {
  if (bytes == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Executor: byte size output must not be null");
  }
  if (storage.byte_size > 0) {
    *bytes = storage.byte_size;
    return Status::Ok();
  }
  const I64 elements = storage.shape.GetShapeSize();
  const I64 element_bytes =
      runtime_graph::PrecisionByteSize(storage.precision);
  if (elements <= 0 || element_bytes <= 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        role + ": storage byte_size and shape/precision are invalid");
  }
  if (static_cast<U64>(elements) >
      std::numeric_limits<Usize>::max() /
          static_cast<Usize>(element_bytes)) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      role + ": storage byte size overflows");
  }
  *bytes = static_cast<Usize>(elements) *
           static_cast<Usize>(element_bytes);
  return Status::Ok();
}

// 为一个 Input/Output op 构造主机 I/O 槽。
//
// 功能：
//   校验 op 恰有一个边界 tensor，复制其 Storage 的形状/精度/布局与字节数，
//   分配并清零主机缓冲。
//
// 参数：
//   op：Input 或 Output op；
//   is_input：true 取 DstTensors（Input），false 取 SrcTensors（Output）；
//   slot：输出参数，成功时写入名称、模板与缓冲。
//
// 返回：
//   成功返回 kOk；op 为空或 tensor 数量不为 1 返回 kInvalidArgument；
//   精度/字节数非法时透传 ResolveByteSize 的错误。
Status BuildHostIoSlot(runtime_graph::Op* op, Bool is_input,
                       HostIoSlot* slot) {
  if (op == nullptr || slot == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Executor: op/slot must not be null");
  }
  const std::vector<runtime_graph::Tensor*>& tensors =
      is_input ? op->DstTensors() : op->SrcTensors();
  if (tensors.size() != 1 || tensors[0] == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "op=" + op->Name() + ": " +
            (is_input ? "input" : "output") +
            " op must have exactly one tensor");
  }
  const runtime_graph::Storage& device = tensors[0]->GetStorage();
  Usize bytes = 0;
  Status status = ResolveByteSize(device, op->Name(), &bytes);
  if (!status.ok()) {
    return status;
  }

  slot->name = op->Name();
  slot->storage = runtime_graph::Storage{};
  slot->storage.shape = device.shape;
  slot->storage.precision = device.precision;
  slot->storage.layout = device.layout;
  slot->storage.device_type = runtime_graph::DeviceType::kHost;
  slot->storage.offset = 0;
  slot->storage.byte_size = bytes;
  slot->storage.data_ptr = nullptr;
  slot->storage.reusable = false;
  slot->buffer.assign(bytes, 0);
  return Status::Ok();
}

// 校验 packed token/position 输入槽。
//
// 功能：
//   确认该输入是 int64 且字节数能被 8 整除；plan 中的 token_ids/positions
//   是 I64，只有 int64 输入可以无损承载。
//
// 参数：
//   slot：Input op 的主机槽；
//   role：日志角色名。
//
// 返回：
//   成功返回 kOk；精度不是 int64 返回 kUnsupportedDtype；字节数不是 8 的
//   整数倍返回 kInvalidArgument。
Status CheckPackedInputSlot(const HostIoSlot& slot,
                            const std::string& role) {
  if (slot.storage.precision != runtime_graph::Precision::kInt64) {
    return MakeStatus(
        ErrorCode::kUnsupportedDtype,
        role + ": token/position input must be int64");
  }
  if (slot.buffer.size() % kI64Bytes != 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        role + ": byte_size is not a multiple of 8");
  }
  return Status::Ok();
}

// 把 PlanEntry 中的一列 I64 数据按 packed 下标写入主机缓冲。
//
// 功能：
//   先清零整个缓冲，再按 entry.packed_offset 与 entry.token_count 拷贝
//   token_ids 或 positions；未被覆盖的位置保持 0，满足 padding 约定。
//
// 参数：
//   plan：本步执行计划；
//   total：所有 entry 的 token_count 之和；
//   field：PlanEntry 的成员指针（&PlanEntry::token_ids 或 positions）；
//   buffer：主机字节缓冲；
//   role：日志角色名。
//
// 返回：
//   成功返回 kOk；buffer 为空、字节数不是 8 的整数倍或 total 超过缓冲
//   容量返回 kInvalidArgument。
Status FillPackedI64(const ExecutionPlan& plan, I64 total,
                     const std::vector<I64> PlanEntry::*field,
                     std::vector<U8>* buffer, const std::string& role) {
  if (buffer == nullptr || field == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      role + ": buffer/field must not be null");
  }
  if (buffer->size() % kI64Bytes != 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        role + ": byte_size is not a multiple of 8");
  }
  const Usize capacity = buffer->size() / kI64Bytes;
  if (total < 0 || static_cast<U64>(total) > capacity) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        role + ": packed tokens exceed host buffer capacity");
  }
  std::memset(buffer->data(), 0, buffer->size());
  for (const PlanEntry& entry : plan.entries) {
    const std::vector<I64>& values = entry.*field;
    for (I32 i = 0; i < entry.token_count; ++i) {
      const I64 value = values[static_cast<Usize>(i)];
      const Usize index =
          static_cast<Usize>(entry.packed_offset) +
          static_cast<Usize>(i);
      std::memcpy(buffer->data() + index * kI64Bytes, &value,
                  kI64Bytes);
    }
  }
  return Status::Ok();
}

// 校验 ExecutionPlan 的尺寸与 packed 布局。
//
// 功能：
//   检查 entries 非空、token_count 为正、token_ids/positions 长度与
//   token_count 一致、packed 区间不越过 total，并验证
//   plan.total_tokens 等于各 entry 之和。
//
// 参数：
//   plan：本步执行计划；
//   total：输出参数，成功时写入 token 总数。
//
// 返回：
//   成功返回 kOk；任何尺寸/编号不一致返回 kInvalidArgument。
Status ValidatePlan(const ExecutionPlan& plan, I64* total) {
  if (total == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Executor: total output must not be null");
  }
  if (plan.entries.empty()) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Execute: plan has no entries");
  }
  if (plan.total_tokens < 0) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Execute: plan.total_tokens must not be negative");
  }
  I64 sum = 0;
  for (const PlanEntry& entry : plan.entries) {
    if (entry.token_count <= 0) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Execute: request " + std::to_string(entry.request_id) +
              " has non-positive token_count");
    }
    if (entry.packed_offset < 0) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Execute: request " + std::to_string(entry.request_id) +
              " has negative packed_offset");
    }
    const Usize count = static_cast<Usize>(entry.token_count);
    if (entry.token_ids.size() != count ||
        entry.positions.size() != count) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Execute: request " + std::to_string(entry.request_id) +
              " token_ids/positions size mismatch");
    }
    if (sum > std::numeric_limits<I32>::max() - entry.token_count) {
      return MakeStatus(ErrorCode::kInvalidArgument,
                        "Execute: packed token count overflows int32");
    }
    sum += entry.token_count;
  }
  if (plan.total_tokens != static_cast<I32>(sum)) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Execute: plan.total_tokens does not match entry sum");
  }
  for (const PlanEntry& entry : plan.entries) {
    if (entry.packed_offset > sum - entry.token_count) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Execute: request " + std::to_string(entry.request_id) +
              " packed range exceeds total tokens");
    }
  }
  *total = sum;
  return Status::Ok();
}

// 由主机槽生成传给 IoBinder 的主机 Storage。
//
// 参数：
//   slot：主机 I/O 槽，buffer 非空。
//
// 返回：
//   绑定到 slot.buffer 的主机 Storage；offset 为 0。
runtime_graph::Storage MakeHostStorage(HostIoSlot* slot) {
  runtime_graph::Storage host = slot->storage;
  host.device_type = runtime_graph::DeviceType::kHost;
  host.offset = 0;
  host.data_ptr = slot->buffer.data();
  host.reusable = false;
  return host;
}

// 默认 Executor 实现。
//
// 功能：
//   持有 Backend 所有权，缓存 Graph/IoBinder 借用指针与主机 I/O 槽，按
//   IExecutor 契约完成 Init/Execute/Deinit。
//
// 边界：
//   - 单线程使用；
//   - Backend 必须比所有 OpEnv/Kernel 更晚释放；Engine 在 Deinit 前先清理
//     OpEnv；
//   - 不缓存 plan/kv_view 指针，Execute 返回后不再引用。
class ExecBackend final : public IExecutor {
 public:
  // 构造空 Executor；未 Init 时 backend() 返回 nullptr。
  ExecBackend() = default;

  // 析构时释放 Backend 与缓冲；忽略 Deinit 错误。
  ~ExecBackend() override;

  ExecBackend(const ExecBackend&) = delete;
  ExecBackend& operator=(const ExecBackend&) = delete;
  ExecBackend(ExecBackend&&) = delete;
  ExecBackend& operator=(ExecBackend&&) = delete;

  // 初始化执行器并接管 Backend。
  //
  // 参数：
  //   backend：非空后端，所有权移交；
  //   graph：已绑定 OpEnv 的图，非空；
  //   io_binder：I/O 绑定接口，非空。
  // 返回：
  //   成功返回 kOk；重复 Init 返回 kStateError；缺少 Input/Output/Sample
  //   op 返回 kUnsupportedOp；tensor/Storage 非法返回 kInvalidArgument；
  //   失败时由 backend 参数自动释放后端，不留下半初始化资源。
  Status Init(std::unique_ptr<Backend> backend, runtime_graph::Graph* graph,
              IoBinder* io_binder) override;

  // 返回拥有的 Backend；未初始化时为 nullptr。
  Backend* backend() override;

  // 执行一个 Step。
  //
  // 参数：
  //   plan：本步计划，不可变快照；
  //   kv_view：KV 只读视图；
  //   result：输出参数，按请求写入 token 与成功状态。
  // 返回：
  //   成功返回 kOk；参数/尺寸/I/O 绑定失败返回 kInvalidArgument；缺少
  //   OpEnv、未初始化或 sample 不可达返回 kStateError；Forward/Synchronize/
  //   Download 失败透传其错误码。失败时不写 result 的业务字段。
  Status Execute(const ExecutionPlan& plan, const KvCacheView& kv_view,
                 StepResult* result) override;

  // 释放 Backend 与全部缓冲；幂等。
  //
  // 返回：
  //   Backend::Deinit 成功或未初始化时返回 kOk；否则返回带前缀的错误码，
  //   但 Backend 仍会被释放。
  Status Deinit() override;

 private:
  // 拥有的后端；非空表示已 Init。
  std::unique_ptr<Backend> backend_;

  // 图借用指针；Init 成功后非空。
  runtime_graph::Graph* graph_ = nullptr;

  // I/O 绑定借用指针；Init 成功后非空。
  IoBinder* io_binder_ = nullptr;

  // 按 Graph::GetAllInputOp() 顺序缓存的主机输入槽；第 0 个是 token_ids，
  // 第 1 个（存在时）是 position_ids。
  std::vector<HostIoSlot> input_slots_;

  // 按 Graph::GetAllOutputOp() 顺序缓存的主机输出槽。
  std::vector<HostIoSlot> output_slots_;
};

ExecBackend::~ExecBackend() {
  const Status status = Deinit();
  (void)status;
}

Status ExecBackend::Init(std::unique_ptr<Backend> backend,
                         runtime_graph::Graph* graph,
                         IoBinder* io_binder) {
  if (backend_ != nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "Executor::Init: already initialized");
  }
  if (backend == nullptr || graph == nullptr || io_binder == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Executor::Init: backend/graph/io_binder must not be null");
  }

  const std::vector<runtime_graph::Op*>& inputs = graph->GetAllInputOp();
  const std::vector<runtime_graph::Op*>& outputs = graph->GetAllOutputOp();
  if (inputs.empty()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "Executor::Init: graph has no Input op");
  }
  if (outputs.empty()) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "Executor::Init: graph has no Output op");
  }
  runtime_graph::Op* sample = graph->GetSampleOp();
  if (sample == nullptr) {
    return MakeStatus(ErrorCode::kUnsupportedOp,
                      "Executor::Init: graph has no Sample op");
  }
  if (sample->DstTensors().size() != 1 ||
      sample->DstTensors()[0] == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Executor::Init: sample op must have exactly one output tensor");
  }

  std::vector<HostIoSlot> input_slots;
  input_slots.reserve(inputs.size());
  for (runtime_graph::Op* op : inputs) {
    HostIoSlot slot;
    const Status status = BuildHostIoSlot(op, true, &slot);
    if (!status.ok()) {
      return status;
    }
    input_slots.push_back(std::move(slot));
  }

  std::vector<HostIoSlot> output_slots;
  output_slots.reserve(outputs.size());
  for (runtime_graph::Op* op : outputs) {
    HostIoSlot slot;
    const Status status = BuildHostIoSlot(op, false, &slot);
    if (!status.ok()) {
      return status;
    }
    output_slots.push_back(std::move(slot));
  }

  // 全部校验与分配成功后再提交状态；失败时 backend 参数析构释放后端。
  backend_ = std::move(backend);
  graph_ = graph;
  io_binder_ = io_binder;
  input_slots_ = std::move(input_slots);
  output_slots_ = std::move(output_slots);
  return Status::Ok();
}

Backend* ExecBackend::backend() { return backend_.get(); }

Status ExecBackend::Execute(const ExecutionPlan& plan,
                            const KvCacheView& kv_view,
                            StepResult* result) {
  if (backend_ == nullptr || graph_ == nullptr ||
      io_binder_ == nullptr) {
    return MakeStatus(ErrorCode::kStateError,
                      "Executor::Execute: executor is not initialized");
  }
  if (result == nullptr) {
    return MakeStatus(ErrorCode::kInvalidArgument,
                      "Executor::Execute: result must not be null");
  }
  result->requests.clear();

  I64 total = 0;
  Status status = ValidatePlan(plan, &total);
  if (!status.ok()) {
    return status;
  }
  if (input_slots_.empty() || output_slots_.empty()) {
    return MakeStatus(ErrorCode::kStateError,
                      "Executor::Execute: I/O slots are not initialized");
  }

  // 第 1 个输入承载 token_ids，第 2 个（存在时）承载 position_ids。
  status = CheckPackedInputSlot(input_slots_[0], input_slots_[0].name);
  if (!status.ok()) {
    return status;
  }
  if (input_slots_.size() > 1) {
    status = CheckPackedInputSlot(input_slots_[1], input_slots_[1].name);
    if (!status.ok()) {
      return status;
    }
  }

  status = FillPackedI64(plan, total, &PlanEntry::token_ids,
                         &input_slots_[0].buffer, input_slots_[0].name);
  if (!status.ok()) {
    return status;
  }
  if (input_slots_.size() > 1) {
    status =
        FillPackedI64(plan, total, &PlanEntry::positions,
                      &input_slots_[1].buffer, input_slots_[1].name);
    if (!status.ok()) {
      return status;
    }
  }

  // 绑定输入与输出主机 Storage；任一步失败即终止本步。
  for (HostIoSlot& slot : input_slots_) {
    status = io_binder_->SetInput(slot.name, MakeHostStorage(&slot));
    if (!status.ok()) {
      return PrefixStatus(status,
                          "IoBinder::SetInput(" + slot.name + ")");
    }
  }
  for (HostIoSlot& slot : output_slots_) {
    status = io_binder_->SetOutput(slot.name, MakeHostStorage(&slot));
    if (!status.ok()) {
      return PrefixStatus(status,
                          "IoBinder::SetOutput(" + slot.name + ")");
    }
  }

  // 构造本步上下文：seq_lens 与 plan 的 entry 一一对应，生命周期覆盖整个
  // op 遍历过程。
  std::vector<I32> seq_lens;
  seq_lens.reserve(plan.entries.size());
  for (const PlanEntry& entry : plan.entries) {
    seq_lens.push_back(entry.token_count);
  }

  StepContext context;
  context.backend = backend_.get();
  context.stream = nullptr;
  context.plan = &plan;
  context.kv_view = &kv_view;
  context.seq_lens = seq_lens.data();
  context.batch = static_cast<I32>(plan.entries.size());
  context.packed_tokens = static_cast<I32>(total);
  context.is_prefill =
      plan.entries.front().phase == RequestPhase::kPrefill;

  runtime_graph::Op* sample_op = graph_->GetSampleOp();
  Bool sample_visited = false;
  for (runtime_graph::Op* op : graph_->GetOrderlyOPList()) {
    if (op == nullptr) {
      return MakeStatus(ErrorCode::kStateError,
                        "Executor::Execute: graph contains a null op");
    }
    runtime_graph::OpEnv* op_env = op->GetOpEnv();
    if (op_env == nullptr) {
      return MakeStatus(
          ErrorCode::kStateError,
          "op=" + op->Name() + ": OpEnv is not bound");
    }
    status = SetStepContext(op_env, &context);
    if (!status.ok()) {
      return PrefixStatus(status, "op=" + op->Name());
    }
    status = op_env->Forward();
    const Status clear_status = SetStepContext(op_env, nullptr);
    if (!status.ok()) {
      return PrefixStatus(status, "op=" + op->Name());
    }
    if (!clear_status.ok()) {
      return PrefixStatus(clear_status, "op=" + op->Name());
    }
    if (op == sample_op) {
      sample_visited = true;
    }
  }
  if (!sample_visited) {
    return MakeStatus(
        ErrorCode::kStateError,
        "Executor::Execute: sample op is not reachable from outputs");
  }

  status = backend_->Synchronize(nullptr);
  if (!status.ok()) {
    return PrefixStatus(status, "Backend::Synchronize");
  }

  // 从 sample.out 的 Storage 回读 token：data_ptr 是 Buffer*，加上 offset
  // 得到有效设备地址。
  const runtime_graph::Storage& sample_storage =
      sample_op->DstTensors()[0]->GetStorage();
  if (sample_storage.precision != runtime_graph::Precision::kInt64) {
    return MakeStatus(ErrorCode::kUnsupportedDtype,
                      "Executor::Execute: sample output must be int64");
  }
  if (sample_storage.data_ptr == nullptr) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Executor::Execute: sample output storage is not bound");
  }
  if (sample_storage.byte_size == 0 ||
      sample_storage.byte_size % kI64Bytes != 0) {
    return MakeStatus(
        ErrorCode::kInvalidArgument,
        "Executor::Execute: sample output byte_size is invalid");
  }
  auto* sample_buffer =
      static_cast<Buffer*>(sample_storage.data_ptr);
  const void* sample_device = sample_buffer->DevicePtr();
  if (sample_device == nullptr) {
    return MakeStatus(
        ErrorCode::kInternal,
        "Executor::Execute: sample output device pointer is null");
  }
  const auto* sample_bytes =
      static_cast<const U8*>(sample_device) + sample_storage.offset;
  std::vector<I64> sample_tokens(sample_storage.byte_size / kI64Bytes, 0);
  status = backend_->Download(sample_tokens.data(), sample_bytes,
                              sample_storage.byte_size, nullptr);
  if (!status.ok()) {
    return PrefixStatus(status, "Executor::Download(sample)");
  }

  // 先校验全部 output_slot，避免失败时留下部分结果。
  for (const PlanEntry& entry : plan.entries) {
    if (entry.output_slot < 0) {
      continue;
    }
    if (static_cast<Usize>(entry.output_slot) >= sample_tokens.size()) {
      return MakeStatus(
          ErrorCode::kInvalidArgument,
          "Executor::Execute: request " +
              std::to_string(entry.request_id) +
              " output_slot exceeds sample output size");
    }
  }

  // 按 output_slot 把 packed 位置的 token 写回对应请求；output_slot<0 表示
  // 本步不产生 token（例如未完成的 chunked prefill），跳过该请求。
  result->requests.reserve(plan.entries.size());
  for (const PlanEntry& entry : plan.entries) {
    if (entry.output_slot < 0) {
      continue;
    }
    const Usize slot = static_cast<Usize>(entry.output_slot);
    RequestStepResult request_result;
    request_result.request_id = entry.request_id;
    request_result.token.token_id = sample_tokens[slot];
    request_result.token.token_prob = 0.0f;
    request_result.token.log_prob = 0.0f;
    request_result.token.select = true;
    request_result.ok = true;
    request_result.error = ErrorCode::kOk;
    result->requests.push_back(request_result);
  }
  return Status::Ok();
}

Status ExecBackend::Deinit() {
  Status status = Status::Ok();
  if (backend_ != nullptr) {
    status = backend_->Deinit();
    if (!status.ok()) {
      status = PrefixStatus(status, "Backend::Deinit");
    }
    backend_.reset();
  }
  graph_ = nullptr;
  io_binder_ = nullptr;
  input_slots_.clear();
  output_slots_.clear();
  return status;
}

}  // namespace

std::unique_ptr<IExecutor> CreateDefaultExecutor() {
  return std::make_unique<ExecBackend>();
}

}  // namespace runtime
}  // namespace mobilenet
