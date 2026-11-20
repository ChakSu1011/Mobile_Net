// 模块：Loader 冒烟测试
//
// 职责：
//   用 Mock 后端加载模型包（默认 Qwen3-0.6B），打印固定头/manifest/图/
//   常量/权重/arena/I/O/OpEnv 统计，并校验所有张量已完成设备绑定；
//   支持 metadata-only 与跳过权重哈希的快速模式。
//
// 边界：
//   - 只验证加载期编排，不执行 Forward、不比较数值；
//   - metadata-only 模式下常量绑定占位缓冲，结果不可用于推理；
//   - 不写模型包、不修改仓库文件。
//
// 依赖：
//   Runtime/loader/{loader,manifest}.h、Runtime/backend/mock 注册、
//   Runtime/common/resource_stack.h、C++17 标准库。
//
// 用法：
//   loader_smoke [package_dir] [--metadata-only] [--no-verify]

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Graph/include/graph.h"
#include "Graph/include/graph_env.h"
#include "Graph/include/op.h"
#include "Graph/include/storage.h"
#include "Graph/include/tensor.h"
#include "Runtime/common/file_util.h"
#include "Runtime/common/resource_stack.h"
#include "Runtime/common/status.h"
#include "Runtime/loader/loader.h"
#include "Runtime/loader/manifest.h"
#include "Utils/cpp/types.h"

namespace {

// 拼接包目录与文件名。
//
// 参数：
//   directory：包目录；
//   name：文件名。
// 返回：
//   使用 '/' 连接的路径。
std::string JoinPackagePath(const std::string& directory,
                            const std::string& name) {
  if (!directory.empty() &&
      (directory.back() == '/' || directory.back() == '\\')) {
    return directory + name;
  }
  return directory + "/" + name;
}

// 打印失败状态并返回 1。
//
// 参数：
//   context：失败阶段；
//   status：失败状态。
// 返回：
//   恒为 1。
I32 PrintFailure(const C8* context, const mobilenet::Status& status) {
  std::printf("FAIL: %s: %s\n", context, status.message().c_str());
  return 1;
}

// 判断张量 storage 是否已绑定到设备。
//
// 参数：
//   tensor：目标张量。
// 返回：
//   true 表示 data_ptr 非空且 device_type=kDevice。
Bool IsBound(const mobilenet::runtime_graph::Tensor* tensor) {
  const mobilenet::runtime_graph::Storage& storage =
      tensor->GetStorage();
  return storage.data_ptr != nullptr &&
         storage.device_type ==
             mobilenet::runtime_graph::DeviceType::kDevice;
}

}  // namespace

// 测试入口。
//
// 参数：
//   argv[1]（可选）：模型包目录，默认 Models/Qwen3/artifacts/qwen3_0_6b；
//   --metadata-only：缺少/跳过 weights.bin，仅验证图与缓冲编排；
//   --no-verify：跳过权重 SHA-256 校验。
// 返回：
//   成功返回 0；失败返回 1。
I32 main(I32 argc, C8** argv) {
  std::string package = "Models/Qwen3/artifacts/qwen3_0_6b";
  std::string backend = "mock";
  Bool metadata_only = false;
  Bool verify_weights = true;
  for (I32 index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--metadata-only") {
      metadata_only = true;
    } else if (arg == "--no-verify") {
      verify_weights = false;
    } else if (arg == "--backend" && index + 1 < argc) {
      backend = argv[++index];
    } else if (arg == "--help") {
      std::printf(
          "usage: loader_smoke [package_dir] [--metadata-only] "
          "[--no-verify] [--backend name]\n");
      return 0;
    } else {
      package = arg;
    }
  }

  std::printf("package: %s\n", package.c_str());
  std::printf("mode: %s, verify_weights: %s, backend: %s\n",
              metadata_only ? "metadata-only" : "full-load",
              verify_weights ? "on" : "off", backend.c_str());

  // 直接读取头部与 manifest，打印加载期将要校验的元信息。
  mobilenet::runtime::ModelPackageHeader header;
  mobilenet::Status status = mobilenet::runtime::ReadModelPackageHeader(
      JoinPackagePath(package, "model.mgf"), &header);
  if (!status.ok()) {
    return PrintFailure("read header", status);
  }
  std::printf("header: version=%u flags=%u manifest_offset=%llu "
              "manifest_length=%llu has_weights=%d\n",
              static_cast<unsigned>(header.version),
              static_cast<unsigned>(header.flags),
              static_cast<unsigned long long>(header.manifest_offset),
              static_cast<unsigned long long>(header.manifest_length),
              header.has_weights ? 1 : 0);
  std::printf("header: weights_offset=%llu weights_length=%llu\n",
              static_cast<unsigned long long>(header.weights_offset),
              static_cast<unsigned long long>(header.weights_length));

  std::vector<U8> manifest_bytes;
  status = mobilenet::runtime::ReadFileRange(
      JoinPackagePath(package, "model.mgf"),
      header.manifest_offset, header.manifest_length, &manifest_bytes);
  if (!status.ok()) {
    return PrintFailure("read manifest", status);
  }
  mobilenet::runtime::ModelManifest manifest;
  status = mobilenet::runtime::ParseManifest(
      std::string(manifest_bytes.begin(), manifest_bytes.end()), &manifest);
  if (!status.ok()) {
    return PrintFailure("parse manifest", status);
  }
  std::printf("manifest: model=%s adapter=%s format=%s v%lld\n",
              manifest.model_name.c_str(), manifest.model_adapter.c_str(),
              manifest.format.c_str(),
              static_cast<long long>(manifest.format_version));
  std::printf("manifest: ir_sha256=%.16s... ir_line_count=%lld\n",
              manifest.graph.ir_sha256.c_str(),
              static_cast<long long>(manifest.graph.ir_line_count));
  std::printf("manifest: stats tensors=%llu ops=%llu constants=%llu "
              "outputs=%llu activations=%llu constant_bytes=%llu\n",
              static_cast<unsigned long long>(manifest.statistics.tensors),
              static_cast<unsigned long long>(manifest.statistics.ops),
              static_cast<unsigned long long>(manifest.statistics.constants),
              static_cast<unsigned long long>(manifest.statistics.outputs),
              static_cast<unsigned long long>(
                  manifest.statistics.activation_tensors),
              static_cast<unsigned long long>(
                  manifest.statistics.constant_bytes));
  std::printf("manifest: memory alignment=%llu arena=%llu peak=%llu "
              "entries=%llu\n",
              static_cast<unsigned long long>(
                  manifest.memory_plan.alignment),
              static_cast<unsigned long long>(
                  manifest.memory_plan.arena_bytes),
              static_cast<unsigned long long>(
                  manifest.memory_plan.peak_bytes),
              static_cast<unsigned long long>(
                  manifest.memory_plan.entries.size()));

  // 正式加载。
  mobilenet::runtime::LoadConfig config;
  config.package_dir = package;
  config.runtime.backend = backend;
  config.runtime.device_id = 0;
  config.runtime.queue_count = 1;
  config.runtime.verify_weights_sha256 = verify_weights;
  config.runtime.allow_metadata_only = metadata_only;

  mobilenet::runtime::ResourceStack stack;
  std::unique_ptr<mobilenet::runtime::ILoader> loader =
      mobilenet::runtime::CreateDefaultLoader();
  if (loader == nullptr) {
    std::printf("FAIL: CreateDefaultLoader returned null\n");
    return 1;
  }
  mobilenet::runtime::LoadResult result;
  status = loader->Load(config, &stack, &result);
  if (!status.ok()) {
    std::printf("FAIL: Load: [%s] %s\n",
                mobilenet::ErrorCodeName(status.code()),
                status.message().c_str());
    const mobilenet::Status release = stack.Clear();
    std::printf("rollback stack: size cleared, status=%s\n",
                release.message().c_str());
    return 1;
  }
  if (result.graph == nullptr || result.backend == nullptr) {
    std::printf("FAIL: Load returned empty graph/backend\n");
    return 1;
  }

  mobilenet::runtime_graph::Graph* graph = result.graph.get();
  std::printf("graph: name=%s inputs=%llu constants=%llu orderly_ops=%llu "
              "outputs=%llu\n",
              graph->Name().c_str(),
              static_cast<unsigned long long>(
                  graph->GetAllInputOp().size()),
              static_cast<unsigned long long>(
                  graph->GetAllConstantTensor().size()),
              static_cast<unsigned long long>(
                  graph->GetOrderlyOPList().size()),
              static_cast<unsigned long long>(
                  graph->GetAllOutputOp().size()));
  std::printf("inputs:");
  for (mobilenet::runtime_graph::Op* op : graph->GetAllInputOp()) {
    std::printf(" %s", op->Name().c_str());
  }
  std::printf("\noutputs:");
  for (mobilenet::runtime_graph::Op* op : graph->GetAllOutputOp()) {
    std::printf(" %s", op->Name().c_str());
  }
  std::printf("\n");
  if (graph->GetSampleOp() != nullptr) {
    std::printf("sample op: %s dst=%s\n",
                graph->GetSampleOp()->Name().c_str(),
                graph->GetSampleOp()->DstTensors().empty()
                    ? "?"
                    : graph->GetSampleOp()->DstTensors()[0]->Name().c_str());
  }

  const auto& ops = graph->GetOrderlyOPList();
  const Usize show = std::min<Usize>(ops.size(), 3);
  for (Usize index = 0; index < show; ++index) {
    std::printf("op[%llu] %s type=%s\n",
                static_cast<unsigned long long>(index),
                ops[index]->Name().c_str(),
                mobilenet::runtime_graph::OpTypeToStr(
                    ops[index]->Type()));
  }
  if (ops.size() > show) {
    const Usize last = ops.size() - 1;
    std::printf("op[%llu] %s type=%s (last)\n",
                static_cast<unsigned long long>(last),
                ops[last]->Name().c_str(),
                mobilenet::runtime_graph::OpTypeToStr(
                    ops[last]->Type()));
  }

  U64 constant_bytes = 0;
  for (mobilenet::runtime_graph::Tensor* tensor :
       graph->GetAllConstantTensor()) {
    constant_bytes += tensor->GetStorage().byte_size;
  }
  std::printf("weights: result.weights_bytes=%llu constants_sum=%llu "
              "constants=%llu\n",
              static_cast<unsigned long long>(result.weights_bytes),
              static_cast<unsigned long long>(constant_bytes),
              static_cast<unsigned long long>(
                  graph->GetAllConstantTensor().size()));
  if (!graph->GetAllConstantTensor().empty()) {
    const mobilenet::runtime_graph::Tensor* first =
        graph->GetAllConstantTensor().front();
    const mobilenet::runtime_graph::Tensor* last =
        graph->GetAllConstantTensor().back();
    std::printf("constants: first=%s bytes=%llu bound=%d, last=%s "
                "bytes=%llu bound=%d\n",
                first->Name().c_str(),
                static_cast<unsigned long long>(
                    first->GetStorage().byte_size),
                IsBound(first) ? 1 : 0, last->Name().c_str(),
                static_cast<unsigned long long>(
                    last->GetStorage().byte_size),
                IsBound(last) ? 1 : 0);
  }

  U64 max_plan_end = 0;
  for (const mobilenet::runtime::MemoryPlanEntry& entry :
       manifest.memory_plan.entries) {
    max_plan_end = std::max(max_plan_end, entry.offset + entry.size);
  }
  std::printf("arena: result.arena_bytes=%llu plan_arena=%llu "
              "max_entry_end=%llu entries=%llu\n",
              static_cast<unsigned long long>(result.arena_bytes),
              static_cast<unsigned long long>(
                  manifest.memory_plan.arena_bytes),
              static_cast<unsigned long long>(max_plan_end),
              static_cast<unsigned long long>(
                  manifest.memory_plan.entries.size()));
  std::printf("io: result.io_bytes=%llu\n",
              static_cast<unsigned long long>(result.io_bytes));

  U64 workspace_bytes = 0;
  Usize workspace_count = 0;
  Bool workspace_bound = true;
  if (graph->GetGraphEnv() != nullptr) {
    for (const auto& item :
         graph->GetGraphEnv()->SharedStorages()) {
      if (item.second.byte_size == 0) {
        continue;
      }
      ++workspace_count;
      workspace_bytes += item.second.byte_size;
      if (item.second.data_ptr == nullptr) {
        workspace_bound = false;
      }
    }
  }
  std::printf("workspace: buffers=%llu bytes=%llu bound=%d\n",
              static_cast<unsigned long long>(workspace_count),
              static_cast<unsigned long long>(workspace_bytes),
              workspace_bound ? 1 : 0);

  std::printf("model_config: layers=%d heads=%d kv_heads=%d head_dim=%d "
              "hidden=%lld intermediate=%lld vocab=%lld context=%lld\n",
              result.model.num_layers, result.model.num_heads,
              result.model.num_kv_heads, result.model.head_dim,
              static_cast<long long>(result.model.hidden_size),
              static_cast<long long>(result.model.intermediate_size),
              static_cast<long long>(result.model.vocab_size),
              static_cast<long long>(result.model.context_length));
  std::printf("model_config: dtype=%s eps=%g rope_theta=%g "
              "max_num_tokens=%d max_batch=%d pad=%lld bos=%lld "
              "stop_ids=%llu\n",
              mobilenet::DataTypeName(result.model.dtype),
              static_cast<double>(result.model.rms_norm_eps),
              result.model.rope_theta, result.model.max_num_tokens,
              result.model.max_batch_size,
              static_cast<long long>(result.model.pad_token_id),
              static_cast<long long>(result.model.bos_token_id),
              static_cast<unsigned long long>(
                  result.model.stop_token_ids.size()));

  // 全部张量绑定检查：Input/常量/计算 op 的 src/dst 必须已绑定设备。
  Bool bound = true;
  for (mobilenet::runtime_graph::Op* op : graph->GetAllInputOp()) {
    for (mobilenet::runtime_graph::Tensor* tensor : op->DstTensors()) {
      bound = bound && IsBound(tensor);
    }
  }
  for (mobilenet::runtime_graph::Tensor* tensor :
       graph->GetAllConstantTensor()) {
    bound = bound && IsBound(tensor);
  }
  for (mobilenet::runtime_graph::Op* op : graph->GetOrderlyOPList()) {
    if (op->Type() == mobilenet::runtime_graph::OpType::kInput ||
        op->Type() == mobilenet::runtime_graph::OpType::kOutput) {
      continue;
    }
    for (mobilenet::runtime_graph::Tensor* tensor : op->SrcTensors()) {
      bound = bound && IsBound(tensor);
    }
    for (mobilenet::runtime_graph::Tensor* tensor : op->DstTensors()) {
      bound = bound && IsBound(tensor);
    }
  }
  std::printf("binding check: %s\n", bound ? "OK" : "FAIL");

  // 释放顺序模拟 Destroy：先清 OpEnv（Graph 析构），再清 ResourceStack，
  // 最后释放 Backend。
  result.graph.reset();
  const mobilenet::Status release = stack.Clear();
  if (!release.ok()) {
    std::printf("FAIL: resource release: %s\n",
                release.message().c_str());
  }
  result.backend.reset();
  std::printf("release: status=%s\n",
              release.ok() ? "OK" : "FAIL");

  const Bool passed = bound && release.ok() && workspace_bound;
  std::printf("%s\n", passed ? "loader smoke PASSED" : "loader smoke FAILED");
  return passed ? 0 : 1;
}
