// 模块：Runtime Mock 后端 Qwen3-0.6B 静态端到端测试
//
// 职责：
//   用 Mock 后端加载 Qwen3-0.6B 模型包，执行单步完整序列前向，比较
//   sample.out 的 argmax token 与 PyTorch 参考值，验证 Loader/Scheduler/
//   KVCache/Executor/Destroy/Engine 的通用链路。
//
// 边界：
//   - 只验证 token 一致性，不比较 logits 数值；
//   - 使用 recompute_attention 模式（Mock attention 不支持 KV cache），
//     每步重算完整序列；
//   - 测试数据针对 Qwen3-0.6B，但 Runtime 代码不写死模型。
//
// 依赖：
//   Runtime 全部模块、Mock 后端、Graph 静态库。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "Runtime/common/sequence.h"
#include "Runtime/engine/engine.h"
#include "Runtime/engine/engine_modules.h"
#include "Utils/cpp/types.h"

namespace {

// 单条参考用例。
struct GoldenCase {
  // 用例文本，仅用于输出。
  const C8* text = "";
  // prompt token IDs。
  std::vector<I64> prompt_ids;
  // PyTorch 参考 argmax token。
  I64 expected_token = -1;
};

// 检查布尔条件。
Bool Check(Bool condition, const C8* message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    return false;
  }
  return true;
}

}  // namespace

// 测试入口。
//
// 参数：
//   argv[1]（可选）：模型包目录，默认 Models/Qwen3/artifacts/qwen3_0_6b；
//   argv[2]（可选）：仅运行指定下标的用例，缺省运行全部。
// 返回：
//   全部用例通过返回 0；否则返回 1。
I32 main(I32 argc, C8** argv) {
  const std::string package =
      argc > 1 ? argv[1] : "Models/Qwen3/artifacts/qwen3_0_6b";
  const I32 only_case = argc > 2 ? std::atoi(argv[2]) : -1;
  const std::vector<GoldenCase> cases = {
      {"Hello", {9707}, 21806},
      {"The capital of France is", {785, 6722, 315, 9625, 374}, 12095},
      {"1+1=", {16, 10, 16, 28}, 17},
  };

  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = package;
  config.runtime.backend = "mock";
  config.runtime.kv_cache_bytes = 8ull * 1024 * 1024;
  config.runtime.max_num_tokens = 256;
  config.runtime.max_concurrent_requests = 1;
  config.runtime.recompute_attention = true;
  config.runtime.threaded = false;

  std::map<U64, I64> tokens;
  engine.SetResponseCallback([&tokens](
                                 const mobilenet::runtime::Response& response) {
    if (!response.sequences.empty()) {
      tokens[response.request_id] =
          response.sequences[0].current_token.token_id;
    }
  });

  mobilenet::Status status = engine.Load(config);
  if (!status.ok()) {
    std::printf("Load failed: %s\n", status.message().c_str());
    return 1;
  }

  I32 failures = 0;
  for (Usize case_index = 0; case_index < cases.size(); ++case_index) {
    if (only_case >= 0 && static_cast<I32>(case_index) != only_case) {
      continue;
    }
    const GoldenCase& item = cases[case_index];
    mobilenet::runtime::Request request;
    request.prompt_token_ids = item.prompt_ids;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens = 1;
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = true;

    U64 request_id = 0;
    status = engine.AddRequest(request, &request_id);
    if (!status.ok()) {
      std::printf("AddRequest failed (%s): %s\n", item.text,
                  status.message().c_str());
      ++failures;
      continue;
    }
    mobilenet::runtime::StepStats stats;
    status = engine.RunOnce(&stats);
    if (!status.ok()) {
      std::printf("RunOnce failed (%s): %s\n", item.text,
                  status.message().c_str());
      ++failures;
      continue;
    }
    const auto it = tokens.find(request_id);
    if (it == tokens.end()) {
      std::printf("FAIL: no response for %s\n", item.text);
      ++failures;
      continue;
    }
    std::printf("%-24s expected=%lld actual=%lld\n", item.text,
                static_cast<long long>(item.expected_token),
                static_cast<long long>(it->second));
    if (!Check(it->second == item.expected_token,
               "token mismatch vs PyTorch reference")) {
      ++failures;
    }
  }

  status = engine.Destroy();
  if (!status.ok()) {
    std::printf("Destroy failed: %s\n", status.message().c_str());
    ++failures;
  }
  if (failures != 0) {
    std::printf("%d mock e2e checks failed\n", failures);
    return 1;
  }
  std::printf("all mock e2e checks passed\n");
  return 0;
}
