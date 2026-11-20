// 模块：OCL 后端 Qwen3-0.6B 端到端一致性测试
//
// 职责：
//   用 OCL 后端加载 Qwen3-0.6B 模型包，执行单步完整序列前向，并与同进程
//   内 Mock 后端使用相同 prompt token 的生成 token 逐用例比较，同时校验
//   PyTorch 参考 argmax token；验证 Loader/Scheduler/KVCache/Executor/
//   Destroy/Engine 在 OCL 后端上的通用链路。
//
// 边界：
//   - 只验证 token 一致性，不比较 logits 数值（数值对齐由 P4.3
//     `ocl_vs_mock_test` 覆盖）；
//   - 使用 recompute_attention 模式（OCL attention 首期不支持 KV cache），
//     每步重算完整序列；
//   - 测试数据针对 Qwen3-0.6B，Runtime 代码不写死模型；
//   - 无 OpenCL 设备时返回 77，由 CTest 记为 Skipped。
//
// 依赖：
//   Runtime 全部模块、OCL 后端、Mock 后端、Graph 静态库。

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/backend/backend.h"
#include "Runtime/backend/backend_registry.h"
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

// 单个后端的运行结果。
struct BackendResult {
  // 加载与执行是否全部成功。
  Bool ok = false;
  // 用例下标 -> 生成 token。
  std::map<I32, I64> tokens;
};

// 检查布尔条件。
Bool Check(Bool condition, const C8* message) {
  if (!condition) {
    std::printf("FAIL: %s\n", message);
    return false;
  }
  return true;
}

// 用指定后端加载模型包并运行用例。
//
// 参数：
//   backend_name：后端注册名（"ocl" 或 "mock"）；
//   package：模型包目录；
//   cases：参考用例列表；
//   only_case：仅运行的下标，负数表示全部；
//   result：输出运行结果，token 按用例下标存放。
// 返回：
//   Load、全部用例执行与 Destroy 均成功返回 true；否则返回 false。
Bool RunBackend(const std::string& backend_name, const std::string& package,
                const std::vector<GoldenCase>& cases, I32 only_case,
                BackendResult* result) {
  if (result == nullptr) {
    return false;
  }
  result->ok = false;
  result->tokens.clear();

  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = package;
  config.runtime.backend = backend_name;
  config.runtime.kv_cache_bytes = 8ull * 1024 * 1024;
  config.runtime.max_num_tokens = 256;
  config.runtime.max_concurrent_requests = 1;
  config.runtime.recompute_attention = true;
  config.runtime.threaded = false;

  std::map<U64, I64> responses;
  engine.SetResponseCallback(
      [&responses](const mobilenet::runtime::Response& response) {
        if (!response.sequences.empty()) {
          responses[response.request_id] =
              response.sequences[0].current_token.token_id;
        }
      });

  std::printf("[%s] loading model package...\n", backend_name.c_str());
  const auto load_start = std::chrono::steady_clock::now();
  mobilenet::Status status = engine.Load(config);
  if (!status.ok()) {
    std::printf("[%s] Load failed: %s\n", backend_name.c_str(),
                status.message().c_str());
    return false;
  }
  const auto load_end = std::chrono::steady_clock::now();
  std::printf("[%s] load done in %.1f ms\n", backend_name.c_str(),
              std::chrono::duration<double, std::milli>(load_end - load_start)
                  .count());

  Bool case_ok = true;
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
      std::printf("[%s] AddRequest failed (%s): %s\n", backend_name.c_str(),
                  item.text, status.message().c_str());
      case_ok = false;
      break;
    }
    mobilenet::runtime::StepStats stats;
    const auto run_start = std::chrono::steady_clock::now();
    status = engine.RunOnce(&stats);
    const auto run_end = std::chrono::steady_clock::now();
    if (!status.ok()) {
      std::printf("[%s] RunOnce failed (%s): %s\n", backend_name.c_str(),
                  item.text, status.message().c_str());
      case_ok = false;
      break;
    }
    const auto it = responses.find(request_id);
    if (it == responses.end()) {
      std::printf("[%s] FAIL: no response for %s\n", backend_name.c_str(),
                  item.text);
      case_ok = false;
      break;
    }
    result->tokens[static_cast<I32>(case_index)] = it->second;
    std::printf("[%s] %-24s token=%lld step=%.1f ms\n", backend_name.c_str(),
                item.text, static_cast<long long>(it->second),
                std::chrono::duration<double, std::milli>(run_end - run_start)
                    .count());
  }

  status = engine.Destroy();
  if (!status.ok()) {
    std::printf("[%s] Destroy failed: %s\n", backend_name.c_str(),
                status.message().c_str());
    case_ok = false;
  }
  result->ok = case_ok;
  return result->ok;
}

// 探测 OCL 后端是否可用。
//
// 返回：
//   后端可创建并 Init 成功返回 true；否则返回 false。
Bool ProbeOclBackend() {
  std::unique_ptr<mobilenet::Backend> backend;
  if (!mobilenet::BackendRegistry::Instance()
           .Create("ocl", mobilenet::BackendConfig{}, &backend)
           .ok() ||
      backend == nullptr) {
    return false;
  }
  const mobilenet::Status status =
      backend->Init(mobilenet::BackendConfig{});
  if (!status.ok()) {
    return false;
  }
  (void)backend->Deinit();
  return true;
}

}  // namespace

// 测试入口。
//
// 参数：
//   argv[1]（可选）：模型包目录，默认 Models/Qwen3/artifacts/qwen3_0_6b；
//   argv[2]（可选）：仅运行指定下标的用例，缺省运行全部；
//   argv[3]（可选）：传 "no-mock" 时跳过 Mock 对比（调试 OCL 时省时）。
// 返回：
//   全部用例通过返回 0；OCL 不可用返回 77；否则返回 1。
I32 main(I32 argc, C8** argv) {
  const std::string package =
      argc > 1 ? argv[1] : "Models/Qwen3/artifacts/qwen3_0_6b";
  const I32 only_case = argc > 2 ? std::atoi(argv[2]) : -1;
  const Bool run_mock =
      !(argc > 3 && std::string(argv[3]) == "no-mock");
  const std::vector<GoldenCase> cases = {
      {"Hello", {9707}, 21806},
      {"The capital of France is", {785, 6722, 315, 9625, 374}, 12095},
      {"1+1=", {16, 10, 16, 28}, 17},
  };

  if (!ProbeOclBackend()) {
    std::printf("SKIP: ocl backend unavailable\n");
    return 77;
  }

  BackendResult ocl_result;
  const Bool ocl_ok =
      RunBackend("ocl", package, cases, only_case, &ocl_result);
  BackendResult mock_result;
  Bool mock_ok = true;
  if (run_mock) {
    mock_ok = RunBackend("mock", package, cases, only_case, &mock_result);
  }

  I32 failures = 0;
  if (!ocl_ok) {
    std::printf("FAIL: ocl backend run did not complete\n");
    ++failures;
  }
  if (!mock_ok) {
    std::printf("FAIL: mock backend run did not complete\n");
    ++failures;
  }
  for (Usize case_index = 0; case_index < cases.size(); ++case_index) {
    if (only_case >= 0 && static_cast<I32>(case_index) != only_case) {
      continue;
    }
    const GoldenCase& item = cases[case_index];
    const auto ocl_it =
        ocl_result.tokens.find(static_cast<I32>(case_index));
    if (ocl_it == ocl_result.tokens.end()) {
      std::printf("FAIL: missing ocl token for %s\n", item.text);
      ++failures;
      continue;
    }
    if (!run_mock) {
      std::printf("%-24s expected=%lld ocl=%lld\n", item.text,
                  static_cast<long long>(item.expected_token),
                  static_cast<long long>(ocl_it->second));
      if (!Check(ocl_it->second == item.expected_token,
                 "ocl token mismatch vs PyTorch reference")) {
        ++failures;
      }
      continue;
    }
    const auto mock_it =
        mock_result.tokens.find(static_cast<I32>(case_index));
    if (mock_it == mock_result.tokens.end()) {
      std::printf("FAIL: missing mock token for %s\n", item.text);
      ++failures;
      continue;
    }
    std::printf("%-24s expected=%lld ocl=%lld mock=%lld\n", item.text,
                static_cast<long long>(item.expected_token),
                static_cast<long long>(ocl_it->second),
                static_cast<long long>(mock_it->second));
    if (!Check(ocl_it->second == item.expected_token,
               "ocl token mismatch vs PyTorch reference")) {
      ++failures;
    }
    if (!Check(ocl_it->second == mock_it->second,
               "ocl token mismatch vs mock backend")) {
      ++failures;
    }
  }

  if (failures != 0) {
    std::printf("%d ocl e2e checks failed\n", failures);
    return 1;
  }
  std::printf("all ocl e2e checks passed\n");
  return 0;
}
