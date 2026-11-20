// 模块：OCL Paged Attention 端到端自回归生成测试
//
// 职责：
//   用 OCL 后端加载 Qwen3-0.6B KV 模型包，执行 prefill + 多步 decode，
//   逐 token 比较 OCL/Mock 输出与 PyTorch 贪心金标准；验证 Scheduler
//   KV 模式、KVCache、Executor 元数据、paged attention 与采样链路。
//
// 边界：
//   - 只比较生成 token，不比较 logits；
//   - 使用单请求、贪心采样；并发/抢占由其他测试覆盖；
//   - 无 OpenCL 设备时返回 77，由 CTest 记为 Skipped；
//   - 金标准由 PyTorch Qwen3ForCausalLM 贪心生成，写在硬编码表中。
//
// 依赖：
//   Runtime 全部模块、OCL/Mock 后端、Graph 静态库。

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

// 单条金标准用例。
struct GoldenCase {
  // 用例文本。
  const C8* text;
  // prompt token ids。
  std::vector<I64> prompt_ids;
  // 期望生成的 8 个 token。
  std::vector<I64> expected;
};

// 单个后端的生成结果。
struct BackendResult {
  // 全部用例是否成功。
  Bool ok = false;
  // 用例下标 -> 生成的 token 序列。
  std::map<I32, std::vector<I64>> tokens;
};

// 用指定后端运行全部用例。
//
// 参数：
//   backend_name："ocl" 或 "mock"；
//   package：KV 模型包目录；
//   cases：金标准用例；
//   result：输出生成 token。
// 返回：
//   Load/全部用例/Destroy 成功返回 true。
Bool RunBackend(const std::string& backend_name, const std::string& package,
                const std::vector<GoldenCase>& cases,
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
  config.runtime.kv_cache_bytes = 512ull * 1024 * 1024;
  config.runtime.max_num_tokens = 256;
  config.runtime.max_concurrent_requests = 1;
  config.runtime.recompute_attention = false;
  config.runtime.threaded = false;

  std::map<U64, std::vector<I64>> responses;
  engine.SetResponseCallback(
      [&responses](const mobilenet::runtime::Response& response) {
        if (!response.sequences.empty() &&
            response.sequences[0].current_token.token_id >= 0) {
          responses[response.request_id].push_back(
              response.sequences[0].current_token.token_id);
        }
      });

  std::printf("[%s] loading KV model package...\n", backend_name.c_str());
  mobilenet::Status status = engine.Load(config);
  if (!status.ok()) {
    std::printf("[%s] Load failed: %s\n", backend_name.c_str(),
                status.message().c_str());
    return false;
  }

  Bool case_ok = true;
  for (Usize case_index = 0; case_index < cases.size(); ++case_index) {
    const GoldenCase& item = cases[case_index];
    mobilenet::runtime::Request request;
    request.prompt_token_ids = item.prompt_ids;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens =
        static_cast<I32>(item.expected.size());
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

    // 单步最多产生 1 个 token；额外步数用于 chunked prefill。
    const I32 max_steps = static_cast<I32>(item.expected.size()) * 4 + 16;
    Bool done = false;
    for (I32 step = 0; step < max_steps; ++step) {
      mobilenet::runtime::StepStats stats;
      status = engine.RunOnce(&stats);
      if (!status.ok()) {
        std::printf("[%s] RunOnce failed (%s): %s\n", backend_name.c_str(),
                    item.text, status.message().c_str());
        case_ok = false;
        break;
      }
      const auto iterator = responses.find(request_id);
      if (iterator != responses.end() &&
          iterator->second.size() >= item.expected.size()) {
        done = true;
        break;
      }
    }
    if (!done && !case_ok) {
      break;
    }
    const auto iterator = responses.find(request_id);
    if (iterator == responses.end() || !done) {
      std::printf("[%s] FAIL: incomplete generation for %s\n",
                  backend_name.c_str(), item.text);
      case_ok = false;
      break;
    }
    result->tokens[static_cast<I32>(case_index)] =
        std::vector<I64>(iterator->second.begin(),
                         iterator->second.begin() +
                             static_cast<Isize>(item.expected.size()));
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

// 并发运行全部用例，验证 Continuous Batching、chunked prefill 与混合 batch。
//
// 参数：
//   backend_name：后端注册名；
//   package：KV 模型包；
//   cases：金标准用例；
//   result：输出每请求生成 token。
// 返回：
//   全部请求完成且无错误返回 true。
Bool RunConcurrent(const std::string& backend_name, const std::string& package,
                   const std::vector<GoldenCase>& cases,
                   I32 max_num_tokens, I32 max_concurrent,
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
  config.runtime.kv_cache_bytes = 512ull * 1024 * 1024;
  // 故意设置较小预算，迫使 scheduler 切 chunk 并把 prefill/decode 混排。
  config.runtime.max_num_tokens = max_num_tokens;
  config.runtime.max_concurrent_requests = max_concurrent;
  config.runtime.recompute_attention = false;
  config.runtime.threaded = false;

  std::map<U64, std::vector<I64>> responses;
  engine.SetResponseCallback(
      [&responses](const mobilenet::runtime::Response& response) {
        if (!response.sequences.empty() &&
            response.sequences[0].current_token.token_id >= 0) {
          responses[response.request_id].push_back(
              response.sequences[0].current_token.token_id);
        }
      });

  std::printf("[%s] loading KV package for concurrent test...\n",
              backend_name.c_str());
  mobilenet::Status status = engine.Load(config);
  if (!status.ok()) {
    std::printf("[%s] concurrent Load failed: %s\n", backend_name.c_str(),
                status.message().c_str());
    return false;
  }

  std::vector<U64> request_ids;
  for (Usize index = 0; index < cases.size(); ++index) {
    mobilenet::runtime::Request request;
    request.prompt_token_ids = cases[index].prompt_ids;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens =
        static_cast<I32>(cases[index].expected.size());
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = true;
    U64 request_id = 0;
    status = engine.AddRequest(request, &request_id);
    if (!status.ok()) {
      std::printf("[%s] concurrent AddRequest failed (%s): %s\n",
                  backend_name.c_str(), cases[index].text,
                  status.message().c_str());
      (void)engine.Destroy();
      return false;
    }
    request_ids.push_back(request_id);
  }

  Bool done = false;
  const I32 max_steps =
      static_cast<I32>(cases.front().expected.size()) * 16 + 64;
  for (I32 step = 0; step < max_steps; ++step) {
    mobilenet::runtime::StepStats stats;
    status = engine.RunOnce(&stats);
    if (!status.ok()) {
      std::printf("[%s] concurrent RunOnce failed: %s\n",
                  backend_name.c_str(), status.message().c_str());
      (void)engine.Destroy();
      return false;
    }
    Bool all_done = true;
    for (Usize index = 0; index < cases.size(); ++index) {
      const auto iterator = responses.find(request_ids[index]);
      if (iterator == responses.end() ||
          iterator->second.size() < cases[index].expected.size()) {
        all_done = false;
        break;
      }
    }
    if (all_done) {
      done = true;
      break;
    }
  }
  if (!done) {
    std::printf("[%s] concurrent generation did not finish\n",
                backend_name.c_str());
    (void)engine.Destroy();
    return false;
  }
  for (Usize index = 0; index < cases.size(); ++index) {
    const auto iterator = responses.find(request_ids[index]);
    result->tokens[static_cast<I32>(index)] =
        std::vector<I64>(iterator->second.begin(),
                         iterator->second.begin() +
                             static_cast<Isize>(cases[index].expected.size()));
  }
  status = engine.Destroy();
  if (!status.ok()) {
    std::printf("[%s] concurrent Destroy failed: %s\n",
                backend_name.c_str(), status.message().c_str());
    return false;
  }
  result->ok = true;
  return true;
}

// 错峰提交：在已有请求 decode 时加入新 prefill，验证混合 batch。
//
// 参数：
//   backend_name：后端注册名；
//   package：KV 模型包；
//   cases：前 3 条金标准用例；
//   result：输出每请求生成 token。
// 返回：
//   全部请求完成且无错误返回 true。
Bool RunMixed(const std::string& backend_name, const std::string& package,
              const std::vector<GoldenCase>& cases, BackendResult* result) {
  if (result == nullptr || cases.size() < 3) {
    return false;
  }
  result->ok = false;
  result->tokens.clear();

  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = package;
  config.runtime.backend = backend_name;
  config.runtime.kv_cache_bytes = 512ull * 1024 * 1024;
  config.runtime.max_num_tokens = 8;
  config.runtime.max_concurrent_requests = 3;
  config.runtime.recompute_attention = false;
  config.runtime.threaded = false;

  std::map<U64, std::vector<I64>> responses;
  engine.SetResponseCallback(
      [&responses](const mobilenet::runtime::Response& response) {
        if (!response.sequences.empty() &&
            response.sequences[0].current_token.token_id >= 0) {
          responses[response.request_id].push_back(
              response.sequences[0].current_token.token_id);
        }
      });
  mobilenet::Status status = engine.Load(config);
  if (!status.ok()) {
    std::printf("[%s] mixed Load failed: %s\n", backend_name.c_str(),
                status.message().c_str());
    return false;
  }

  std::vector<U64> request_ids;
  for (Usize index = 0; index < 2; ++index) {
    mobilenet::runtime::Request request;
    request.prompt_token_ids = cases[index].prompt_ids;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens =
        static_cast<I32>(cases[index].expected.size());
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = true;
    U64 request_id = 0;
    status = engine.AddRequest(request, &request_id);
    if (!status.ok()) {
      (void)engine.Destroy();
      return false;
    }
    request_ids.push_back(request_id);
  }

  // 先让前两个请求进入 decode，再加入第三个请求。
  for (I32 step = 0; step < 3; ++step) {
    mobilenet::runtime::StepStats stats;
    status = engine.RunOnce(&stats);
    if (!status.ok()) {
      (void)engine.Destroy();
      return false;
    }
  }
  {
    mobilenet::runtime::Request request;
    request.prompt_token_ids = cases[2].prompt_ids;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens =
        static_cast<I32>(cases[2].expected.size());
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = true;
    U64 request_id = 0;
    status = engine.AddRequest(request, &request_id);
    if (!status.ok()) {
      (void)engine.Destroy();
      return false;
    }
    request_ids.push_back(request_id);
  }

  Bool done = false;
  for (I32 step = 0; step < 256; ++step) {
    mobilenet::runtime::StepStats stats;
    status = engine.RunOnce(&stats);
    if (!status.ok()) {
      (void)engine.Destroy();
      return false;
    }
    Bool all_done = true;
    for (Usize index = 0; index < request_ids.size(); ++index) {
      const auto iterator = responses.find(request_ids[index]);
      if (iterator == responses.end() ||
          iterator->second.size() < cases[index].expected.size()) {
        all_done = false;
        break;
      }
    }
    if (all_done) {
      done = true;
      break;
    }
  }
  if (!done) {
    (void)engine.Destroy();
    return false;
  }
  for (Usize index = 0; index < request_ids.size(); ++index) {
    const auto iterator = responses.find(request_ids[index]);
    result->tokens[static_cast<I32>(index)] =
        std::vector<I64>(iterator->second.begin(),
                         iterator->second.begin() +
                             static_cast<Isize>(cases[index].expected.size()));
  }
  status = engine.Destroy();
  result->ok = status.ok();
  return result->ok;
}

// 验证加载期“图 KV 语义与 recompute 配置”一致性检查。
//
// 参数：
//   package：KV 模型包。
// 返回：
//   Load 因模式错配失败返回 true；否则返回 false。
Bool RunMismatchCheck(const std::string& package) {
  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = package;
  config.runtime.backend = "ocl";
  config.runtime.kv_cache_bytes = 64ull * 1024 * 1024;
  config.runtime.max_num_tokens = 16;
  config.runtime.max_concurrent_requests = 1;
  // KV 图必须配 recompute=false；这里故意错配。
  config.runtime.recompute_attention = true;
  config.runtime.threaded = false;
  const mobilenet::Status status = engine.Load(config);
  if (status.ok()) {
    std::printf("FAIL: mismatched attention mode was accepted\n");
    (void)engine.Destroy();
    return false;
  }
  std::printf("mismatch rejected: %s\n", status.message().c_str());
  return true;
}

// 探测 OCL 后端是否可用。
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
//   argv[1]：KV 模型包目录，默认 Models/Qwen3/artifacts/qwen3_0_6b_kv；
//   argv[2]：传 "mock" 时额外运行 Mock 对比（很慢，默认跳过）。
// 返回：
//   全部通过返回 0；OCL 不可用返回 77；否则返回 1。
I32 main(I32 argc, C8** argv) {
  const std::string package =
      argc > 1 ? argv[1] : "Models/Qwen3/artifacts/qwen3_0_6b_kv";
  const Bool run_mock =
      argc > 2 && std::string(argv[2]) == "mock";
  const std::vector<GoldenCase> cases = {
      {"Hello", {9707}, {21806, 0, 358, 2776, 264, 2699, 21815, 911}},
      {"The capital of France is",
       {785, 6722, 315, 9625, 374},
       {12095, 13, 576, 6722, 315, 15344, 374, 21718}},
      {"1+1=", {16, 10, 16, 28},
       {17, 11, 220, 16, 10, 16, 28, 17}},
      {"The weather is nice today",
       {785, 9104, 374, 6419, 3351},
       {11, 323, 279, 9315, 374, 220, 17, 15}},
  };
  // 20-token 生成：每条序列跨越 2 个 KV 块，两个并发请求的物理块表
  // 会出现 [0,2] / [1,3] 这类非连续映射。
  const std::vector<GoldenCase> multi_block_cases = {
      {"Hello (20)",
       {9707},
       {21806, 0, 358, 2776, 264, 2699, 21815, 911, 279, 3491, 13,
        6771, 752, 1430, 311, 3535, 432, 1156, 13, 576}},
      {"The capital of France is (20)",
       {785, 6722, 315, 9625, 374},
       {12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722,
        315, 17689, 374, 24081, 13, 576, 6722, 315, 5616}},
  };

  if (!ProbeOclBackend()) {
    std::printf("SKIP: ocl backend unavailable\n");
    return 77;
  }
  if (argc > 2 && std::string(argv[2]) == "mismatch") {
    return RunMismatchCheck(package) ? 0 : 1;
  }

  BackendResult ocl_result;
  const Bool ocl_ok = RunBackend("ocl", package, cases, &ocl_result);
  BackendResult concurrent_result;
  const Bool concurrent_ok =
      RunConcurrent("ocl", package, cases, 8, 4, &concurrent_result);
  BackendResult multi_block_result;
  const Bool multi_block_ok = RunConcurrent(
      "ocl", package, multi_block_cases, 4, 2, &multi_block_result);
  BackendResult mixed_result;
  const Bool mixed_ok = RunMixed("ocl", package, cases, &mixed_result);

  BackendResult mock_result;
  Bool mock_ok = true;
  if (run_mock) {
    mock_ok = RunBackend("mock", package, cases, &mock_result);
  }

  I32 failures = 0;
  if (!ocl_ok) {
    std::printf("FAIL: ocl backend run did not complete\n");
    ++failures;
  }
  if (!concurrent_ok) {
    std::printf("FAIL: ocl concurrent run did not complete\n");
    ++failures;
  }
  if (!multi_block_ok) {
    std::printf("FAIL: ocl multi-block run did not complete\n");
    ++failures;
  }
  if (!mixed_ok) {
    std::printf("FAIL: ocl mixed prefill/decode run did not complete\n");
    ++failures;
  }
  if (!mock_ok) {
    std::printf("FAIL: mock backend run did not complete\n");
    ++failures;
  }
  for (Usize case_index = 0; case_index < cases.size(); ++case_index) {
    const GoldenCase& item = cases[case_index];
    const auto ocl_it =
        ocl_result.tokens.find(static_cast<I32>(case_index));
    const auto concurrent_it =
        concurrent_result.tokens.find(static_cast<I32>(case_index));
    if (ocl_it == ocl_result.tokens.end()) {
      std::printf("FAIL: missing ocl tokens for %s\n", item.text);
      ++failures;
      continue;
    }
    if (concurrent_it == concurrent_result.tokens.end()) {
      std::printf("FAIL: missing concurrent tokens for %s\n", item.text);
      ++failures;
      continue;
    }
    Bool case_ok = ocl_it->second == item.expected;
    Bool concurrent_case_ok = concurrent_it->second == item.expected;
    std::printf("%-24s expected=", item.text);
    for (const I64 token : item.expected) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    std::printf(" ocl=");
    for (const I64 token : ocl_it->second) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    std::printf(" concurrent=");
    for (const I64 token : concurrent_it->second) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    if (run_mock) {
      const auto mock_it =
          mock_result.tokens.find(static_cast<I32>(case_index));
      if (mock_it == mock_result.tokens.end()) {
        std::printf(" FAIL: missing mock tokens");
        ++failures;
        continue;
      }
      std::printf(" mock=");
      for (const I64 token : mock_it->second) {
        std::printf("%lld,", static_cast<long long>(token));
      }
      if (mock_it->second != item.expected) {
        std::printf(" FAIL: mock mismatch");
        ++failures;
      }
    }
    std::printf("%s%s\n",
                case_ok ? " PASS" : " FAIL:ocl",
                concurrent_case_ok ? "" : " FAIL:concurrent");
    if (!case_ok || !concurrent_case_ok) {
      ++failures;
    }
  }

  // 多块 / 非连续块表用例。
  for (Usize case_index = 0; case_index < multi_block_cases.size();
       ++case_index) {
    const GoldenCase& item = multi_block_cases[case_index];
    const auto iterator =
        multi_block_result.tokens.find(static_cast<I32>(case_index));
    if (iterator == multi_block_result.tokens.end()) {
      std::printf("FAIL: missing multi-block tokens for %s\n", item.text);
      ++failures;
      continue;
    }
    Bool case_ok = iterator->second == item.expected;
    std::printf("%-28s expected=", item.text);
    for (const I64 token : item.expected) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    std::printf(" ocl=");
    for (const I64 token : iterator->second) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    std::printf("%s\n", case_ok ? " PASS" : " FAIL");
    if (!case_ok) {
      ++failures;
    }
  }

  // 错峰提交：已有 decode 时加入新 prefill。
  for (Usize case_index = 0; case_index < 3; ++case_index) {
    const GoldenCase& item = cases[case_index];
    const auto iterator =
        mixed_result.tokens.find(static_cast<I32>(case_index));
    if (iterator == mixed_result.tokens.end()) {
      std::printf("FAIL: missing mixed tokens for %s\n", item.text);
      ++failures;
      continue;
    }
    Bool case_ok = iterator->second == item.expected;
    std::printf("%-24s mixed=", item.text);
    for (const I64 token : iterator->second) {
      std::printf("%lld,", static_cast<long long>(token));
    }
    std::printf("%s\n", case_ok ? " PASS" : " FAIL");
    if (!case_ok) {
      ++failures;
    }
  }

  if (failures != 0) {
    std::printf("%d ocl KV e2e checks failed\n", failures);
    return 1;
  }
  std::printf("all ocl KV e2e checks passed\n");
  return 0;
}
