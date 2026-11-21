// 模块：Qwen3 OCL KV 调度性能测试
//
// 职责：
//   用 OCL 后端加载 Qwen3 KV 模型包，提交 N 个相同长度 prompt 的并发请求，
//   执行 1 次 prefill + 指定次数 decode，分别统计 prefill 与 decode 的
//   墙钟时间、Engine 执行时间、吞吐与首 token 时延。
//
// 边界：
//   - 只测单机单进程性能，不测 CPU/GPU 利用率、功耗或温度；
//   - 默认 4 并发、prompt=256、decode=8 次；prefill 会按 max_num_tokens
//     切 chunk，因此 prefill 步数可能大于 1；
//   - 不注册 CTest（性能数据依赖机器状态）；
//   - 首次运行前先做一次小 prompt 预热，避免把 kernel 编译计入测量。
//
// 依赖：
//   Runtime 全部模块、OCL 后端、Graph 静态库、C++17 标准库。

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"
#include "Runtime/engine/engine.h"
#include "Utils/cpp/types.h"

namespace {

// 命令行参数。
struct Options {
  // 模型包目录。
  std::string package = "Models/Qwen3/artifacts/qwen3_0_6b_kv";

  // 并发请求数。
  I32 concurrency = 4;

  // 每条 prompt 的 token 数。
  I32 prompt_len = 256;

  // decode 步数（不含 prefill 产生的首 token）。
  I32 decode_steps = 8;

  // 单步 token 预算；默认等于静态图上限 256。
  I32 max_num_tokens = 256;

  // KV 预算（MiB）。
  U64 kv_mb = 512;

  // 是否跳过预热。
  Bool no_warmup = false;

  // 是否打印每一步。
  Bool verbose = true;
};

// 单请求时间信息。
struct RequestTiming {
  // 是否已收到首 token。
  Bool first_token = false;

  // 首 token 时延（ms，从测量开始算）。
  FP64 ttft_ms = 0.0;

  // 已生成 token 数。
  I32 generated = 0;

  // 是否已到终态。
  Bool finished = false;
};

// 单步采样。
struct StepSample {
  // 本步 prefill token 数。
  I32 prefill_tokens = 0;

  // 本步 decode token 数。
  I32 decode_tokens = 0;

  // 墙钟时间（ms）。
  FP64 wall_ms = 0.0;

  // Engine 执行时间（ms，StepStats.execute_ns）。
  FP64 execute_ms = 0.0;
};

// 打印用法。
void PrintUsage(const C8* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --package <dir>       model package (default: "
         "Models/Qwen3/artifacts/qwen3_0_6b_kv)\n"
      << "  --concurrency <n>     concurrent requests (default: 4)\n"
      << "  --prompt-len <n>      prompt tokens per request (default: 256)\n"
      << "  --decode-steps <n>    decode steps after prefill (default: 8)\n"
      << "  --max-num-tokens <n>  per-step token budget (default: 256)\n"
      << "  --kv-mb <n>           KV budget in MiB (default: 512)\n"
      << "  --no-warmup           skip warmup\n"
      << "  --quiet               only print summary\n"
      << "  --help                show this help\n";
}

// 解析整数参数。
Bool ParseI64(const std::string& text, I64* value) {
  if (value == nullptr || text.empty()) {
    return false;
  }
  C8* end = nullptr;
  const long long parsed = std::strtoll(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0') {
    return false;
  }
  *value = static_cast<I64>(parsed);
  return true;
}

// 解析命令行。
Bool ParseOptions(I32 argc, C8** argv, Options* options, Bool* help) {
  if (options == nullptr || help == nullptr) {
    return false;
  }
  *help = false;
  for (I32 i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      *help = true;
      return true;
    }
    if (arg == "--no-warmup") {
      options->no_warmup = true;
      continue;
    }
    if (arg == "--quiet") {
      options->verbose = false;
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << arg << "\n";
      return false;
    }
    const std::string value = argv[++i];
    I64 parsed = 0;
    if (arg == "--package") {
      options->package = value;
    } else if (arg == "--concurrency") {
      if (!ParseI64(value, &parsed) || parsed <= 0 || parsed > 64) {
        std::cerr << "invalid --concurrency\n";
        return false;
      }
      options->concurrency = static_cast<I32>(parsed);
    } else if (arg == "--prompt-len") {
      if (!ParseI64(value, &parsed) || parsed <= 0 || parsed > 4096) {
        std::cerr << "invalid --prompt-len\n";
        return false;
      }
      options->prompt_len = static_cast<I32>(parsed);
    } else if (arg == "--decode-steps") {
      if (!ParseI64(value, &parsed) || parsed < 0 || parsed > 4096) {
        std::cerr << "invalid --decode-steps\n";
        return false;
      }
      options->decode_steps = static_cast<I32>(parsed);
    } else if (arg == "--max-num-tokens") {
      if (!ParseI64(value, &parsed) || parsed <= 0 || parsed > 4096) {
        std::cerr << "invalid --max-num-tokens\n";
        return false;
      }
      options->max_num_tokens = static_cast<I32>(parsed);
    } else if (arg == "--kv-mb") {
      if (!ParseI64(value, &parsed) || parsed <= 0) {
        std::cerr << "invalid --kv-mb\n";
        return false;
      }
      options->kv_mb = static_cast<U64>(parsed);
    } else {
      std::cerr << "unknown option: " << arg << "\n";
      return false;
    }
  }
  return true;
}

// 构造长度为 length 的确定性 prompt。
std::vector<I64> MakePrompt(I32 length) {
  std::vector<I64> prompt;
  prompt.reserve(static_cast<Usize>(length));
  for (I32 i = 0; i < length; ++i) {
    // 固定 token，避免随机性；9707 在 Qwen3 词表内。
    prompt.push_back(9707);
  }
  return prompt;
}

// 打印单步采样。
void PrintStep(I32 step, const StepSample& sample) {
  const C8* kind = "mixed";
  if (sample.prefill_tokens > 0 && sample.decode_tokens == 0) {
    kind = "prefill";
  } else if (sample.decode_tokens > 0 && sample.prefill_tokens == 0) {
    kind = "decode";
  }
  std::cout << "step " << std::setw(2) << step << " " << std::setw(7)
            << kind << " prefill=" << std::setw(4)
            << sample.prefill_tokens << " decode=" << std::setw(3)
            << sample.decode_tokens << " wall_ms=" << std::fixed
            << std::setprecision(2) << sample.wall_ms
            << " execute_ms=" << sample.execute_ms << "\n";
}

// 汇总并打印性能结果。
void PrintSummary(const std::vector<StepSample>& samples,
                  const std::map<U64, RequestTiming>& timings,
                  const std::vector<U64>& request_ids,
                  FP64 total_wall_ms) {
  FP64 prefill_wall = 0.0;
  FP64 prefill_execute = 0.0;
  FP64 decode_wall = 0.0;
  FP64 decode_execute = 0.0;
  I32 prefill_tokens = 0;
  I32 decode_tokens = 0;
  I32 prefill_steps = 0;
  I32 decode_steps = 0;

  for (const StepSample& sample : samples) {
    if (sample.prefill_tokens > 0) {
      prefill_wall += sample.wall_ms;
      prefill_execute += sample.execute_ms;
      prefill_tokens += sample.prefill_tokens;
      ++prefill_steps;
    }
    if (sample.decode_tokens > 0) {
      decode_wall += sample.wall_ms;
      decode_execute += sample.execute_ms;
      decode_tokens += sample.decode_tokens;
      ++decode_steps;
    }
  }

  std::cout << "\n=== summary ===\n";
  if (prefill_steps > 0) {
    const FP64 wall_s = prefill_wall / 1000.0;
    const FP64 exec_s = prefill_execute / 1000.0;
    std::cout << "[prefill] steps=" << prefill_steps
              << " tokens=" << prefill_tokens
              << " total_wall_ms=" << std::fixed << std::setprecision(2)
              << prefill_wall
              << " avg_step_ms=" << prefill_wall / prefill_steps
              << " wall_tok_per_s="
              << (wall_s > 0.0 ? prefill_tokens / wall_s : 0.0)
              << " exec_tok_per_s="
              << (exec_s > 0.0 ? prefill_tokens / exec_s : 0.0)
              << "\n";
  }
  if (decode_steps > 0) {
    const FP64 wall_s = decode_wall / 1000.0;
    const FP64 exec_s = decode_execute / 1000.0;
    std::cout << "[decode]  steps=" << decode_steps
              << " tokens=" << decode_tokens
              << " total_wall_ms=" << decode_wall
              << " avg_step_ms=" << decode_wall / decode_steps
              << " per_token_ms="
              << (decode_tokens > 0
                      ? decode_wall / decode_tokens
                      : 0.0)
              << " wall_tok_per_s="
              << (wall_s > 0.0 ? decode_tokens / wall_s : 0.0)
              << " exec_tok_per_s="
              << (exec_s > 0.0 ? decode_tokens / exec_s : 0.0)
              << "\n";
  }
  FP64 ttft_sum = 0.0;
  I32 ttft_count = 0;
  for (Usize index = 0; index < request_ids.size(); ++index) {
    const auto iterator = timings.find(request_ids[index]);
    if (iterator == timings.end() || !iterator->second.first_token) {
      continue;
    }
    std::cout << "[ttft]    request " << index + 1
              << " (id=" << request_ids[index]
              << ") = " << iterator->second.ttft_ms << " ms\n";
    ttft_sum += iterator->second.ttft_ms;
    ++ttft_count;
  }
  if (ttft_count > 0) {
    std::cout << "[ttft]    avg=" << ttft_sum / ttft_count << " ms\n";
  }
  std::cout << "[total]   wall_ms=" << total_wall_ms << "\n";
}

}  // namespace

// 测试入口。
//
// 参数：
//   argc/argv：命令行参数，见 PrintUsage。
// 返回：
//   成功返回 0；参数/加载/执行失败返回 1。
I32 main(I32 argc, C8** argv) {
  Options options;
  Bool help = false;
  if (!ParseOptions(argc, argv, &options, &help)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (help) {
    PrintUsage(argv[0]);
    return 0;
  }

  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = options.package;
  config.runtime.backend = "ocl";
  config.runtime.kv_cache_bytes = options.kv_mb * 1024ull * 1024ull;
  config.runtime.max_num_tokens = options.max_num_tokens;
  config.runtime.max_concurrent_requests = options.concurrency;
  config.runtime.recompute_attention = false;
  config.runtime.threaded = false;

  std::map<U64, RequestTiming> timings;
  std::chrono::steady_clock::time_point measure_start;
  Bool measuring = false;
  engine.SetResponseCallback(
      [&timings, &measure_start, &measuring](
          const mobilenet::runtime::Response& response) {
        RequestTiming& timing = timings[response.request_id];
        if (!response.sequences.empty() &&
            response.sequences[0].current_token.token_id >= 0) {
          if (!timing.first_token && measuring) {
            timing.first_token = true;
            timing.ttft_ms =
                std::chrono::duration<FP64, std::milli>(
                    std::chrono::steady_clock::now() - measure_start)
                    .count();
          }
          ++timing.generated;
        }
        if (response.status ==
                mobilenet::runtime::RequestStatus::kStopped ||
            response.status ==
                mobilenet::runtime::RequestStatus::kIllegal) {
          timing.finished = true;
        }
      });

  const auto load_start = std::chrono::steady_clock::now();
  const mobilenet::Status load_status = engine.Load(config);
  const auto load_end = std::chrono::steady_clock::now();
  if (!load_status.ok()) {
    std::cerr << "Load failed: " << load_status.message() << "\n";
    return 1;
  }
  std::cout << "[load]    " << std::fixed << std::setprecision(1)
            << std::chrono::duration<FP64, std::milli>(load_end - load_start)
                   .count()
            << " ms\n";

  if (!options.no_warmup) {
    std::cout << "[warmup]  running...\n";
    mobilenet::runtime::Request warm_request;
    warm_request.prompt_token_ids = {9707, 9707, 9707, 9707};
    warm_request.sample_param.is_valid = true;
    warm_request.sample_param.max_tokens = 2;
    warm_request.sample_param.min_tokens = 1;
    warm_request.sample_param.ignore_eos = true;
    U64 warm_id = 0;
    const mobilenet::Status warm_add =
        engine.AddRequest(warm_request, &warm_id);
    if (!warm_add.ok()) {
      std::cerr << "warmup AddRequest failed: " << warm_add.message()
                << "\n";
      (void)engine.Destroy();
      return 1;
    }
    Bool warm_done = false;
    for (I32 step = 0; step < 8; ++step) {
      mobilenet::runtime::StepStats stats;
      const mobilenet::Status warm_run = engine.RunOnce(&stats);
      if (!warm_run.ok()) {
        std::cerr << "warmup RunOnce failed: " << warm_run.message()
                  << "\n";
        (void)engine.Destroy();
        return 1;
      }
      if (timings[warm_id].finished) {
        warm_done = true;
        break;
      }
    }
    if (!warm_done) {
      std::cerr << "warmup did not finish\n";
      (void)engine.Destroy();
      return 1;
    }
    std::cout << "[warmup]  done\n";
  }

  timings.clear();
  const std::vector<I64> prompt = MakePrompt(options.prompt_len);
  measure_start = std::chrono::steady_clock::now();
  measuring = true;

  std::vector<U64> request_ids;
  request_ids.reserve(static_cast<Usize>(options.concurrency));
  for (I32 i = 0; i < options.concurrency; ++i) {
    mobilenet::runtime::Request request;
    request.prompt_token_ids = prompt;
    request.sample_param.is_valid = true;
    // prefill 产生首 token，decode_steps 再产生对应数量的 token。
    request.sample_param.max_tokens = options.decode_steps + 1;
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = true;
    U64 request_id = 0;
    const mobilenet::Status status =
        engine.AddRequest(request, &request_id);
    if (!status.ok()) {
      std::cerr << "AddRequest failed: " << status.message() << "\n";
      (void)engine.Destroy();
      return 1;
    }
    request_ids.push_back(request_id);
  }

  const I32 prefill_plan_steps =
      (options.concurrency * options.prompt_len +
       options.max_num_tokens - 1) /
      options.max_num_tokens;
  const I32 max_steps =
      prefill_plan_steps + options.decode_steps + 64;
  std::vector<StepSample> samples;
  samples.reserve(static_cast<Usize>(max_steps));
  Bool all_done = false;
  for (I32 step = 0; step < max_steps; ++step) {
    const auto step_start = std::chrono::steady_clock::now();
    mobilenet::runtime::StepStats stats;
    const mobilenet::Status status = engine.RunOnce(&stats);
    const auto step_end = std::chrono::steady_clock::now();
    if (!status.ok()) {
      std::cerr << "RunOnce failed: " << status.message() << "\n";
      (void)engine.Destroy();
      return 1;
    }
    StepSample sample;
    sample.prefill_tokens = stats.prefill_tokens;
    sample.decode_tokens = stats.decode_tokens;
    sample.wall_ms =
        std::chrono::duration<FP64, std::milli>(step_end - step_start)
            .count();
    sample.execute_ms =
        static_cast<FP64>(stats.execute_ns) / 1000000.0;
    samples.push_back(sample);
    if (options.verbose) {
      PrintStep(step + 1, sample);
    }
    all_done = true;
    for (U64 request_id : request_ids) {
      if (!timings[request_id].finished) {
        all_done = false;
        break;
      }
    }
    if (all_done) {
      break;
    }
  }
  const auto measure_end = std::chrono::steady_clock::now();
  measuring = false;
  const FP64 total_wall_ms =
      std::chrono::duration<FP64, std::milli>(measure_end - measure_start)
          .count();

  if (!all_done) {
    std::cerr << "warning: not all requests finished within max steps\n";
  }
  PrintSummary(samples, timings, request_ids, total_wall_ms);

  const mobilenet::Status destroy_status = engine.Destroy();
  if (!destroy_status.ok()) {
    std::cerr << "Destroy failed: " << destroy_status.message() << "\n";
    return 1;
  }
  return 0;
}
