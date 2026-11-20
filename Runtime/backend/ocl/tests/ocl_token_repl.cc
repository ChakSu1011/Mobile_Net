// 模块：Qwen3 OCL Token REPL
//
// 职责：
//   在终端读取一行 token id，调用 Engine + OCL 后端执行 prefill/decode，
//   逐 token 打印自回归生成结果，并保留多轮输入。
//
// 边界：
//   - 只接受 token id，不做 tokenizer/文本编解码；
//   - 默认使用 KV 模型包 qwen3_0_6b_kv（use_kv_cache=true）与 OCL 后端；
//   - 单请求、贪心采样；并发/吞吐测试由 e2e 测试覆盖；
//   - 无 OpenCL 设备时打印错误并返回非 0。
//
// 依赖：
//   Runtime 全部模块、OCL/Mock 后端、Graph 静态库、C++17 标准库。

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Runtime/common/sequence.h"
#include "Runtime/common/status.h"
#include "Runtime/engine/engine.h"
#include "Utils/cpp/types.h"

namespace {

// REPL 单轮生成状态。
struct ReplState {
  // 已生成 token。
  std::vector<I64> generated;

  // 最近一次 Response 状态。
  mobilenet::runtime::RequestStatus status =
      mobilenet::runtime::RequestStatus::kInvalid;

  // 是否已到终态。
  Bool finished = false;
};

// 命令行参数。
struct Options {
  // 模型包目录。
  std::string package = "Models/Qwen3/artifacts/qwen3_0_6b_kv";

  // 后端名称。
  std::string backend = "ocl";

  // 单轮最大生成 token 数。
  I32 max_new_tokens = 16;

  // 单步 token 预算。
  I32 max_num_tokens = 256;

  // KV 预算（MB）。
  U64 kv_mb = 512;

  // 是否走完整历史重算（配合旧 prefill 包）。
  Bool recompute = false;

  // 是否输出每步统计。
  Bool verbose = false;

  // 是否忽略 EOS。
  Bool ignore_eos = false;

  // 是否只打印帮助。
  Bool help = false;
};

// 返回请求状态名称。
const C8* StatusName(mobilenet::runtime::RequestStatus status) {
  using mobilenet::runtime::RequestStatus;
  switch (status) {
    case RequestStatus::kWaiting:
      return "waiting";
    case RequestStatus::kRunning:
      return "running";
    case RequestStatus::kStopped:
      return "stopped";
    case RequestStatus::kRecompute:
      return "recompute";
    case RequestStatus::kSwapped:
      return "swapped";
    case RequestStatus::kIllegal:
      return "illegal";
    case RequestStatus::kChunkedPrefill:
      return "chunked_prefill";
    case RequestStatus::kInvalid:
    default:
      return "invalid";
  }
}

// 打印用法。
void PrintUsage(const C8* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --package <dir>      model package (default: "
         "Models/Qwen3/artifacts/qwen3_0_6b_kv)\n"
      << "  --backend <name>     ocl or mock (default: ocl)\n"
      << "  --max-new <n>        max generated tokens per prompt "
         "(default: 16)\n"
      << "  --max-num-tokens <n> per-step token budget (default: 256)\n"
      << "  --kv-mb <n>          KV cache budget in MiB (default: 512)\n"
      << "  --recompute          use full-history recompute (old prefill "
         "package)\n"
      << "  --ignore-eos         do not stop on EOS token\n"
      << "  --verbose            print per-step statistics\n"
      << "  --help               show this help\n"
      << "\n"
      << "Interactive commands:\n"
      << "  :max <n>             set max generated tokens\n"
      << "  :help                show commands\n"
      << "  :quit / quit / exit  leave\n"
      << "\n"
      << "Token input examples:\n"
      << "  9707\n"
      << "  785 6722 315 9625 374\n"
      << "  9707, 1234, 5678\n";
}

// 解析时长/数量参数。
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

// 解析一行 token id；允许空格、逗号、方括号分隔。
Bool ParseTokens(const std::string& line, std::vector<I64>* tokens,
                 std::string* error) {
  if (tokens == nullptr || error == nullptr) {
    return false;
  }
  tokens->clear();
  Usize index = 0;
  while (index < line.size()) {
    const unsigned char ch =
        static_cast<unsigned char>(line[index]);
    if (std::isspace(ch) || ch == ',' || ch == '[' || ch == ']') {
      ++index;
      continue;
    }
    if (!std::isdigit(ch)) {
      *error = "token id must be a non-negative integer";
      return false;
    }
    I64 value = 0;
    while (index < line.size() &&
           std::isdigit(
               static_cast<unsigned char>(line[index]))) {
      value = value * 10 +
              static_cast<I64>(line[index] - '0');
      if (value > 1000000000LL) {
        *error = "token id is too large";
        return false;
      }
      ++index;
    }
    tokens->push_back(value);
  }
  if (tokens->empty()) {
    *error = "no token ids found";
    return false;
  }
  return true;
}

// 解析命令行。
Bool ParseOptions(I32 argc, C8** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (I32 i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options->help = true;
      return true;
    }
    if (arg == "--verbose") {
      options->verbose = true;
      continue;
    }
    if (arg == "--recompute") {
      options->recompute = true;
      continue;
    }
    if (arg == "--ignore-eos") {
      options->ignore_eos = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "missing value for " << arg << "\n";
      return false;
    }
    const std::string value = argv[++i];
    if (arg == "--package") {
      options->package = value;
    } else if (arg == "--backend") {
      options->backend = value;
    } else if (arg == "--max-new") {
      I64 parsed = 0;
      if (!ParseI64(value, &parsed) || parsed <= 0 ||
          parsed > 4096) {
        std::cerr << "invalid --max-new\n";
        return false;
      }
      options->max_new_tokens = static_cast<I32>(parsed);
    } else if (arg == "--max-num-tokens") {
      I64 parsed = 0;
      if (!ParseI64(value, &parsed) || parsed <= 0 ||
          parsed > 4096) {
        std::cerr << "invalid --max-num-tokens\n";
        return false;
      }
      options->max_num_tokens = static_cast<I32>(parsed);
    } else if (arg == "--kv-mb") {
      I64 parsed = 0;
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
  if (options->backend != "ocl" && options->backend != "mock") {
    std::cerr << "backend must be ocl or mock\n";
    return false;
  }
  return true;
}

}  // namespace

// 测试入口。
//
// 参数：
//   argc/argv：命令行参数，见 PrintUsage。
// 返回：
//   正常退出返回 0；加载/执行失败返回 1。
I32 main(I32 argc, C8** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (options.help) {
    PrintUsage(argv[0]);
    return 0;
  }

  mobilenet::runtime::Engine engine;
  mobilenet::runtime::EngineConfig config;
  config.package_dir = options.package;
  config.runtime.backend = options.backend;
  config.runtime.kv_cache_bytes =
      options.kv_mb * 1024ull * 1024ull;
  config.runtime.max_num_tokens = options.max_num_tokens;
  config.runtime.max_concurrent_requests = 1;
  config.runtime.recompute_attention = options.recompute;
  config.runtime.threaded = false;

  ReplState state;
  engine.SetResponseCallback(
      [&state](const mobilenet::runtime::Response& response) {
        if (!response.sequences.empty()) {
          const mobilenet::runtime::Sequence& sequence =
              response.sequences[0];
          if (sequence.current_token.token_id >= 0) {
            state.generated.push_back(
                sequence.current_token.token_id);
            std::cout << "token "
                      << static_cast<long long>(
                             sequence.current_token.token_id)
                      << std::endl;
          }
        }
        state.status = response.status;
        if (response.status ==
                mobilenet::runtime::RequestStatus::kStopped ||
            response.status ==
                mobilenet::runtime::RequestStatus::kIllegal) {
          state.finished = true;
        }
      });

  std::cout << "loading package " << options.package
            << " on backend " << options.backend << "...\n";
  const mobilenet::Status load_status = engine.Load(config);
  if (!load_status.ok()) {
    std::cerr << "Load failed: " << load_status.message() << "\n";
    return 1;
  }
  std::cout << "ready. enter token ids (e.g. 9707), or :help\n";

  while (true) {
    std::cout << "tokens> " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
      break;
    }
    // 去掉首尾空白。
    const Usize first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      continue;
    }
    const Usize last = line.find_last_not_of(" \t\r\n");
    line = line.substr(first, last - first + 1);
    if (line == "quit" || line == "exit" || line == ":q" ||
        line == ":quit") {
      break;
    }
    if (line == ":help") {
      PrintUsage(argv[0]);
      continue;
    }
    if (line.rfind(":max ", 0) == 0) {
      I64 parsed = 0;
      if (!ParseI64(line.substr(5), &parsed) || parsed <= 0 ||
          parsed > 4096) {
        std::cout << "invalid max; usage: :max 16\n";
        continue;
      }
      options.max_new_tokens = static_cast<I32>(parsed);
      std::cout << "max_new_tokens=" << options.max_new_tokens << "\n";
      continue;
    }
    if (!line.empty() && line[0] == ':') {
      std::cout << "unknown command; :help for list\n";
      continue;
    }

    std::vector<I64> prompt;
    std::string parse_error;
    if (!ParseTokens(line, &prompt, &parse_error)) {
      std::cout << "invalid input: " << parse_error << "\n";
      continue;
    }

    state = ReplState{};
    mobilenet::runtime::Request request;
    request.prompt_token_ids = prompt;
    request.sample_param.is_valid = true;
    request.sample_param.max_tokens = options.max_new_tokens;
    request.sample_param.min_tokens = 1;
    request.sample_param.ignore_eos = options.ignore_eos;

    U64 request_id = 0;
    const mobilenet::Status add_status =
        engine.AddRequest(request, &request_id);
    if (!add_status.ok()) {
      std::cerr << "AddRequest failed: " << add_status.message() << "\n";
      continue;
    }

    const I32 prefill_steps =
        (static_cast<I32>(prompt.size()) +
         options.max_num_tokens - 1) /
        options.max_num_tokens;
    const I32 max_steps =
        options.max_new_tokens + prefill_steps + 64;
    Bool run_ok = true;
    for (I32 step = 0; step < max_steps && !state.finished; ++step) {
      mobilenet::runtime::StepStats stats;
      const mobilenet::Status run_status = engine.RunOnce(&stats);
      if (!run_status.ok()) {
        std::cerr << "RunOnce failed: " << run_status.message()
                  << "\n";
        run_ok = false;
        break;
      }
      if (options.verbose) {
        std::cout << "[step " << step + 1
                  << "] prefill=" << stats.prefill_tokens
                  << " decode=" << stats.decode_tokens
                  << " total=" << stats.total_tokens
                  << " kv_free=" << stats.kv_free_blocks
                  << " preempted=" << stats.preempted_requests
                  << std::endl;
      }
    }

    std::cout << "[done] status=" << StatusName(state.status)
              << " generated=" << state.generated.size()
              << " tokens=";
    for (const I64 token : state.generated) {
      std::cout << static_cast<long long>(token) << " ";
    }
    std::cout << std::endl;
    if (!run_ok) {
      std::cerr << "request failed; engine may be in fatal state\n";
    }
    if (!state.finished) {
      std::cerr << "warning: generation stopped before terminal status\n";
    }
  }

  const mobilenet::Status destroy_status = engine.Destroy();
  if (!destroy_status.ok()) {
    std::cerr << "Destroy failed: " << destroy_status.message() << "\n";
    return 1;
  }
  return 0;
}
