// 模块：Session 服务命令行入口
//
// 职责：
//   解析 --config，加载配置并启动服务；收到 SIGINT/SIGTERM 后优雅停止。
//
// 边界：
//   - 只支持 --config 与 --help；
//   - 不解析其它命令行参数；
//   - 不处理网络以外的配置来源。

#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

#include "Service/session/server.h"
#include "Service/session/server_config.h"

namespace {

// 停止标记；信号处理函数只写该值。
volatile std::sig_atomic_t g_stop_requested = 0;

// 信号处理。
void HandleSignal(I32 signal_number) {
  (void)signal_number;
  g_stop_requested = 1;
}

// 打印用法。
void PrintUsage(const C8* program) {
  std::printf(
      "usage: %s --config <path>\n"
      "  --config  JSON configuration file\n"
      "  --help    show this help\n",
      program);
}

}  // namespace

// 入口。
//
// 参数：
//   argc/argv：命令行参数。
// 返回：
//   正常停止 0；参数/初始化/启动失败 1。
I32 main(I32 argc, C8** argv) {
  std::string config_path;
  for (I32 index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (argument == "--config") {
      if (index + 1 >= argc) {
        std::printf("error: --config requires a path\n");
        PrintUsage(argv[0]);
        return 1;
      }
      config_path = argv[++index];
      continue;
    }
    std::printf("error: unknown argument: %s\n", argument.c_str());
    PrintUsage(argv[0]);
    return 1;
  }
  if (config_path.empty()) {
    std::printf("error: --config is required\n");
    PrintUsage(argv[0]);
    return 1;
  }

  mobilenet::service::ServerConfig config;
  mobilenet::Status status =
      mobilenet::service::LoadServerConfig(config_path, &config);
  if (!status.ok()) {
    std::printf("config error: %s\n", status.message().c_str());
    return 1;
  }

  std::signal(SIGINT, HandleSignal);
#if defined(SIGTERM)
  std::signal(SIGTERM, HandleSignal);
#endif

  mobilenet::service::Server server;
  status = server.Init(config);
  if (!status.ok()) {
    std::printf("init error: %s\n", status.message().c_str());
    return 1;
  }
  status = server.Start();
  if (!status.ok()) {
    std::printf("start error: %s\n", status.message().c_str());
    (void)server.Stop();
    return 1;
  }

  while (g_stop_requested == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  (void)server.Stop();
  std::printf("server stopped\n");
  return 0;
}
