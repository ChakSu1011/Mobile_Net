# Runtime Common 测试结果

## 1. 测试范围

- `json.h/.cc`：严格 JSON 解析；
- `sha256.h/.cc`：流式 SHA-256；
- `file_util.h/.cc`：文件大小、区间读、流式读；
- `sequence.h`/`plan.h`/`status.h`/`runtime_config.h`/`model_config.h`
  的头文件编译。

## 2. 环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | zig clang wrapper，C++17 |
| 包含路径 | `d:/Mobile_Net` |

## 3. 命令与输出

```bash
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/json.cc -o TempSpace/common_json.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/sha256.cc -o TempSpace/common_sha256.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/file_util.cc -o TempSpace/common_file_util.o
```

实际输出：

```text
（无输出）
```

三个命令退出码均为 `0`。

真实 manifest 解析由 Loader smoke 覆盖：

```text
T1 Qwen3 full load: PASSED, exit=0
constants=310, activations=312, weights_bytes=1192099840,
arena_bytes=78317568, IR SHA=4326aaa8...adaf
```

## 4. 结论

- JSON/SHA/文件工具编译通过，无警告；
- Loader smoke 使用本模块解析 Qwen3 manifest、校验 SHA-256 与 IR 哈希，
  结果通过；
- 未覆盖项：JSON 大规模 fuzz、非 ASCII 路径、异常磁盘 IO。
