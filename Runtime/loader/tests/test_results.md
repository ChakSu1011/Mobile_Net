# Runtime Loader 测试结果

## 1. 测试范围

本文件记录 Runtime Loader（`Runtime/loader/`）与基础工具
（`Runtime/common/{json,sha256,file_util}`）的实际编译与运行结果：

- 全部新源文件 `-Wall -Wextra` 编译；
- 真实 Qwen3-0.6B 包全量加载（含 manifest/权重/IR SHA-256 校验）；
- metadata-only 包加载；
- 头部、JSON、IR 哈希、权重尺寸、OpEnv 回滚、后端注册、目录缺失等负向用例；
- C++ `DumpIr` 与 Python `dump_ir` 哈希交叉验证。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| Shell | Git Bash（POSIX sh） |
| 编译器 | `python -m ziglang c++`（clang 22.1.8，target `x86_64-windows-gnu`） |
| 语言标准 | C++17 |
| 优化级别 | `-O2` |
| 警告选项 | `-Wall -Wextra` |
| 包含路径 | `d:/Mobile_Net` |
| Graph 库 | `Graph/build/libgraph.a` |
| Backend 库 | `Runtime/backend/build/libruntime_backend.a`（whole-archive） |
| 模型包 | `Models/Qwen3/artifacts/qwen3_0_6b/`（model.mgf 385,587 B；weights.bin 1,192,099,840 B） |
| 构建/日志目录 | `TempSpace/loader_final_20261120_110000/`（不提交 Git） |

## 3. 编译与链接

### 3.1 编译命令

```bash
cd d:/Mobile_Net
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/json.cc       -o <out>/obj/Runtime_common_json.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/sha256.cc     -o <out>/obj/Runtime_common_sha256.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/file_util.cc  -o <out>/obj/Runtime_common_file_util.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/loader/manifest.cc   -o <out>/obj/Runtime_loader_manifest.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/loader/ir_dump.cc    -o <out>/obj/Runtime_loader_ir_dump.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/loader/loader.cc     -o <out>/obj/Runtime_loader_loader.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/loader/tests/loader_smoke.cc \
  -o <out>/obj/Runtime_loader_tests_loader_smoke.o
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I "d:/Mobile_Net" \
  -c Runtime/common/resource_stack.cc \
  -o <out>/obj/Runtime_common_resource_stack.o
```

### 3.2 编译实际输出

```text
（每个文件均无任何输出）
exit=0
```

### 3.3 链接命令

```bash
python -m ziglang c++ -O2 -o <out>/loader_smoke.exe <out>/obj/*.o \
  -Wl,--whole-archive Runtime/backend/build/libruntime_backend.a \
  -Wl,--no-whole-archive Graph/build/libgraph.a
```

### 3.4 链接实际输出

```text
link_exit=0
-rwxr-xr-x 1 18626 197609 1239552 Nov 20 08:55 <out>/loader_smoke.exe
```

说明：`runtime_backend` 使用 whole-archive，保证 `register_mock.cc` 的静态
注册器不被链接器裁剪；否则 `BackendRegistry::Create("mock")` 返回
`kUnsupportedBackend`。

### 3.5 最终复核构建

代码审查后做了三处修正：“过短文件返回 `kInvalidModelFile`”、
`FileReader::Open` 状态清理、权重“单条目 8 MB 分块读取 + 增量 SHA-256”。
修正完成后在 `TempSpace/loader_final_20261120_110000/` 重新执行了第 3.1
与 3.3 节的完整编译/链接命令。

实际输出：

```text
COMPILE_ALL_OK=1
LINK_EXIT=0
-rwxr-xr-x 1 18626 197609 1240064 Nov 20 08:59 <out>/loader_smoke.exe
```

第 4–6 节的全部结果均由该最终二进制重新执行得到。

补充：代码规范复核时将 `json.cc` 中 `strtoll/strtod` 的原始类型局部变量
改为 `I64/FP64` 别名，并用相同编译命令重新编译 `json.cc`、重新链接；
T1、T2、T4、T7、T9、T12 复核结果与上文一致（成功用例 PASSED，负向用例
仍以预期错误码退出）。

## 4. 正向测试

### 4.1 T1：Qwen3 完整包全量加载（开启权重 SHA-256）

命令：

```bash
<out>/loader_smoke.exe Models/Qwen3/artifacts/qwen3_0_6b
```

实际输出：

```text
package: Models/Qwen3/artifacts/qwen3_0_6b
mode: full-load, verify_weights: on, backend: mock
header: version=1 flags=1 manifest_offset=112 manifest_length=385475 has_weights=1
header: weights_offset=385587 weights_length=1192099840
manifest: model=qwen3_0_6b adapter=qwen3 format=MobileNet.ModelPackage v1
manifest: ir_sha256=4326aaa8c2505f02... ir_line_count=633
manifest: stats tensors=624 ops=312 constants=310 outputs=1 activations=312 constant_bytes=1192099840
manifest: memory alignment=256 arena=78317568 peak=78315520 entries=312
graph: name=qwen3_0_6b inputs=2 constants=310 orderly_ops=315 outputs=1
inputs: input.input_ids input.position_ids
outputs: output.tokens
sample op: sample dst=sample.out
op[0] input.input_ids type=input
op[1] embedding type=gather
op[2] layer.0.attn_norm type=rms_norm
op[314] output.tokens type=output (last)
weights: result.weights_bytes=1192099840 constants_sum=1192099840 constants=310
constants: first=embedding.weight bytes=311164928 bound=1, last=final_norm.weight bytes=2048 bound=1
arena: result.arena_bytes=78317568 plan_arena=78317568 max_entry_end=78317568 entries=312
io: result.io_bytes=4096
workspace: buffers=0 bytes=0 bound=1
model_config: layers=28 heads=16 kv_heads=8 head_dim=128 hidden=1024 intermediate=3072 vocab=151936 context=40960
model_config: dtype=float16 eps=1e-06 rope_theta=1e+06 max_num_tokens=256 max_batch=1 pad=151643 bos=151643 stop_ids=2
binding check: OK
release: status=OK
loader smoke PASSED
exit=0
```

结论：

- 头部、manifest SHA-256、weights 全文件 SHA-256、310 个权重逐项
  SHA-256、C++ `DumpIr` SHA-256 全部通过；
- `weights_bytes` 与常量 Storage 字节和一致（1,192,099,840）；
- arena 高水位、计划条目数、最大条目结束偏移一致（78,317,568）；
- Input/常量/计算 op 的 src/dst 全部绑定 `data_ptr` 且
  `device_type=kDevice`；
- ResourceStack 逆序释放成功。

### 4.2 T2：metadata-only 包（缺少 weights.bin）

测试包：`TempSpace/loader_negative_20261120_094500/no_weights/`，仅包含
`model.mgf`。

命令：

```bash
<out>/loader_smoke.exe TempSpace/loader_negative_20261120_094500/no_weights \
  --metadata-only --no-verify
```

实际输出（节选关键行）：

```text
mode: metadata-only, verify_weights: off, backend: mock
manifest: ir_sha256=4326aaa8c2505f02... ir_line_count=633
graph: name=qwen3_0_6b inputs=2 constants=310 orderly_ops=315 outputs=1
weights: result.weights_bytes=0 constants_sum=1192099840 constants=310
arena: result.arena_bytes=78317568 plan_arena=78317568 max_entry_end=78317568 entries=312
io: result.io_bytes=4096
model_config: dtype=float16 eps=1e-06 rope_theta=1e+06 max_num_tokens=256 max_batch=1 pad=151643 bos=151643 stop_ids=2
binding check: OK
release: status=OK
loader smoke PASSED
exit=0
```

结论：metadata-only 允许常量绑定 64 字节占位缓冲；`weights_bytes=0` 如实
反映未常驻权重。该结果不可用于推理（见风险）。

### 4.3 T10：Python `dump_ir` 交叉验证

命令（节选）：

```bash
PYTHONPATH="Graph/build/python;." python -I - <<'PY'
from ToolChain.serializer.loader import load_package, rebuild_graph
pkg = load_package(Path('Models/Qwen3/artifacts/qwen3_0_6b'), verify=False)
g = rebuild_graph(pkg, verify_ir=True)
print('python dump_ir sha256:', hashlib.sha256(g.dump_ir().encode()).hexdigest())
print('python dump_ir lines:', len(g.dump_ir().splitlines()))
PY
```

实际输出：

```text
python dump_ir sha256: 4326aaa8c2505f020184f721df6a2ac947fc2d1d4959edd5d78ddda53f95adaf
python dump_ir lines: 633
```

结论：Python `dump_ir`、manifest `graph.ir_sha256`、C++ `DumpIr` 三者哈希
一致（C++ 端在 T1 加载中通过比对），证明格式逐字节一致（G1）。

### 4.4 T11：全量加载耗时

命令：

```bash
time <out>/loader_smoke.exe Models/Qwen3/artifacts/qwen3_0_6b > /dev/null
```

实际输出：

```text
real    0m9.328s
user    0m0.000s
sys     0m0.015s
exit=0
```

## 5. 负向测试

以下测试包由 `model.mgf` 副本构造，放在 `TempSpace`（不提交 Git）。
`bad_ir`、`bad_op` 通过重新编码 canonical JSON 并重算头部 manifest SHA
构造；`bad_op` 另将原始 `weights.bin` 硬链接到包目录。

### 5.1 T3：坏 magic

```text
package: TempSpace/loader_negative_20261120_094500/bad_magic
mode: full-load, verify_weights: off, backend: mock
FAIL: read header: bad model file magic
exit=1
```

### 5.2 T4：非法 JSON（尾部 `}` 改为 `]`）

```text
package: TempSpace/loader_negative_20261120_094500/bad_json
mode: full-load, verify_weights: off, backend: mock
header: version=1 flags=1 manifest_offset=112 manifest_length=385475 has_weights=1
header: weights_offset=385587 weights_length=1192099840
FAIL: parse manifest: manifest: JSON parse failed: JSON error at offset 385474: expected ',' or '}' in object
exit=1
```

结论：错误信息包含字节偏移，可用于定位损坏的 manifest。

### 5.3 T5：IR 哈希不匹配

```text
package: TempSpace/loader_negative_20261120_094500/bad_ir
mode: metadata-only, verify_weights: off, backend: mock
manifest: ir_sha256=0000000000000000... ir_line_count=633
manifest: stats tensors=624 ops=312 constants=310 outputs=1 activations=312 constant_bytes=1192099840
manifest: memory alignment=256 arena=78317568 peak=78315520 entries=312
FAIL: Load: [INVALID_MODEL_FILE] model package: rebuilt graph IR SHA-256 mismatch: expected 0000000000000000000000000000000000000000000000000000000000000000, got 4326aaa8c2505f020184f721df6a2ac947fc2d1d4959edd5d78ddda53f95adaf
rollback stack: size cleared, status=
exit=1
```

结论：图重建与 IR 哈希校验在绑定 data_ptr 之前执行，错误码符合契约。

### 5.4 T6：权重文件尺寸不匹配

测试包中 `weights.bin` 为 100 字节。

```text
package: TempSpace/loader_negative_20261120_094500/bad_weights
mode: full-load, verify_weights: off, backend: mock
manifest: ir_sha256=4326aaa8c2505f02... ir_line_count=633
manifest: stats tensors=624 ops=312 constants=310 outputs=1 activations=312 constant_bytes=1192099840
manifest: memory alignment=256 arena=78317568 peak=78315520 entries=312
FAIL: Load: [INVALID_MODEL_FILE] model package: weights file size mismatch: header=1192099840, manifest=1192099840, file=100
rollback stack: size cleared, status=
exit=1
```

### 5.5 T7：OpEnv 创建失败后的完整回滚

测试包将 `embedding`（gather）attr `axis` 从 0 改为 1，并按新图重算
`ir_sha256` 与 manifest SHA，使 IR 校验通过、在绑定权重/arena/IO 缓冲
之后由 Gather spec 拒绝。

```text
package: TempSpace/loader_negative_20261120_094500/bad_op
mode: full-load, verify_weights: off, backend: mock
manifest: ir_sha256=95f78ed8917398f7... ir_line_count=633
manifest: stats tensors=624 ops=312 constants=310 outputs=1 activations=312 constant_bytes=1192099840
manifest: memory alignment=256 arena=78317568 peak=78315520 entries=312
FAIL: Load: [UNSUPPORTED_OP] op=embedding: op=embedding: gather only supports axis=0
rollback stack: size cleared, status=
exit=1
```

结论：失败发生在权重与 arena/IO 缓冲已创建、部分 OpEnv 已创建之后；
`rollback stack` 状态为空消息（kOk），说明 Graph 析构清 OpEnv、缓冲释放、
后端 Deinit 全部成功，无泄漏错误。

### 5.6 T8：未注册后端

```text
package: Models/Qwen3/artifacts/qwen3_0_6b
mode: full-load, verify_weights: off, backend: missing
manifest: ir_sha256=4326aaa8c2505f02... ir_line_count=633
FAIL: Load: [UNSUPPORTED_BACKEND] BackendRegistry::Create: unknown backend: missing
rollback stack: size cleared, status=
exit=1
```

### 5.7 T9：包目录不存在

```text
package: TempSpace/loader_negative_20261120_094500/does_not_exist
mode: full-load, verify_weights: off, backend: mock
FAIL: read header: file open failed: TempSpace/loader_negative_20261120_094500/does_not_exist/model.mgf
exit=1
```

### 5.8 T12：文件短于 112 字节头

测试包 `truncated_header/model.mgf` 仅 100 字节。

```text
package: TempSpace/loader_negative_20261120_094500/truncated_header
mode: full-load, verify_weights: off, backend: mock
FAIL: read header: model file is too short for header: TempSpace/loader_negative_20261120_094500/truncated_header/model.mgf
exit=1
```

## 6. 结论

| 用例 | 期望 | 实际 | 结果 |
| --- | --- | --- | --- |
| 编译（8 个文件，-Wall -Wextra） | 0 警告 0 错误 | 无输出，exit=0 | 通过 |
| 链接（whole-archive） | 成功 | exit=0 | 通过 |
| T1 全量加载 | 成功且校验全部通过 | exit=0，统计一致 | 通过 |
| T2 metadata-only | 成功，weights_bytes=0 | exit=0 | 通过 |
| T3 坏 magic | kInvalidModelFile | “bad model file magic”，exit=1 | 通过 |
| T4 非法 JSON | 带偏移错误 | offset 385474，exit=1 | 通过 |
| T5 IR 哈希不匹配 | kInvalidModelFile | 报错并回滚，exit=1 | 通过 |
| T6 权重尺寸不匹配 | kInvalidModelFile | 报错并回滚，exit=1 | 通过 |
| T7 OpEnv 失败回滚 | 非零错误且回滚成功 | kUnsupportedOp，release 空状态 | 通过 |
| T8 未知后端 | kUnsupportedBackend | 报错，exit=1 | 通过 |
| T9 目录缺失 | kIoError | 报错，exit=1 | 通过 |
| T12 过短文件 | kInvalidModelFile | 报错，exit=1 | 通过 |
| T10 Python IR 交叉 | 哈希一致 | 4326aaa8…adaf / 633 行 | 通过 |
| T11 全量耗时 | 10 s 量级 | real 9.328 s | 通过 |

## 7. 未覆盖项与风险

1. **未执行 Forward**：本测试只验证加载期编排；权重数值、布局与算子正确性
   由 Mock e2e/算子测试覆盖。
2. **workspace 路径未实测**：Qwen3 当前没有 `need_workspace` 的 op，
   `GraphEnv::SharedStorages()` 为空；分配/回填代码仅通过编译审查。
3. **精度转换未触发**：包内权重已是 FP16，`BF16/FP32→FP16` 的
   `ConvertParam` 路径未被真实数据覆盖。
4. **仅 Mock 后端**：OCL/Vulkan 后端的能力声明与对齐/缓冲上限行为未验证。
5. **metadata-only 不可推理**：常量绑定 64 字节占位缓冲，执行 Kernel 会
   越界读取；该模式只用于加载期冒烟。
6. **Destroy 顺序依赖**：正常销毁依赖 Destroyer 先清 OpEnv、再清
   ResourceStack、最后 `Executor::Deinit`；顺序错误会导致悬空指针。
7. **非 ASCII 路径**：`std::ifstream` 按平台本地编码解释路径；本机测试
   路径均为 ASCII，未验证中文/Unicode 路径。
8. **性能**：SHA-256 为纯 C++ 实现；1.19 GB 全量 + 逐项校验约 8.7 s，
   更大模型需评估是否引入硬件加速。
9. **后端生命周期边界**：Loader 的缓冲释放闭包保存 Backend 裸指针。正常
   Destroy 按“ClearOpEnv → ResourceStack.Clear → Executor::Deinit”顺序安全；
   但 Engine 在 `IExecutor::Init` 失败时先由 Executor 释放 Backend，再调用
   Destroyer 清栈（`backend=null`），此时缓冲闭包可能访问已释放的后端。
   该问题属于 Engine/Executor/Destroyer 的跨模块生命周期契约，建议
   Executor::Init 失败时不要把 Backend 提前释放，或 Destroyer 在该路径
   跳过 `DestroyBuffer` 直接析构 Buffer。Loader 侧已按任务契约使用
   `Backend::DestroyBuffer`。

未完成的公共头变更需求：无。实现只使用既有
`loader.h/backend.h/backend_registry.h/op_env_factory.h/resource_stack.h`
与 Graph 公共接口，未修改任何 Engine/Scheduler/KVCache/Executor/Destroy/
Graph/Backend 公共头。
