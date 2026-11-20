# Executor 模块测试结果

## 1. 测试范围

本文件记录 `Runtime/executor/` 默认 Executor 实现的真实测试输出，覆盖：

- `CreateDefaultExecutor()` 返回实例与未初始化状态；
- `Init` 接管 Backend、按 Graph Storage 预分配主机 I/O 缓冲、错误路径与
  后端释放；
- `Execute` 的 plan 校验、token/position packed 填充与补 0、IoBinder
  输入/输出绑定、StepContext 注入、逐 op Forward、sample 回读与
  `StepResult` 填充；
- 单输入图与双输入图；
- `output_slot<0`、超容量、total 不一致、输出 slot 越界、缺失 OpEnv、
  空 plan、空 result；
- `Deinit` 幂等与 Backend 释放。

测试图：`Input → Gather → Linear → Sample → Output`，由 MockBackend 的
CPU 参考 Kernel 执行。权重固定为 `embedding[i]=[i,1]`、
`W[v]=[2v,-v²]`，使 `logits[v]=-(v-i)²+i²`，argmax 恒等于输入 token id。

不包含：真实设备后端、KV Cache、Scheduler/Engine 端到端、非 int64 输入
拒绝、第 3 个及以后输入、`storage.byte_size==0` 推导路径。

## 2. 测试环境

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-11-20 |
| 操作系统 | Windows 11 Home China 10.0.26200 |
| 编译器 | `python -m ziglang c++`，clang 22.1.8，zig 0.17.1 |
| 目标平台 | x86_64-unknown-windows-gnu |
| 语言标准 | C++17 |
| 包含路径 | 仓库根目录 `d:/Mobile_Net` |
| 链接库 | `Runtime/backend/build/libruntime_backend.a`、`Graph/build/libgraph.a` |
| 工作目录 | `d:/Mobile_Net` |
| 测试源文件 | `Runtime/executor/executor.cc`、`Runtime/executor/tests/executor_test.cc` |
| 测试程序 | `Runtime/executor/build/executor_test.exe` |

## 3. 测试命令

```bash
cd d:/Mobile_Net

# 编译与链接
python -m ziglang c++ -std=c++17 -Wall -Wextra -I "d:/Mobile_Net" \
  Runtime/executor/executor.cc Runtime/executor/tests/executor_test.cc \
  Runtime/backend/build/libruntime_backend.a Graph/build/libgraph.a \
  -o Runtime/executor/build/executor_test.exe

# 运行
./Runtime/executor/build/executor_test.exe
```

## 4. 实际输出

### 4.1 编译

```text
（无输出）
```

退出码：`0`，无错误、无 `-Wall -Wextra` 告警。

### 4.2 运行

```text
PASS CreateDefaultExecutor returns instance
PASS backend is null before Init
PASS Deinit before Init is idempotent
PASS Init rejects null backend [Executor::Init: backend/graph/io_binder must not be null]
PASS Init rejects graph without Input op [Executor::Init: graph has no Input op]
PASS failed Init releases backend
PASS Init rejects graph without Sample op [Executor::Init: graph has no Sample op]
PASS Init rejects null graph/io_binder [Executor::Init: backend/graph/io_binder must not be null]
PASS build two-input graph: ok
PASS create two-input OpEnvs
PASS two-input Init succeeds
PASS backend() returns owned backend
PASS double Init is rejected [Executor::Init: already initialized]
PASS double Init keeps original backend
PASS Execute prefill succeeds
PASS prefill returns one result
PASS prefill result is ok
PASS prefill token equals argmax of last packed row
PASS prefill token is selected with zero probs
PASS binds 2 inputs and 1 output
PASS input names are bound by op name
PASS token host buffer is packed in plan order
PASS position host buffer is packed in plan order
PASS output name is bound by op name
PASS output host buffer receives all sample rows
PASS Execute propagates IoBinder failure [IoBinder::SetInput(input.input_ids): injected io bind failure]
PASS failed binding leaves StepResult empty
PASS Execute decode succeeds
PASS decode returns two results
PASS decode result 0 maps to packed slot 0
PASS decode result 1 maps to packed slot 1
PASS decode token buffer pads remainder with zero
PASS decode position buffer pads remainder with zero
PASS Execute chunked prefill succeeds
PASS output_slot<0 produces no request result
PASS Execute rejects oversize packed tokens [input.input_ids: packed tokens exceed host buffer capacity]
PASS Execute rejects total_tokens mismatch [Execute: plan.total_tokens does not match entry sum]
PASS Execute rejects out-of-range output_slot [Executor::Execute: request 41 output_slot exceeds sample output size]
PASS Execute rejects empty plan [Execute: plan has no entries]
PASS Execute rejects null result [Executor::Execute: result must not be null]
PASS Execute rejects missing OpEnv [op=linear: OpEnv is not bound]
PASS Deinit releases backend
PASS backend is null after Deinit
PASS Deinit is idempotent
PASS build single-input graph: ok
PASS single-input graph has one Input op
PASS create single-input OpEnvs
PASS single-input Init succeeds
PASS single-input Execute succeeds
PASS single-input returns two results
PASS single-input tokens equal plan token ids
PASS single-input binds exactly one input
PASS single-input token buffer is padded with zero
PASS single-input Deinit succeeds
ALL TESTS PASSED
```

共 54 条 PASS 断言；退出码：`0`。

## 5. 用例与覆盖对照

| 用例 | 需求 | 实际断言（节选） | 结果 |
| --- | --- | --- | --- |
| TC-EX-01 | REQ-EX-01 | `CreateDefaultExecutor returns instance`、`backend is null before Init` | PASS |
| TC-EX-02 | REQ-EX-01/14 | `Deinit before Init is idempotent`、`failed Init releases backend`、`Deinit is idempotent` | PASS |
| TC-EX-03 | REQ-EX-02 | `build two-input graph: ok`、`two-input Init succeeds` | PASS |
| TC-EX-04 | REQ-EX-03/06/07/08/09/10/11/12 | `prefill token equals argmax of last packed row`、`binds 2 inputs and 1 output`、`token/position host buffer is packed in plan order`、`output host buffer receives all sample rows` | PASS |
| TC-EX-05 | REQ-EX-04/08/12 | `decode result 0/1 maps to packed slot`、`decode token/position buffer pads remainder with zero` | PASS |
| TC-EX-06 | REQ-EX-12 | `output_slot<0 produces no request result` | PASS |
| TC-EX-07 | REQ-EX-13 | `Execute rejects missing OpEnv [op=linear: OpEnv is not bound]` | PASS |
| TC-EX-08 | REQ-EX-05/13 | `Execute rejects oversize packed tokens` | PASS |
| TC-EX-09 | REQ-EX-13 | `total_tokens mismatch`、`out-of-range output_slot`、`empty plan`、`null result` | PASS |
| TC-EX-10 | REQ-EX-03 | `single-input binds exactly one input`、`single-input token buffer is padded with zero`、`single-input tokens equal plan token ids` | PASS |
| TC-EX-11 | REQ-EX-01/13 | `Init rejects graph without Input/Sample op`、`Init rejects null graph/io_binder`、`double Init is rejected` | PASS |
| TC-EX-12 | REQ-EX-14 | `Deinit releases backend`、`backend is null after Deinit` | PASS |
| TC-EX-13 | REQ-EX-13 | `Execute propagates IoBinder failure`、`failed binding leaves StepResult empty` | PASS |

## 6. 开发过程中的失败与修复

为保证记录真实，以下列出首次运行的真实失败及修复；最终输出已全部 PASS。

1. **首次编译：`IoBinder` 命名空间错误**

   ```text
   executor_test.cc:58:18: error: no member named 'IoBinder' in namespace 'mobilenet'
   ```

   原因：`IoBinder` 位于 `mobilenet::runtime`，测试 `using` 写成了
   `mobilenet::IoBinder`。修复：改为 `mobilenet::runtime::IoBinder`。

2. **第一次运行：`host storage is not bound for this step`**

   ```text
   FAIL Execute prefill succeeds [op=input.input_ids: op=input.input_ids: host storage is not bound for this step]
   ```

   原因：测试 `TestIoBinder` 只记录绑定，未像 Engine 一样把主机 Storage
   写入 `op->SetAttr("storage", ...)`。修复：测试 Binder 增加 op 映射并在
   校验成功后写入 attr，模拟真实 Engine 行为。

3. **第二次运行：sample token 全为 0**

   ```text
   FAIL output host buffer receives all sample rows actual=[0,0,0,0] expected=[3,5,0,7]
   ```

   原因：夹具权重 `embedding[i]=[1,i]`、`W[v]=[-v,1]` 使
   `logits[v]=i-v`，argmax 恒为 `v=0`，测试预期本身错误。修复：改用
   `embedding[i]=[i,1]`、`W[v]=[2v,-v²]`，使 argmax 唯一等于 token id。

上述问题均属于测试台架/夹具错误，不涉及 Executor 实现；修复后未放宽任何
断言，最终 54 条断言全部通过。

## 7. 未覆盖项与风险

| 项目 | 说明 | 风险 |
| --- | --- | --- |
| 真实设备后端 | 仅覆盖 MockBackend（主机 memcpy） | 异步流、设备地址与同步语义未验证 |
| Engine/Scheduler/KVCache 联调 | 默认 Loader 尚无可用模型包 | `output_slot<0` 时 Engine 当前会按缺失结果判 `kInternal`（已知接口限制） |
| 非 int64 token/position | 代码返回 `kUnsupportedDtype`，测试未构造该图 | 错误路径分支未执行 |
| 第 3+ 输入 | 缓冲按 Init 分配并绑定，但不填 plan 数据 | 扩展输入语义未验证 |
| `byte_size==0` 推导 | `ResolveByteSize` 已实现，测试 Storage 均带字节数 | 推导分支未执行 |
| 动态形状/更大 batch | 测试固定 `[1,4]`、词表 8 | 多 batch、真实 Qwen3 图未验证 |
| 内存复用优化 | `seq_lens/sample_tokens` 每步分配 | 高频 Decode 下有少量分配开销 |

## 8. 结论

- 编译与链接：成功，无告警，退出码 0；
- 运行：54 条断言全部 PASS，输出 `ALL TESTS PASSED`，退出码 0；
- 需求 REQ-EX-01 至 REQ-EX-15 的已覆盖部分均通过，未覆盖项如实列于
  第 7 节；
- 结论：Executor 默认实现达到本阶段验收标准，可作为 Engine 默认
  `CreateDefaultExecutor()` 实现使用；端到端集成需待 Loader 模型包与
  Engine/Scheduler 的“无 token 步”支持。
