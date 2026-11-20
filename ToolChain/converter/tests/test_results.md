# Converter 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-15 |
| 测试对象 | `ToolChain.converter` |
| 设计文档 | [converter_design.md](../.doc/converter_design.md) |
| 完整阶段测试 | [ToolChain/tests/test_results.md](../../tests/test_results.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Graph/build/python/_graph.pyd`（2026-11-15 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf；safetensors 由仓库内解析器读取 |

## 2. 测试命令与结果

```bash
PYTHONUTF8=1 python -m unittest ToolChain.converter.test_converter -v
```

实际输出（末尾）：

```text
Ran 22 tests in 0.109s

OK
```

结论：通过，退出码 0。

### 2.1 输入配置合并验证

Qwen3 有效运行参数（`ConversionResult.inference_config.to_dict()`）：

```json
{
  "context_length": 40960,
  "vocab_size": 151936,
  "max_num_tokens": 256,
  "max_batch_size": 1,
  "do_sample": true,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95,
  "repetition_penalty": 1.0,
  "seed": 0,
  "stop_token_ids": [151645, 151643],
  "bos_token_id": 151643,
  "pad_token_id": 151643
}
```

其中 `context_length`/`vocab_size` 来自 `config.json`，
`do_sample`/`temperature`/`top_k`/`top_p`/`eos`/`pad` 来自
`generation_config.json`；`max_num_tokens`/`max_batch_size`/
`repetition_penalty`/`seed` 来自
`Models/Qwen3/input/inference_config.json`。`max_num_tokens` 表示单 Step
所有请求累加的 token 预算；单请求生成上限由运行时按
`context_length - prompt_length` 推导。

结论：通过。官方字段未被重复声明；重复声明 `temperature`、未知字段、
`max_num_tokens` 越界、官方 `top_p=2.0` 与派生字段 `max_new_tokens`
均被拒绝；官方采样字段为 `null` 时按缺省值处理。

## 3. 覆盖范围

- `RecordingGraph` 代理图与原生 `Graph` 的 `dump_ir()` 完全一致；
- `create_sample` 默认 argmax/axis=-1/int64、`sample.out` 命名与元数据；
- 输入、常量、多输出 op 的元数据记录与输出命名；
- Qwen3 图计数：2 inputs、310 constants、312 compute ops、1 output，
  IR `ops` 段为 315（含 3 个 I/O op）；
- LM Head 与 Output 之间的默认 Argmax Sample 记录与 `output.tokens` 绑定；
- 310 个常量的 Hugging Face 权重名映射；
- tied / untied LM Head 处理；
- `ModelGraph` JSON 往返与权重缺失降级；
- 输入配置与官方 config/generation_config 合并、缺失降级、null 缺省、
  派生字段拒绝与错误路径；
- 非 JSON 属性、重复名称、外部 tensor 等错误路径。

## 4. 未覆盖项与风险

1. **无真实 Qwen3-0.6B 权重**：仓库未包含 safetensors 权重，
   `weight_available` 只能通过合成索引或缺失路径测试，未做真实权重读取。
2. **只读 safetensors 头信息**：不验证权重内容、数值正确性或真实文件布局。
3. **依赖 Graph 扩展**：测试需要已构建的 `_graph.pyd`，未在无扩展环境验证。
4. **仅 Windows 主机验证**：未在 Linux/macOS 或 Android 环境执行。
5. **不执行推理**：Converter 只构图与记录元数据，不创建 OpEnv/Kernel。
6. **输入配置无版本字段**：当前格式固定且拒绝未知字段；未来不兼容变更
   需要新增字段兼容策略或新文件名，不能静默改变语义。

## 5. Qwen3 KV Adapter（2026-11-21）

新增 `qwen3_kv` adapter 与 `UT-CONV-012`：复用 Qwen3 图结构，把 28 个
attention op 的 `use_kv_cache` 置为 true，并写入 `kv_layer_index=0..27`。

```bash
python -m pytest -q ToolChain/converter
```

实际输出：

```text
.......................                                                  [100%]
23 passed in 0.20s
```

结论：KV adapter 已注册，图属性与 IR 输出符合 Paged Attention 契约；
默认 `qwen3` adapter 继续输出 `use_kv_cache=false`。
