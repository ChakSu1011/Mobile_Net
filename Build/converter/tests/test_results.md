# Converter 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-14 |
| 测试对象 | `Build.converter` |
| 设计文档 | [converter_design.md](../.doc/converter_design.md) |
| 完整阶段测试 | [Build/tests/test_results.md](../../tests/test_results.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Build/graph/python/_graph.pyd`（2026-11-14 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf；safetensors 由仓库内解析器读取 |

## 2. 测试命令与结果

```bash
PYTHONUTF8=1 python -m unittest Build.converter.test_converter -v
```

实际输出（末尾）：

```text
Ran 13 tests in 0.056s

OK
```

结论：通过，退出码 0。

## 3. 覆盖范围

- `RecordingGraph` 代理图与原生 `Graph` 的 `dump_ir()` 完全一致；
- 输入、常量、多输出 op 的元数据记录与输出命名；
- Qwen3 图计数：2 inputs、310 constants、423 compute ops、1 output，
  IR `ops` 段为 426（含 3 个 I/O op）；
- 310 个常量的 Hugging Face 权重名映射；
- tied / untied LM Head 处理；
- `ModelGraph` JSON 往返与权重缺失降级；
- 非 JSON 属性、重复名称、外部 tensor 等错误路径。

## 4. 未覆盖项与风险

1. **无真实 Qwen3-0.6B 权重**：仓库未包含 safetensors 权重，
   `weight_available` 只能通过合成索引或缺失路径测试，未做真实权重读取。
2. **只读 safetensors 头信息**：不验证权重内容、数值正确性或真实文件布局。
3. **依赖 Graph 扩展**：测试需要已构建的 `_graph.pyd`，未在无扩展环境验证。
4. **仅 Windows 主机验证**：未在 Linux/macOS 或 Android 环境执行。
5. **不执行推理**：Converter 只构图与记录元数据，不创建 OpEnv/Kernel。
