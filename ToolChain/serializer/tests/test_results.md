# Serializer 模块测试结果

| 项目 | 内容 |
|------|------|
| 报告日期 | 2026-11-15 |
| 测试对象 | `ToolChain.serializer` |
| 设计文档 | [serializer_design.md](../.doc/serializer_design.md) |
| 完整阶段测试 | [ToolChain/tests/test_results.md](../../tests/test_results.md) |

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.13.3 |
| 解释器 | `C:\Users\18626\AppData\Local\Programs\Python\Python313\python.exe` |
| Graph 扩展 | `Graph/build/python/_graph.pyd`（2026-11-15 构建） |
| 字符编码 | `PYTHONUTF8=1` |
| 外部依赖 | 无 PyTorch / LibTorch / Protobuf；权重由仓库内解析器读取 |

## 2. 测试命令与结果

```bash
PYTHONUTF8=1 python -m unittest ToolChain.serializer.test_serializer -v
```

实际输出（末尾）：

```text
Ran 13 tests in 0.219s

OK
```

结论：通过，退出码 0。

## 3. 覆盖范围

- 112 字节固定头与版本字段；
- manifest / 权重 SHA-256 篡改检测、截断文件、版本不兼容；
- IR SHA-256 校验与重建图一致性；
- 小型图序列化与加载往返；
- 合成权重打包、单张量 256 B 对齐、逐权重哈希；
- 两次序列化字节一致；
- metadata-only 包生成与加载；
- Qwen3-0.6B 模型包加载与重建图。

## 4. 未覆盖项与风险

1. **模块测试未覆盖真实权重**：Serializer 单测使用 metadata-only 包与
   合成 safetensors；真实 Qwen3 FP16 权重打包由阶段级流水线测试覆盖
   （见 [ToolChain/tests/test_results.md](../../tests/test_results.md)）。
2. **Serializer 不做 dtype 转换**：它原样打包传入的权重字节。流水线会
   优先传入 Precision Pass 生成的 FP16 转换文件；手工调用时若图精度与
   权重 dtype 不一致，会因 `dtype mismatch` 全部不可用。
3. **未执行运行时加载**：未验证设备端权重上传、Kernel 创建与推理数值。
4. **未实现旧版本迁移**：首版只有 format version 1。
5. **仅 Windows 主机验证**：未在 Android/Linux 目标环境执行。
