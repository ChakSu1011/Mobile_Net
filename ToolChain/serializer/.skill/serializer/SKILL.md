---
name: serializer
description: 把优化后的 ModelGraph、显存计划与可选权重打包为可版本化的 model.mgf，并实现加载、哈希校验与重建 Graph；适用于离线模型包交付与 Runtime Loader 对接。
---

# Serializer：model.mgf 打包与加载

## 适用场景

- 需要把优化后的 `ModelGraph` 固化为自描述、可版本化、可确定性生成的
  模型包。
- 需要在无 Protobuf、无 torch 的环境中打包权重并记录逐权重哈希。
- 需要生成 metadata-only 包用于图、显存计划与加载流程验证。
- 需要加载模型包并按 manifest 重建 Graph，校验 IR 哈希。

## 不适用场景

- 权重 dtype 转换、量化或压缩：Serializer 只打包原始字节。
- 设备端加载、权重上传、Kernel 创建与推理执行。
- 加密、签名、分片下载与旧版本迁移（首版只有 format version 1）。

## 输入

1. `OptimizeResult`：优化后的 `ModelGraph` 与 `MemoryPlan`。
2. 可选 safetensors 文件或目录。
3. 输出目录与 `overwrite` 选项。

## 输出

- `model.mgf`：112 字节固定头 `<8sIIQQ32sQQ32s>` + 规范化 JSON
  manifest，可选权重区。
- `weights.bin`：可选独立权重文件，按 256 B 对齐。
- `compile_report.json`：构建摘要、IR 哈希、显存计划、验证结果。
- `ModelPackage` 与重建的 Graph（加载路径）。

## 核心不变量

- **格式自描述**：头部包含 magic、version、flags、manifest offset/length、
  manifest SHA-256、weights offset/length、weights SHA-256。
- **manifest 确定性**：canonical JSON（键排序、无多余空白、UTF-8），
  不含时间戳；时间戳只能进 `compile_report.json`。
- **IR 哈希闭环**：manifest 记录重建图 `dump_ir()` 的 SHA-256 与行数；
  Loader 重建后必须复算并比对。
- **权重原样打包**：不做 dtype 转换；源权重 dtype 必须与优化后图精度一致。
- **对齐与哈希**：每个权重按 256 B 对齐，记录 offset、byte_size 与
  SHA-256。
- **metadata-only**：无权重时生成只含 manifest 的包，不残留
  `weights.bin`；310 个权重条目写 `available=false` 与 `reason`。
- **不覆盖非空目录**：默认拒绝覆盖；流水线显式传 `overwrite=True`。
- **包目录干净**：只允许 `model.mgf`、`compile_report.json`、可选
  `weights.bin`；中间产物放 work 目录。

## 标准流程

1. 校验 `OptimizeResult`：精度已确定、activation 形状无符号维、
   显存计划存在。
2. 用优化后的 `ModelGraph` 重建 Graph，计算 `dump_ir()` 的 SHA-256
   与行数。
3. 扫描常量与可选权重，生成 `WeightEntry`；不可用时写入
   `weights not supplied`、`source tensor not found`、`shape mismatch`、
   `dtype mismatch` 或 `byte_size mismatch`。
4. 按常量顺序把可用权重写入 `weights.bin`，逐张量 256 B 对齐并记录
   offset 与 SHA-256。
5. 写 `model.mgf` 与 `compile_report.json`。
6. 重新加载：校验头部、版本、manifest 哈希、权重哈希，并重建 Graph
   比对 IR 哈希。

## 检查清单

- [ ] 112 字节头字段自洽，magic/version 正确。
- [ ] manifest 是 canonical JSON，两次构建字节一致。
- [ ] 重建图 IR SHA-256 与行数和 manifest 一致。
- [ ] 每个可用权重的 offset 对齐、byte_size 与 dtype 匹配。
- [ ] 权重 SHA-256 可在加载时复算通过。
- [ ] 无权重时生成 metadata-only 包且不残留 `weights.bin`。
- [ ] 包目录只含格式文件，work 产物在独立目录。
- [ ] 篡改、截断、版本不兼容均能报错。

## 常见陷阱

1. **默认优化目标是 FP16**：直接提供官方 BF16 权重会全部
   `dtype mismatch`；改用 `--target-precision bfloat16`。
2. **把时间戳写进 manifest**：破坏字节确定性与哈希稳定性。
3. **只比较 manifest 哈希**：IR 哈希必须由重建图计算，不能复用转换期
   缓存值。
4. **覆盖非空输出目录**：`serialize_package` 默认拒绝；手工调用时不要
   误删用户目录。
5. **metadata-only 包当成可用模型**：`weights_available=0` 只能验证图与
   加载流程，不能用于推理。
6. **权重对齐错误**：未按 256 B 对齐会导致 Runtime 直接映射失败。
