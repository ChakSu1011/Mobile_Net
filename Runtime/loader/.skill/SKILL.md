# Skill: Runtime Loader 与模型包加载

## 适用场景

- 为 `Runtime/loader/` 实现或修改模型包加载流程（`ILoader::Load`）；
- 新增/调整 `model.mgf` 固定头、manifest 视图、权重表或 memory plan 解析；
- 校验 C++ `DumpIr` 与 Python `dump_ir` 的字节级一致性（G1/G2）；
- 排查“加载失败但错误码不明确”“缓冲绑定后 IR 哈希不一致”“权重上传后
  Kernel 取址错误”等问题；
- 为新的模型包（非 Qwen3）做通用加载适配。

不适用：Scheduler/Executor/KVCache/Destroy 实现、算子数值正确性、
OCL 设备调优、模型转换与量化。

## 输入

1. `Runtime/.doc/Runtime_design.md`：4.2.1 Loader、4.3/4.4、附录 B（权威）；
2. `Runtime/loader/.doc/loader_design.md`：本模块详细设计；
3. 公共契约：`loader.h`、`backend.h`、`backend_registry.h`、
   `op_env_factory.h`、`resource_stack.h`、`kernel_context.h`；
4. Graph 接口：`Graph/include/*.h`、`Graph/python/graph_python.cc`（DumpIr）；
5. 包格式：`ToolChain/serializer/{format,loader}.py`、
   `ToolChain/common/model_ir.py`；
6. 真实包：`Models/<模型>/artifacts/<包名>/model.mgf` + `weights.bin`；
7. 构建环境：C++17、`python -m ziglang c++`、Graph 与 backend 静态库。

## 步骤

### 1. 对齐契约

- 确认 112B 头字段顺序：`<8sIIQQ32sQQ32s`；
- 确认 Graph 的 `SortOp` 会把 Input op 纳入拓扑序（Input tensor 有
  producer），因此 Orderly 列表含 Input/Output op；
- 确认 `KernelOpEnv::Init` 在创建 Kernel 前读取 Storage 并校验
  `device_type==kDevice`、`byte_size==shape×dtype`，所以必须先绑定。

### 2. 实现基础工具

- `Runtime/common/json.*`：严格 JSON、整数/浮点区分、偏移错误；
- `Runtime/common/sha256.*`：流式 SHA-256，小写 hex；
- `Runtime/common/file_util.*`：大小、区间、顺序分块读取与文件哈希；
- 全部返回 `Status`，不抛异常，使用 `U8/I32/U64/Usize/C8`。

### 3. 实现 manifest 视图

- 头部：magic/version/flags/manifest 边界/weights flag 一致性；
- 视图：tensors/ops/outputs/weights/memory_plan/statistics；
- 校验：静态维度、名称唯一、引用完整、统计交叉、权重条目与常量一一对应、
  plan 覆盖全部 activation、对齐与字节数一致；
- `ModelConfig.dtype` 取 `optimization.config.target_precision`，不要取
  `model.config.dtype`（Qwen3 官方为 bf16，包内目标是 fp16）。

### 4. 实现 DumpIr

- 复制 `Graph/python/graph_python.cc` 的 DumpIr 逻辑，调用相同的
  `StorageToString/AttrValueToString/OpTypeToStr`；
- 在**绑定任何 data_ptr 之前**计算 SHA-256 并与 `graph.ir_sha256` 比较；
  绑定后 Storage 文本中的 `device/data/offset/bytes` 会变化。

### 5. 实现 Loader 编排

顺序不可交换：

```text
头 → manifest → ModelConfig → 图重建 → SortOp → IR 哈希
→ 张量索引 → 预填 Storage 元数据 → Backend Init/Capability
→ 权重 ConvertParam + 绑定 → 输入 Buffer → arena → GraphEnv/OpEnv
→ workspace 回填 → 移交结果
```

- 常量 `data_ptr = Buffer 对象地址`（不是 `DevicePtr()`）；
- 每个 Buffer 的释放闭包捕获 `shared_ptr<unique_ptr<Buffer>>` 与后端状态；
  后端释放闭包最先 Push，LIFO 最后执行；
- OpEnv 创建按 Input + Orderly + Output 并用 `set<Op*>` 去重。

### 6. 编译与验证

```bash
cd d:/Mobile_Net
# 编译（新文件 + resource_stack.cc）
python -m ziglang c++ -std=c++17 -O2 -Wall -Wextra -I d:/Mobile_Net \
  -c Runtime/loader/loader.cc -o <build>/loader.o
# 链接：runtime_backend 必须 whole-archive，避免 Mock 自注册被裁剪
python -m ziglang c++ -O2 -o <build>/loader_smoke.exe <objs> \
  -Wl,--whole-archive Runtime/backend/build/libruntime_backend.a \
  -Wl,--no-whole-archive Graph/build/libgraph.a
# 正向
<build>/loader_smoke.exe Models/Qwen3/artifacts/qwen3_0_6b
# metadata-only
<build>/loader_smoke.exe <只含 model.mgf 的目录> --metadata-only --no-verify
# 负向
<build>/loader_smoke.exe <坏 magic 包> --no-verify
```

### 7. 记录四件套

- 设计文档：`Runtime/loader/.doc/loader_design.md`；
- 代码：`manifest/ir_dump/loader/tests`；
- 测试结果：`Runtime/loader/tests/test_results.md`（真实命令与输出）；
- skill：本文件。

## 输出

- 可编译的 `manifest.*`、`ir_dump.*`、`loader.cc`、`loader_smoke.cc`；
- `LoadResult`：Graph、Backend、ModelConfig、weights/arena/io 字节数；
- 失败时 result 不修改、ResourceStack 可完整回滚；
- 测试结果与设计文档已更新。

## 常见陷阱

1. **空 pitch 被误判**：manifest 中 `pitch: []` 表示与 shape 相同，不是
   长度不一致；只有非空且长度不匹配才是错误。
2. **IR 哈希时机**：先哈希后绑定；填充 input/activation Storage 也要放在
   哈希之后。
3. **输入/activation Storage 未预填**：Graph 重建后它们的 Storage 是空
   描述；必须在 OpEnv 创建前从 manifest 填 shape/precision/layout/
   byte_size，否则 `CheckTensor` 或能力校验失败。
4. **常量绑定对象地址**：Mock Kernel 把 `view.data` 当作 `MockBuffer*`
   并读取 `Data()+offset`；绑定 `DevicePtr()` 会崩溃。
5. **OpEnv 重复创建**：Orderly 列表已含 Input/Output op，需要按指针去重，
   否则 `SetOpEnv` 会替换并销毁刚创建的 OpEnv。
6. **Mock 注册被裁剪**：链接静态库必须 whole-archive，或显式链接
   `register_mock.cc` 对象。
7. **metadata-only 不可执行**：占位缓冲只有 64 字节，执行 Kernel 会越界。
8. **权重 SHA 成本**：全量 + 逐项哈希约 1.19 GB × 2；测试可用
   `--no-verify`，但发布路径必须开启。
9. **后端生命周期**：正常 Destroy 必须先 `ClearOpEnv`，再
   `ResourceStack.Clear()`，最后 `Executor::Deinit`；顺序错误会导致
   Buffer/Kernel 悬空。
10. **错误码映射**：包内容问题（含 JSON/IR/权重哈希）统一
    `kInvalidModelFile`；文件打不开用 `kIoError`；不要混用。
