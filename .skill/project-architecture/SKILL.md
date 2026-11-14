---
name: project-architecture
description: 导航 MobileNet 推理引擎的整体架构、模块边界与仓库目录规则，定位 Graph、ToolChain、Models、Runtime 的设计文档、skill 和测试结果；适用于跨模块开发、评审与交付检查。
---

# 项目整体架构导航

## 适用场景

- 需要在多个模块之间定位职责边界，判断改动应落在 Graph、ToolChain、
  Models 还是 Runtime。
- 需要查找某个模块的权威设计文档、独立 skill 或测试结果。
- 需要确认模型构建、序列化产物与运行时加载的流转路径。
- 需要进行交付检查，确认文档、代码、测试和 skill 四件套齐全。

## 不适用场景

- 具体模块内部的算法实现：进入对应模块 `.doc` 与 `.skill`。
- 运行推理、性能调优或设备端问题定位：属于 Runtime 模块。
- 训练、量化校准或模型结构搜索。

## 模块分层

```text
Models/       模型结构定义与 Qwen3 组图
    │
    ▼
Graph/        C++17 runtime 静态图 IR + pybind11 最小构图接口
    │
    ▼
ToolChain/    Converter → Optimizer → Serializer（Python 离线工具链）
    │
    ▼
Models/<model>/artifacts/   模型包 model.mgf / weights.bin
    │
    ▼
Runtime/      Loader / Engine / Scheduler / Kernel 执行静态图
```

- **Graph**：唯一的可执行图表示；`Graph/Op/Tensor/Storage/GraphEnv/OpEnv`。
- **ToolChain**：模型转换、精度/形状/显存优化、模型包序列化。
- **Models**：模型配置、权重命名与组图逻辑；产物放在模型目录下。
- **Runtime**：运行时加载、调度、KV Cache、Kernel 与后端。

## 目录与文档规则

- 每个模块的设计文档：`.doc/<模块名>_design.md`。
- 每个模块的 skill：`.skill/<skill-name>/SKILL.md`。
- 每个模块的测试结果：`tests/test_results.md`。
- 模型构建产物：`Models/<模型目录>/artifacts/`。
- 根目录设计文档：`.doc/MobileNet_design.md`。
- Graph 原生构建输出：`Graph/build/`，不是 ToolChain 目录。
- 外部参考知识库：`Reference/`，只作本地参考，不提交 Git/GitHub。

## 输入

1. 当前任务涉及的模块与目标。
2. `.doc/MobileNet_design.md` 与对应模块设计文档。
3. 模块 skill、测试结果与源码目录。

## 输出

- 明确的模块归属与改动范围。
- 对应设计文档、skill、测试结果与代码路径。
- 交付检查结论。

## 标准流程

1. 先读根设计文档的“总体架构”“模块划分”“关键流程”。
2. 按任务定位到 Graph、ToolChain、Models 或 Runtime 的模块文档。
3. 检查模块 `.skill` 中的适用场景、流程与常见陷阱。
4. 修改代码前确认设计文档；设计变更先更新文档。
5. 按模块测试命令执行测试，结果写入 `tests/test_results.md`。
6. 若产物变化，确认输出在 `Models/<模型目录>/artifacts/`。

## 检查清单

- [ ] 改动落在正确模块，未把模型逻辑塞进 Graph 或 ToolChain。
- [ ] 设计文档位于模块 `.doc/<模块名>_design.md`。
- [ ] skill 位于模块 `.skill/`，未放到仓库级目录。
- [ ] 测试结果位于模块 `tests/test_results.md`。
- [ ] 模型包位于 `Models/<模型目录>/artifacts/`。
- [ ] Graph 仍是唯一可执行图，`ModelGraph` 仅作为 sidecar。
- [ ] Runtime 不依赖 torch/LibTorch/Protobuf。
- [ ] 文档、代码、测试、skill 四件套一致。

## 常见陷阱

1. **把 `Graph/build` 当成 ToolChain**：它是 Graph 的 CMake 构建输出，
   不是 Python 工具链代码。
2. **继续使用 `Build/` 路径**：目录已改名为 `ToolChain/`，Python 包名
   也是 `ToolChain.*`。
3. **把模型包放在 ToolChain 下**：产物属于模型目录，应写入
   `Models/<模型目录>/artifacts/`。
4. **把 IR `ops=426` 当成计算 op 数**：Qwen3 计算 op 是 423，另外 3 个
   是 I/O op。
5. **在 ToolChain 中复制模型层结构**：模型层结构应复用 `Models/Qwen3`
   的 Builder。
6. **引入 Protobuf 或 torch 作为运行时依赖**：当前模型包是自定义
   `model.mgf`，构图与运行时不依赖这些库。
7. **设计文档保留备份或合并版**：只保留一份权威文档，历史版本由 git
   记录。
