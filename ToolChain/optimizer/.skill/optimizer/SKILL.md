---
name: optimizer
description: 对 Converter 产出的 ModelGraph 执行精度推导、Shape 推导与激活显存复用规划，输出确定性的 OptimizeResult 与优化报告；适用于无 torch 的离线 Build 阶段。
---

# Optimizer：精度、Shape 与显存规划

## 适用场景

- 已有一份包含输入、常量、activation、op 与输出绑定的 `ModelGraph`，
  需要为序列化与运行时准备静态精度、静态形状和显存偏移。
- 需要把 BF16 规划转换为 FP16，并保留 FP32 累加等敏感策略。
- 需要基于静态生命周期做激活复用，减少 arena 高水位。
- 需要可重复、可审查的优化报告。

## 不适用场景

- 修改 Graph 图对象或增删 op：Optimizer 只修改 `ModelGraph` 元数据。
- 量化（INT8/INT4）实现：当前只有扩展点。
- 真实显存分配、Device 上传、Kernel 执行或 Dummy Run。
- 动态 shape 或运行时自适应调度。

## 输入

1. `ModelGraph`：Converter 的旁路记录，必须通过自校验。
2. `OptimizeConfig`：`target_precision`、`max_num_tokens`、
   `context_length`、`alignment`、`enable_reuse`。
3. 可选的精度策略表（当前按 op 类型内置）。

## 输出

- `OptimizeResult`：优化后的 `ModelGraph`、`MemoryPlan`、优化报告。
- work 目录下的 `model_graph.json` 与 `optimize_report.json`。
- 精度表、形状表、逐张量生命周期与显存条目。

## 核心不变量

- **只做等宽精度转换**：BF16 → FP16 允许（2 B → 2 B）；
  FP32 → FP16 改变元素大小时必须报错，禁止静默截断。
- **输入精度不变**：`int64` 输入保持 `int64`，不参与浮点转换。
- **Shape 绑定**：`B=1`、`S=max_num_tokens`；所有 activation 必须得到
  静态形状，符号维不得进入 Memory Pass。
- **形状契约**：linear K 维匹配、GQA `num_heads/num_kv_heads` 整除、
  RMSNorm 末维整除、Add 同形。
- **Memory 前置条件**：Memory Pass 入口拒绝未解析形状的 activation。
- **显存不重叠**：生命周期重叠的张量 `[offset, offset+size)` 不得相交；
  所有 offset 按 alignment 对齐。
- **输入/常量不进 arena**：只有 activation 参与复用；图输出 `reusable=false`。
- **确定性**：相同输入与配置产生完全一致的 `ModelGraph` 与显存计划。

## 标准流程

1. 校验 `ModelGraph` 与 `OptimizeConfig`。
2. Precision Pass：按目标精度转换浮点张量；未知精度从生产 op 的
   `precision` 属性继承；非法转换立即报错。
3. Shape Pass：绑定 `B`、`S`，按 op 规则推导输出形状并校验契约；
   标记输出不可复用。
4. Memory Pass：按拓扑序计算 `birth` / `last_use`；输出绑定视为下标
   `N` 的消费者；按 `(birth, -size, name)` 排序做 first-fit 复用；
   校验对齐、重叠、峰值与常量字节。
5. 生成报告并做不变量自检。

## 检查清单

- [ ] 所有 activation 的 `static_shape` 非空，无符号维。
- [ ] 精度转换后 `byte_size` 与元素大小一致。
- [ ] 每个 op 的输入/输出形状满足对应契约。
- [ ] 生命周期 `birth <= last_use`，输出绑定被计入最后使用。
- [ ] 复用张量字节区间不重叠，offset 全部对齐。
- [ ] `arena_bytes` 与 `peak_bytes` 口径正确，常量区字节独立统计。
- [ ] 关闭 `enable_reuse` 后 arena 增大且仍无重叠。
- [ ] 报告字段可 JSON 序列化且两次运行一致。

## 常见陷阱

1. **在 Memory Pass 里处理符号维**：必须先完成 Shape Pass；否则会得到
   不可执行的偏移计划。
2. **清掉常量 pitch**：Shape Pass 只清输入 pitch；常量 pitch 必须保留，
   否则重建 Graph 时 IR 哈希漂移。
3. **把 FP32 静默转 FP16**：元素大小变化会破坏 `byte_size` 与权重打包，
   必须报错。
4. **输出参与复用**：图输出生命周期延续到图结束，必须
   `reusable=false`。
5. **生命周期边界错误**：输出绑定是虚拟消费者，`last_use` 不能只取
   最后一个 op 下标。
6. **报告写入时间戳**：优化报告可含时间戳，但 `ModelGraph` 与显存计划
   必须确定，避免影响序列化字节稳定性。
