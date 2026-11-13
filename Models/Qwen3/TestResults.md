# Qwen3-0.6B Graph 组图测试结果

| 项目 | 内容 |
|------|------|
| 文档版本 | v1.0 |
| 日期 | 2026-11-13 |
| 被测代码 | `Models/Qwen3/qwen3_config.py`、`Models/Qwen3/qwen3_graph_builder.py` |
| 测试代码 | `Models/Qwen3/test_qwen3_graph.py` |
| 设计文档 | [Models/模型组图设计文档.md](../模型组图设计文档.md) |
| 结论 | 12 个用例全部通过；图计数、权重形状和错误路径符合设计 |

## 1. 测试环境

| 项目 | 实际值 |
|------|--------|
| 操作系统 | Windows-11-10.0.26200-SP0 |
| Shell | Git Bash |
| Python | 3.13.3 |
| PyTorch | 2.12.0+cpu（仅测试） |
| Transformers | 5.9.0（仅测试） |
| Graph Python 包 | `Graph/python/mobilenet_runtime_graph/__init__.py` |
| Graph 原生扩展 | `Build/graph/python/_graph.pyd` |
| 执行方式 | `PYTHONUTF8=1 python Models/Qwen3/test_qwen3_graph.py -v` |

## 2. 测试命令

```bash
# 1. Python 语法检查
PYTHONUTF8=1 python -m py_compile \
  Models/Qwen3/qwen3_config.py \
  Models/Qwen3/qwen3_graph_builder.py \
  Models/Qwen3/test_qwen3_graph.py

# 2. 完整测试
PYTHONUTF8=1 python Models/Qwen3/test_qwen3_graph.py -v

# 3. 命令行构图摘要
PYTHONUTF8=1 python Models/Qwen3/qwen3_graph_builder.py \
  --config Models/Qwen3/config.json

# 4. 完整 IR 行数
PYTHONUTF8=1 python Models/Qwen3/qwen3_graph_builder.py \
  --config Models/Qwen3/config.json --dump-ir | wc -l
```

## 3. 用例与结果

| 编号 | 测试方法 | 验证内容 | 结果 |
|------|----------|----------|------|
| UT-QWEN3-001 | `test_config_defaults` | 配置字段、q/kv 宽度、GQA 比例 | 通过 |
| UT-QWEN3-002 | `test_graph_counts` | 2 inputs、310 constants、426 ops、1 output | 通过 |
| UT-QWEN3-003 | `test_layer_zero_topology`、`test_final_lm_head_and_output` | 层内数据依赖、final norm、lm_head、output 连接 | 通过 |
| UT-QWEN3-004 | `test_all_layers_have_fifteen_ops` | 28 层，每层 15 个唯一 op | 通过 |
| UT-QWEN3-005 | `test_layer_zero_topology` | q/k norm、RoPE、GQA attention 输入连接 | 通过 |
| UT-QWEN3-006 | `test_constants_and_tied_weight` | tied LM Head 复用 `embedding.weight` | 通过 |
| UT-QWEN3-007 | `test_untied_lm_head` | untied 时新增 `lm_head.weight`，constant 数 311 | 通过 |
| UT-QWEN3-008 | `test_invalid_configs` | 7 类非法配置抛出 `ValueError` | 通过 |
| UT-QWEN3-009 | `test_weight_shapes_and_parameter_count`、`test_transformers_weight_shapes` | 权重形状与 Transformers meta 模型一致；参数总量 596,049,920 | 通过 |
| UT-QWEN3-010 | `test_dump_is_stable` | 连续两次 `dump_ir()` 完全一致 | 通过 |
| UT-QWEN3-011 | `test_config_file_matches_transformers` | `config.json` 可被 Transformers `Qwen3Config` 直接加载 | 通过 |

补充说明：

- `test_transformers_weight_shapes` 使用 `torch.device("meta")` 实例化
  `Qwen3ForCausalLM`，不分配真实权重内存；
- 该用例用 `named_parameters(remove_duplicate=False)`，否则 tied 的
  `lm_head.weight` 会被 PyTorch 去重，无法核对；
- `test_layer_zero_topology` 断言的是 `sort_op()` 输出的数据依赖顺序，
  不是 builder 的创建顺序。

## 4. 实际测试输出

```text
test_all_layers_have_fifteen_ops (__main__.Qwen3GraphTest.test_all_layers_have_fifteen_ops)
28 个 Decoder Layer 各包含 15 个唯一 op。 ... ok
test_config_defaults (__main__.Qwen3GraphTest.test_config_defaults)
配置字段与 Qwen3-0.6B 一致，派生宽度正确。 ... ok
test_config_file_matches_transformers (__main__.Qwen3GraphTest.test_config_file_matches_transformers)
config.json 可被 Transformers Qwen3Config 直接加载。 ... ok
test_constants_and_tied_weight (__main__.Qwen3GraphTest.test_constants_and_tied_weight)
默认图登记 310 个 constant，LM Head 复用 embedding 权重。 ... ok
test_dump_is_stable (__main__.Qwen3GraphTest.test_dump_is_stable)
重复调用 dump_ir() 输出完全一致。 ... ok
test_final_lm_head_and_output (__main__.Qwen3GraphTest.test_final_lm_head_and_output)
最后一层、final_norm、lm_head 和 output 的连接正确。 ... ok
test_graph_counts (__main__.Qwen3GraphTest.test_graph_counts)
图名称和 inputs/constants/ops/outputs 计数符合设计。 ... ok
test_invalid_configs (__main__.Qwen3GraphTest.test_invalid_configs)
非法配置在 builder 初始化阶段抛出 ValueError。 ... ok
test_layer_zero_topology (__main__.Qwen3GraphTest.test_layer_zero_topology)
layer.0 的 15 个 op 顺序和关键连接符合源码结构。 ... ok
test_transformers_weight_shapes (__main__.Qwen3GraphTest.test_transformers_weight_shapes)
与 Transformers Qwen3ForCausalLM meta 模型的权重形状一致。 ... ok
test_untied_lm_head (__main__.Qwen3GraphTest.test_untied_lm_head)
tie_word_embeddings=false 时新增独立 lm_head.weight。 ... ok
test_weight_shapes_and_parameter_count (__main__.Qwen3GraphTest.test_weight_shapes_and_parameter_count)
权重形状、总参数和 bfloat16 字节数与 Qwen3-0.6B 一致。 ... ok

----------------------------------------------------------------------
Ran 12 tests in 6.309s

OK
```

## 5. 命令行验证

构图摘要实际输出：

```text
graph qwen3_0_6b
inputs 2
constants 310
ops 426
outputs 1
```

完整 IR 实际行数：

```text
750
```

IR 关键片段（与 `dump_ir()` 实际输出一致）：

```text
  [424] lm_head linear src=[final_norm.out,embedding.weight] dst=[lm_head.out] attrs={bias=false,in_features=1024,logits_to_keep=0,out_features=151936,precision=bfloat16,tie_word_embeddings=true,transpose_weight=true}
  [425] output.logits output src=[lm_head.out] dst=[] attrs={storage=storage(precision=unknown,layout=row_major,device=unknown,shape=[],offset=0,bytes=0,data=unset,reusable=true)}
outputs 1
  output.logits <- [lm_head.out]
```

`[424]` 证明 tied 模式下 LM Head 直接复用 `embedding.weight`，没有额外的
`lm_head.weight` constant。

## 6. 静态检查

```bash
PYTHONUTF8=1 python -m py_compile \
  Models/Qwen3/qwen3_config.py \
  Models/Qwen3/qwen3_graph_builder.py \
  Models/Qwen3/test_qwen3_graph.py
```

实际结果：退出码 0，无输出。

80 显示列检查（中文字符按 2 列计算）覆盖三个 Python 文件，实际结果：

```text
Models/Qwen3/qwen3_config.py
Models/Qwen3/qwen3_graph_builder.py
Models/Qwen3/test_qwen3_graph.py
```

三个文件均无超过 80 显示列的代码行。

## 7. 中间失败与修复

测试过程中真实出现并修复了以下问题，最终 12 个用例全部通过：

| 问题 | 实际现象 | 原因 | 修复 |
|------|----------|------|------|
| tied LM Head 找不到常量 | CLI 首次运行抛 `KeyError: 'lm_head.weight'` | tied 模式只登记 `embedding.weight`，`_add_lm_head()` 固定查找 `lm_head.weight` | 按 `tie_word_embeddings` 选择 `embedding.weight` 或 `lm_head.weight` |
| 层内顺序断言失败 | 首次测试 `indices` 为 `[2, 3, 7, 10, ...]`，与创建顺序不同 | `sort_op()` 从输出反向 DFS，执行序不等于创建序 | 测试改为断言数据依赖顺序；设计文档待确认后同步说明创建序与执行序的区别 |
| Transformers 形状对照 KeyError | `KeyError: 'lm_head.weight'` | PyTorch `named_parameters()` 默认去重 tied 参数 | 改用 `named_parameters(remove_duplicate=False)` |
| 首层 norm 输入断言失败 | `src=[layer.0.attn_norm.out,...]` 与实际不符 | 首层 norm 输入应为 `embedding.out` | 修正断言为 `src=[embedding.out,...]` |

## 8. 未覆盖项与风险

| 项目 | 说明 |
|------|------|
| 真实权重 | 本任务只登记元信息，`data_ptr=0`，未验证权重内容 |
| Shape/Precision 推导 | `Graph` 不推导激活形状，测试只验证连接关系 |
| 数值正确性 | 未运行 kernel，无法验证 logits 数值 |
| KV Cache | 本图是 prefill 图，不含 past/present KV cache |
| Padding Mask | 只建模 causal mask，不含 padding mask |
| 其他精度 | 代码声明支持 float16/float32，但字节数测试只覆盖 bfloat16 |
| 平台覆盖 | 只在 Windows 11 + Python 3.13.3 验证 |
| 线程安全 | 未测试并发构图；Graph 本身不保证线程安全 |

## 9. 结论

- Qwen3-0.6B 默认图包含 2 个输入、310 个 constant、426 个 op、1 个输出；
- 28 个 Decoder Layer 每层 15 个 op，连接关系与 Transformers 5.9.0
  `modeling_qwen3.py` 一致；
- 权重形状与 `Qwen3ForCausalLM` meta 模型逐项一致；
- tied 与 untied LM Head 两条路径均通过测试；
- 12 个 `unittest` 用例全部通过，命令行构图和完整 IR 输出正常。
