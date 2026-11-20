# Runtime Common 开发 Skill

## 1. 适用场景

- 修改 `Runtime/common/` 的公共数据对象、错误码、类型别名、JSON/SHA-256/
  文件工具；
- 为 Loader/Engine/Scheduler 等模块提供跨模块契约；
- 排查 manifest 解析、哈希校验、文件区间读取问题。

## 2. 输入

- `Runtime/.doc/Runtime_design.md` 与 `Runtime/common/.doc/common_design.md`；
- `Utils/cpp/types.h`；
- 模型包 `model.mgf` / `weights.bin`；
- memory.md（Google Style、中文注释、四件套）。

## 3. 步骤

1. 类型：所有 Host 端类型用 `Utils/cpp/types.h` 别名，不写原始类型名。
2. 数据契约：Request/Response/Sequence 在 `sequence.h`；计划/结果在
   `plan.h`；错误码在 `status.h`；配置在 `runtime_config.h`/`model_config.h`。
3. JSON：使用 `json.h`，严格 object/array/string/number/bool/null；数字区分
   I64/FP64；错误带 offset。
4. SHA：使用 `sha256.h` 流式接口；大文件不要一次性读入。
5. 文件：使用 `file_util.h` 的区间/分块接口，校验 offset+length 防溢出。
6. 测试：优先用小 TU 编译 `-Wall -Wextra`，再用 Loader smoke 验证真实
   manifest；记录实际命令与输出。

## 4. 输出

- `common/*.h/.cc`；
- `common/.doc/common_design.md`；
- `common/.skill/SKILL.md`；
- `common/tests/test_results.md`（真实输出）。

## 5. 常见陷阱

1. JSON 中 `null` 存在于 metadata，不能在解析器里直接拒绝；
2. op attrs 是扁平标量/数组，遇到嵌套对象应报错而不是静默转换；
3. 数字 1 与 1.0 在 AttrValue 中分别是 I64 与 FP64，dump_ir 必须一致；
4. SHA-256 必须流式处理 weights，避免 1.19GB 整读；
5. 文件区间读取要检查 length 与文件大小，防止溢出；
6. 别名不得在其它头文件重复定义；
7. 错误码只追加，不重编号。

## 6. 检查清单

- [ ] 所有新类型使用 Utils 别名；
- [ ] JSON/SHA/文件工具无第三方依赖；
- [ ] 错误带 offset/路径，可定位；
- [ ] 编译 `-Wall -Wextra` 无警告；
- [ ] 文档/skill/测试结果同步。
