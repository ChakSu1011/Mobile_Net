# Runtime Common 基础工具设计（json / sha256 / file_util）

## 1. 背景与目标

### 1.1 背景

Runtime Loader 需要在没有第三方依赖的前提下完成模型包加载：

- 解析 `model.mgf` 中的 canonical JSON manifest（含 null、整数/浮点区分）；
- 校验 manifest、`weights.bin` 与 IR 文本的 SHA-256；
- 在 1 GB 级权重文件上做区间读取与流式分块读取。

`Runtime/common/` 已有 `status/data_types/model_config/runtime_config/
resource_stack` 等契约文件，但缺少 JSON、哈希与文件读取实现。本设计覆盖
新增的 `json.*`、`sha256.*`、`file_util.*` 三个基础工具模块。

### 1.2 目标

1. 提供严格、可定位错误的 JSON 解析器，满足 manifest 解析需要。
2. 提供流式 SHA-256，输出与 Python `hashlib.sha256(...).hexdigest()` 一致。
3. 提供文件大小、区间读取、顺序分块读取与文件哈希能力，支持 >4 GB 文件。
4. 全部实现只依赖 C++17 标准库，不引入第三方库，不抛出异常。

### 1.3 非目标

- 不实现 JSON 序列化、JSON Pointer、注释、尾逗号或 JSON5 扩展。
- 不实现内存映射、异步 IO、文件写入或目录遍历。
- 不校验未转义非 ASCII 字节的 UTF-8 合法性（manifest 由可信工具链生成，
  且 manifest SHA-256 先行校验）。
- 不实现 SHA-224/384/512 或硬件加速。

## 2. 需求分析

### 2.1 功能需求

| 编号 | 需求 | 验收标准 |
| --- | --- | --- |
| CM-JSON-01 | 支持 object/array/string/number/bool/null | 解析 Qwen3 manifest 全量成功 |
| CM-JSON-02 | 数字按语法区分整数（I64）与浮点（FP64） | `1`→I64，`1.0`/`1e-3`→FP64；与 Python json.loads 类型一致 |
| CM-JSON-03 | 字符串支持 RFC 8259 转义与 `\uXXXX` 代理对 | `\" \\ \/ \b \f \n \r \t`、代理对编码为 UTF-8 |
| CM-JSON-04 | 错误信息携带字节偏移 | 非法 JSON 返回含 `offset N` 的 kInvalidArgument |
| CM-JSON-05 | 对象重复键、非法控制字符、整数/浮点溢出、尾随数据报错 | 对应用例返回错误 |
| CM-SHA-01 | 流式 SHA-256，输出 64 字符小写十六进制 | 与 manifest/weights/IR 的 Python 哈希一致 |
| CM-FILE-01 | 查询文件大小 | 支持 1.19 GB 文件 |
| CM-FILE-02 | 读取任意区间，内部按 8 MB 分块循环 | 短读返回 kIoError |
| CM-FILE-03 | 顺序分块读取，返回实际字节数 | 文件尾返回 0 |
| CM-FILE-04 | 流式计算文件 SHA-256 | 与 CM-SHA-01 组合 |

### 2.2 非功能需求

| 编号 | 需求 |
| --- | --- |
| CM-NF-01 | 不抛出异常；可预期错误全部通过 `Status` 返回 |
| CM-NF-02 | 单线程使用；不做内部锁 |
| CM-NF-03 | 解析深度上限 256，防止恶意输入耗尽调用栈 |
| CM-NF-04 | 峰值额外内存不超过调用方指定的块大小；解析结果与输入同量级 |
| CM-NF-05 | 代码遵守 Google C++ 规范，使用 `U8/I32/U64/Usize/C8` 别名 |
| CM-NF-06 | 每个文件/类/函数提供中文注释（功能边界、输入输出、主要逻辑） |

### 2.3 约束条件

- 只写 `Runtime/common/` 下的实现文件，不修改已有公共头。
- 编译器为 C++17（本机 `python -m ziglang c++`，clang 22）。
- 路径按平台本地编码解释；本机测试路径为 ASCII。

### 2.4 验收标准

- AC-01：Qwen3 manifest 解析成功，统计字段与 JSON 一致。
- AC-02：manifest SHA-256、weights SHA-256、IR SHA-256 全部与 Python 一致。
- AC-03：非法 JSON 输出带偏移的错误。
- AC-04：`-Wall -Wextra` 零警告。
- 测试记录见 `Runtime/loader/tests/test_results.md`。

## 3. 概要设计

### 3.1 模块划分

| 文件 | 职责 | 依赖 |
| --- | --- | --- |
| `json.h/.cc` | 严格 JSON 解析与只读访问 | status.h |
| `sha256.h/.cc` | 流式 SHA-256 | 无 |
| `file_util.h/.cc` | 文件大小/区间/顺序读取/文件哈希 | status.h、sha256.h |

### 3.2 关键流程

```text
Loader
  ├─ GetFileSize(model.mgf)
  ├─ ReadFileRange(112B 头)
  ├─ ReadFileRange(manifest)
  ├─ Sha256::Hex(manifest)  ← 与头部 manifest_sha256 比较
  ├─ ParseJson(manifest) → JsonValue 树 → manifest 视图
  ├─ HashFileSha256(weights.bin) → 与头部 weights_sha256 比较
  └─ FileReader::ReadAt(entry.offset, entry.byte_size)
```

### 3.3 技术选型与取舍

| 选择 | 理由 | 备选与放弃原因 |
| --- | --- | --- |
| 自研递归下降 JSON | 无第三方依赖；可精确控制整数/浮点类型与错误偏移 | nlohmann/json 引入大依赖且默认数字类型不满足阶段契约 |
| 对象成员用 `vector<JsonMember>` | 保留插入顺序；`std::vector` 支持不完整类型 | `std::map<string,JsonValue>` 在 C++17 不保证支持不完整类型 |
| 自研 SHA-256 | 与 Python 输出一致；无 OpenSSL 依赖 | OpenSSL/CNG 增加部署依赖 |
| `std::ifstream` + 8 MB 循环 | 支持大文件、实现简单 | mmap 跨平台差异大；一次读入 1.19 GB 峰值过高 |

## 4. 详细设计

### 4.1 JsonValue

```cpp
enum class Type : U8 { kNull, kBool, kNumber, kString, kArray, kObject };
class JsonValue {
  Type type_; Bool bool_value_; Bool is_integer_;
  I64 integer_; FP64 number_;
  std::string string_value_;
  std::vector<JsonValue> array_value_;
  std::vector<JsonMember> object_value_;  // JsonMember{string key; JsonValue value;}
};
```

- 数字统一为 `kNumber`，`IsInteger()` 区分 I64/FP64；`AsFloat64()` 对整数
  做无损提升。
- `AsString/AsArray/AsObject` 类型不匹配时返回静态空对象，调用方先用
  `Is*` 判断；`Find` 对非对象返回 nullptr。
- `JsonMember` 在 `JsonValue` 之后定义；`JsonValue` 的特殊成员函数在
  `json.cc` 中实现，保证 `std::vector<JsonMember>` 实例化时类型完整。

### 4.2 解析算法

```text
ParseValue:
  '{' → ParseObject；'[' → ParseArray；'"' → ParseString
  '-'/digit → ParseNumber；true/false/null → 字面量
  其它 → "unexpected character"
```

- **数字**：严格匹配 `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`；
  无小数/指数用 `strtoll` 并要求无溢出；否则用 `strtod` 并要求有限。
- **字符串**：拒绝 `<0x20` 控制字符；`\u` 高代理必须紧跟低代理，合并后
  编码 UTF-8；孤立低代理报错。
- **对象**：重复键在解析期拒绝；尾随逗号被 "expected ',' or '}'" 拒绝。
- **深度**：超过 256 报错。
- **尾随数据**：顶层值后除空白外还有字符时报错。

### 4.3 Sha256

- 状态：8×U32 初始值、64 字节缓冲、总字节数、finalized 标记、摘要缓存。
- `Update`：先补满缓冲，再直接压缩完整块，尾部留存。
- `Finish`：追加 `0x80`，补零到 56 mod 64，写入 64 位大端比特长度，
  压缩并输出大端摘要。
- `HexDigest`：小写十六进制，与 Python `hexdigest()` 一致。
- `Final` 幂等；`Final` 后 `Update` 为空操作。

### 4.4 FileReader 与文件函数

| 接口 | 主要逻辑 | 错误码 |
| --- | --- | --- |
| `GetFileSize` | 二进制打开，seek end + tellg | kIoError |
| `ReadFileRange` | 打开文件并委托 `ReadAt` | kIoError/kInvalidArgument |
| `Open` | 打开、取大小、校验 seek | kIoError/kInvalidArgument |
| `ReadAt` | 区间上界用 `offset > size_ || length > size_ - offset` 防溢出；8 MB 分块 `seekg/read`，检查 `gcount` | kInvalidArgument/kIoError |
| `ReadNext` | 从 `position_` 顺序读取，`position_ += want` | kStateError/kIoError |
| `HashFileSha256` | `ReadNext` 循环 + `Sha256::Update` | kIoError |

`ReadAt` 不修改顺序读取位置，`ReadNext` 每次显式 seek 到 `position_`，
两者互不干扰。

### 4.5 异常与错误处理

- 所有函数返回 `Status`；JSON 错误统一为 `kInvalidArgument`，文件错误为
  `kIoError`，参数错误为 `kInvalidArgument`。
- 错误消息包含路径、偏移或区间，便于定位损坏包。
- 不通过异常传播错误；字符串/容器操作使用标准库（bad_alloc 视为进程级）。

### 4.6 性能与内存

- JSON 解析 O(n)，对象按键查找 O(m)，manifest 规模（385 KB）可忽略。
- SHA-256 纯 C++ 实现，1.19 GB 文件全量哈希与逐项哈希合计约数秒。
- 文件读取单次块 8 MB；1.19 GB 权重按条目读入（最大条目 311 MB），
  峰值由调用方控制。

## 5. 实施步骤

| 阶段 | 任务 | 依赖 |
| --- | --- | --- |
| S1 | 实现 sha256.h/.cc，按已知向量自检 | 无 |
| S2 | 实现 json.h/.cc 与错误偏移 | status.h |
| S3 | 实现 file_util.h/.cc（大小/区间/顺序/哈希） | S1 |
| S4 | 由 Loader 集成 json/sha256/file_util | S1–S3 |
| S5 | 编译（`-Wall -Wextra`）与冒烟/负向测试 | S4 |

风险与应对：

- 数字类型与 Python 不一致 → 用 manifest 全量属性回归（IR 哈希比对会失败）。
- 大文件偏移溢出 → 区间检查使用减法形式，避免 `offset+length` 溢出。
- 非 ASCII 路径 → 本机测试路径为 ASCII；文档标注限制。

## 6. 测试与验证设计

### 6.1 测试策略

- 单元级：由 `loader_smoke` 间接触发 JSON/SHA/文件读取全路径。
- 集成级：真实 Qwen3 包 manifest/IR/weights 三类哈希与 Python 参考一致。
- 负向：损坏 magic、非法 JSON（带偏移）、错误权重尺寸、错误 IR 哈希。

### 6.2 用例

| 用例 | 输入 | 通过标准 | 追溯 |
| --- | --- | --- | --- |
| TC-CM-01 | Qwen3 manifest | 解析成功，统计一致 | CM-JSON-01/02/03 |
| TC-CM-02 | 篡改末尾 `}` 为 `]` | 返回 JSON 错误并给出 offset | CM-JSON-04 |
| TC-CM-03 | Qwen3 manifest/weights/IR | 三类 SHA 与 Python 一致 | CM-SHA-01 |
| TC-CM-04 | 1.19 GB weights.bin | 区间读取成功，8.7 s 内完成全量加载 | CM-FILE-01/02/04 |

### 6.3 测试环境与通过标准

- 环境：Windows 11、zig clang 22（`x86_64-windows-gnu`）、C++17、-O2。
- 通过标准：全部用例退出码符合预期，`-Wall -Wextra` 零警告。
- 实际命令、输出与未覆盖项见 `Runtime/loader/tests/test_results.md`。
