// 模块：Runtime 模型包 manifest 视图
//
// 职责：
//   解析并校验 model.mgf 固定头（112 字节）、manifest JSON、图元数据
//   （constants/inputs/ops/outputs/tensors）、权重表、activation 显存计划、
//   统计信息与 ModelConfig；向 Loader 提供只读视图。
//
// 边界：
//   - 只做格式与引用校验，不创建 Graph、不分配缓冲、不读取 weights.bin；
//   - 不写文件、不修改模型包；
//   - manifest 中的 symbol 维度、null 属性、重复名称与越界偏移一律拒绝；
//   - 所有精度/布局/op 类型名称转换是单向的（字符串→枚举），名称集合
//     必须与 ToolChain 序列化格式一致。
//
// 依赖：
//   Graph/include/{attr,op_type,types}.h、Runtime/common/{data_types,json,
//   model_config,status}.h、Utils/cpp/types.h、C++17 标准库。

#ifndef MOBILENET_RUNTIME_LOADER_MANIFEST_H_
#define MOBILENET_RUNTIME_LOADER_MANIFEST_H_

#include <array>
#include <map>
#include <string>
#include <vector>

#include "Graph/include/attr.h"
#include "Graph/include/op_type.h"
#include "Graph/include/types.h"
#include "Runtime/common/data_types.h"
#include "Runtime/common/json.h"
#include "Runtime/common/model_config.h"
#include "Runtime/common/status.h"
#include "Utils/cpp/types.h"

namespace mobilenet {
namespace runtime {

// model.mgf 固定头字节数。
inline constexpr Usize kModelPackageHeaderSize = 112;

// model.mgf 容器版本。
inline constexpr U32 kModelPackageFormatVersion = 1;

// 头部 flags 中表示存在 weights.bin 的位。
inline constexpr U32 kModelPackageFlagWeights = 0x1;

// 头部魔数 "MNETMGF\0"。
inline constexpr C8 kModelPackageMagic[8] = {'M', 'N', 'E', 'T',
                                             'M', 'G', 'F', '\0'};

// 模型包固定头。
//
// 功能：
//   对应 struct "<8sIIQQ32sQQ32s"：magic/version/flags/manifest 偏移与长度/
//   manifest SHA-256/weights 偏移与长度/weights SHA-256。
//
// 边界：
//   - 只保存解析结果，不持有文件；weights 位于独立 weights.bin；
//   - has_weights 由 flags bit0 与 weights_length 共同校验。
struct ModelPackageHeader {
  // 容器版本，必须为 1。
  U32 version = 0;

  // flags 位标记；当前只允许 bit0。
  U32 flags = 0;

  // manifest 在 model.mgf 中的起始偏移。
  U64 manifest_offset = 0;

  // manifest 字节数。
  U64 manifest_length = 0;

  // manifest 字节的 SHA-256。
  std::array<U8, 32> manifest_sha256 = {};

  // weights.bin 内的逻辑偏移；无权重时为 0。
  U64 weights_offset = 0;

  // weights.bin 字节数；无权重时为 0。
  U64 weights_length = 0;

  // weights.bin 的 SHA-256。
  std::array<U8, 32> weights_sha256 = {};

  // 是否存在 weights.bin。
  Bool has_weights = false;
};

// 把 32 字节摘要格式化为小写十六进制。
//
// 参数：
//   digest：二进制摘要。
// 返回：
//   64 字符小写十六进制字符串。
std::string Sha256BytesToHex(const std::array<U8, 32>& digest);

// 解析并校验 112 字节固定头。
//
// 参数：
//   data：至少 kModelPackageHeaderSize 字节的头部数据；
//   size：data 的有效字节数；
//   file_size：model.mgf 总字节数，用于检查 manifest 边界；
//   header：输出参数，非空。
// 返回：
//   成功返回 kOk；magic/version/flags/偏移非法时返回 kInvalidModelFile。
Status ParseModelPackageHeader(const U8* data, Usize size, U64 file_size,
                               ModelPackageHeader* header);

// 读取 model.mgf 头部。
//
// 参数：
//   model_file：model.mgf 路径；
//   header：输出参数，非空。
// 返回：
//   成功返回 kOk；文件读取失败返回 kIoError；头部非法返回
//   kInvalidModelFile。
Status ReadModelPackageHeader(const std::string& model_file,
                              ModelPackageHeader* header);

// manifest 张量类别。
enum class TensorKind : U8 {
  kInput = 0,
  kConstant = 1,
  kActivation = 2,
  kOutput = 3,
};

// manifest 张量元数据。
//
// 边界：
//   - dims 必须是静态整数维度；symbol 维度在解析期拒绝；
//   - pitch 可为空，空表示与 dims 相同；
//   - byte_size 为 manifest 声明的目标精度字节数。
struct TensorMeta {
  // 张量名，图内唯一。
  std::string name;

  // 张量类别。
  TensorKind kind = TensorKind::kActivation;

  // 逻辑维度。
  std::vector<I64> dims;

  // 物理 pitch；空表示等于 dims。
  std::vector<I64> pitch;

  // 目标精度。
  runtime_graph::Precision precision = runtime_graph::Precision::kUnknown;

  // 布局；首期只允许 row_major。
  runtime_graph::Layout layout = runtime_graph::Layout::kRowMajor;

  // 目标精度字节数。
  U64 byte_size = 0;

  // manifest 是否声明该常量有可用权重。
  Bool weight_available = false;
};

// 图输出绑定。
struct IOBinding {
  // 输出 op 名称。
  std::string name;

  // 被输出的 activation 张量名。
  std::string tensor;
};

// manifest 中的算子元数据。
//
// 边界：
//   - attrs 已转换为 Graph AttrMap；null/嵌套对象等不可表示的值已拒绝；
//   - inputs/outputs 是张量名列表，顺序即构图顺序。
struct OpMeta {
  // op 名称，图内唯一。
  std::string name;

  // 稳定 op 类型名，例如 "sample"。
  std::string op_type;

  // 输入张量名列表，非空。
  std::vector<std::string> inputs;

  // 输出张量名列表，非空。
  std::vector<std::string> outputs;

  // 静态属性。
  runtime_graph::AttrMap attrs;
};

// 图元数据视图。
struct GraphManifest {
  // 图名称。
  std::string name;

  // 模型Adapter名称，例如 "qwen3"。
  std::string adapter;

  // 输入张量名（创建顺序）。
  std::vector<std::string> inputs;

  // 常量张量名（创建顺序）。
  std::vector<std::string> constants;

  // 输出绑定。
  std::vector<IOBinding> outputs;

  // 全部张量元数据，按名称索引。
  std::map<std::string, TensorMeta> tensors;

  // 计算 op（创建顺序，不含 Input/Output）。
  std::vector<OpMeta> ops;

  // manifest 记录的 IR SHA-256。
  std::string ir_sha256;

  // manifest 记录的 IR 行数；0 表示未声明。
  I64 ir_line_count = 0;
};

// 单个权重条目。
struct WeightEntry {
  // 常量张量名。
  std::string name;

  // 源权重名；可为空。
  std::string source;

  // 权重形状。
  std::vector<I64> shape;

  // 源精度名称，例如 "float16"。
  std::string dtype;

  // 源字节数。
  U64 byte_size = 0;

  // weights.bin 内偏移；has_offset=false 时无意义。
  U64 offset = 0;

  // 是否包含有效 offset。
  Bool has_offset = false;

  // 源字节 SHA-256；可为空表示未声明。
  std::string sha256;

  // 是否已打包到 weights.bin。
  Bool available = false;

  // 不可用原因；可用时为空。
  std::string reason;
};

// 权重表。
struct WeightTable {
  // weights 文件名，默认 "weights.bin"。
  std::string file;

  // 权重偏移对齐字节数。
  U64 alignment = 0;

  // weights.bin 总字节数。
  U64 total_bytes = 0;

  // 权重条目。
  std::vector<WeightEntry> entries;

  // 按名称查找权重条目。
  //
  // 参数：
  //   name：常量张量名。
  // 返回：
  //   找到返回非空指针；否则返回 nullptr。
  const WeightEntry* Find(const std::string& name) const;
};

// activation 显存计划条目。
struct MemoryPlanEntry {
  // activation 张量名。
  std::string name;

  // arena 内字节偏移。
  U64 offset = 0;

  // 字节数。
  U64 size = 0;

  // 首次使用步序号。
  I64 birth = 0;

  // 最后使用步序号。
  I64 last_use = 0;

  // 是否允许复用。
  Bool reusable = true;
};

// activation 显存计划。
struct MemoryPlan {
  // 对齐字节数，必须大于 0。
  U64 alignment = 0;

  // arena 高水位字节数。
  U64 arena_bytes = 0;

  // 理论峰值字节数。
  U64 peak_bytes = 0;

  // 常量区字节数。
  U64 constants_bytes = 0;

  // 逐张量条目。
  std::vector<MemoryPlanEntry> entries;

  // 按张量名查找计划条目。
  //
  // 参数：
  //   name：activation 张量名。
  // 返回：
  //   找到返回非空指针；否则返回 nullptr。
  const MemoryPlanEntry* Find(const std::string& name) const;
};

// manifest 统计信息。
struct ManifestStatistics {
  // 张量总数。
  U64 tensors = 0;

  // 计算 op 数。
  U64 ops = 0;

  // 常量数。
  U64 constants = 0;

  // 输出数。
  U64 outputs = 0;

  // activation 张量数。
  U64 activation_tensors = 0;

  // 常量字节数。
  U64 constant_bytes = 0;
};

// 完整 manifest 视图。
//
// 边界：
//   - model_config 与 graph_metadata 保留为 JsonValue，由
//     ParseModelConfig/上层按需读取；
//   - 所有引用完整性已在 ParseManifest 中校验。
struct ModelManifest {
  // 格式名，必须为 "MobileNet.ModelPackage"。
  std::string format;

  // 格式版本，必须为 1。
  I64 format_version = 0;

  // 模型名。
  std::string model_name;

  // 模型Adapter。
  std::string model_adapter;

  // model.config 原始对象。
  JsonValue model_config;

  // graph.metadata 原始对象。
  JsonValue graph_metadata;

  // 图元数据。
  GraphManifest graph;

  // optimization.config 原始对象。
  JsonValue optimization_config;

  // activation 显存计划。
  MemoryPlan memory_plan;

  // 权重表。
  WeightTable weights;

  // 统计信息。
  ManifestStatistics statistics;
};

// 解析 manifest JSON 文本。
//
// 参数：
//   text：canonical JSON 文本；
//   manifest：输出参数，非空。
// 返回：
//   成功返回 kOk；JSON 非法、字段缺失、类型错误、引用不一致或重复名称
//   返回 kInvalidModelFile，message 含具体位置。
Status ParseManifest(const std::string& text, ModelManifest* manifest);

// 从 manifest 提取 ModelConfig。
//
// 功能：
//   结构参数来自 graph.metadata.model_config；context_length/max_num_tokens/
//   batch/pad/bos/stop ids 来自 graph.metadata.inference_config；目标精度
//   来自 optimization.config.target_precision（不使用官方 bf16 声明）。
//
// 参数：
//   manifest：已解析 manifest；
//   config：输出参数，非空。
// 返回：
//   成功返回 kOk；必需字段缺失/非法返回 kInvalidModelFile。
Status ParseModelConfig(const ModelManifest& manifest, ModelConfig* config);

// 精度名称转 Graph Precision。
//
// 参数：
//   name：稳定名称，例如 "float16"；
//   precision：输出参数，非空。
// 返回：
//   成功返回 kOk；未知名称返回 kInvalidModelFile。
Status PrecisionFromName(const std::string& name,
                         runtime_graph::Precision* precision);

// 精度名称转后端 DataType。
//
// 参数：
//   name：稳定名称，例如 "float16"；
//   dtype：输出参数，非空。
// 返回：
//   成功返回 kOk；未知或后端不支持的名称返回 kInvalidModelFile。
Status DataTypeFromName(const std::string& name, DataType* dtype);

// 布局名称转 Graph Layout。
//
// 参数：
//   name："row_major" 或 "column_major"；
//   layout：输出参数，非空。
// 返回：
//   成功返回 kOk；未知名称返回 kInvalidModelFile。
Status LayoutFromName(const std::string& name,
                      runtime_graph::Layout* layout);

// op 类型名称转 Graph OpType。
//
// 参数：
//   name：稳定名称，例如 "mega_attention"；
//   op_type：输出参数，非空。
// 返回：
//   成功返回 kOk；未知名称返回 kInvalidModelFile。
Status OpTypeFromName(const std::string& name,
                      runtime_graph::OpType* op_type);

}  // namespace runtime
}  // namespace mobilenet

#endif  // MOBILENET_RUNTIME_LOADER_MANIFEST_H_
