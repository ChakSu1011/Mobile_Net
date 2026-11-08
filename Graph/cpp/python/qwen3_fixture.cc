// 模块：Qwen3-0.6B 参考图构建器实现
//
// 职责：
//   按 Qwen3-0.6B config.json 与 model.safetensors 的参数命名，构造 28 层
//   Decoder Block 的融合算子图。权重与设备边使用 meta tensor，输入/输出
//   host_tensor 使用小型 CPU tensor，避免加载 1.5 GB 真实权重。
//
// 边界：
//   - 不读取权重数值，只使用形状与 dtype 元数据；
//   - 不执行任何 Kernel 或 RuntimeGraph 绑定；
//   - 节点、边、权重的插入顺序即为 Canonical JSON 的比较基准，修改顺序
//     必须同步修改 Python 参考构建器。
//
// 依赖：
//   graph_builder.h、tensor_desc.h、torch_bridge.h、Python torch。

#include "qwen3_fixture.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph/graph_builder.h"
#include "graph/tensor_desc.h"
#include "torch_bridge.h"

namespace mobilenet {
namespace graph {
namespace {

// Qwen3-0.6B 超参数；来源：D:\mLLM\models\Qwen3-0.6B\config.json。
constexpr int32_t kNumLayers = 28;
constexpr int32_t kHiddenSize = 1024;
constexpr int32_t kIntermediateSize = 3072;
constexpr int32_t kNumAttentionHeads = 16;
constexpr int32_t kNumKvHeads = 8;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kVocabSize = 151936;
constexpr int32_t kMaxContextLength = 40960;
constexpr double kNormEps = 1e-6;
constexpr double kRopeTheta = 1000000.0;
constexpr int64_t kBatchExample = 1;
constexpr int64_t kSequenceExample = 8;

// 构造失败异常；在 BuildQwen3ReferenceGraph 边界转换为 Status。
class BuildFailure : public std::runtime_error {
 public:
  BuildFailure(ErrorCode code, const std::string& message)
      : std::runtime_error(message), code_(code) {}

  ErrorCode code() const { return code_; }

 private:
  ErrorCode code_;
};

// 校验状态，失败抛 BuildFailure。
void Require(const Status& status) {
  if (!status.ok()) {
    throw BuildFailure(status.code(), status.message());
  }
}

// 构造 Python 形状元组。
py::tuple MakePyShape(const std::vector<int64_t>& shape) {
  py::list values;
  for (int64_t dim : shape) {
    values.append(py::int_(dim));
  }
  return py::tuple(values);
}

// 在 meta device 上创建张量；只使用元数据，不分配内存。
at::Tensor MakeMetaTensor(const std::vector<int64_t>& shape,
                          at::ScalarType dtype) {
  const py::object tensor = py::module_::import("torch").attr("empty")(
      MakePyShape(shape), py::arg("dtype") = ScalarTypeToPython(dtype),
      py::arg("device") = "meta");
  return TensorFromPython(tensor);
}

// 在 CPU 上创建小型 I/O host tensor。
at::Tensor MakeCpuTensor(const std::vector<int64_t>& shape,
                         at::ScalarType dtype) {
  const py::object tensor = py::module_::import("torch").attr("empty")(
      MakePyShape(shape), py::arg("dtype") = ScalarTypeToPython(dtype),
      py::arg("device") = "cpu");
  return TensorFromPython(tensor);
}

// 全常量形状。
SymbolicShape ConcreteShape(const std::vector<int64_t>& values) {
  SymbolicShape shape;
  shape.dims.reserve(values.size());
  for (int64_t value : values) {
    shape.dims.push_back(Dim::Constant(value));
  }
  return shape;
}

// [batch, sequence, hidden]。
SymbolicShape HiddenShape() {
  return SymbolicShape({Dim::Symbolic("batch", kBatchExample),
                        Dim::Symbolic("sequence", kSequenceExample),
                        Dim::Constant(kHiddenSize)});
}

// [batch, sequence, hidden_size] 之外的中间激活形状。
SymbolicShape SequenceShape(int64_t last_dimension) {
  return SymbolicShape({Dim::Symbolic("batch", kBatchExample),
                        Dim::Symbolic("sequence", kSequenceExample),
                        Dim::Constant(last_dimension)});
}

// logits 形状 [batch, vocab]。
SymbolicShape LogitsShape() {
  return SymbolicShape(
      {Dim::Symbolic("batch", kBatchExample), Dim::Constant(kVocabSize)});
}

// 构造 Qwen3-0.6B 模型配置；数值与 config.json 一致。
ModelConfig MakeQwen3ModelConfig() {
  ModelConfig config;
  config.model_family = "qwen3";
  config.num_layers = kNumLayers;
  config.hidden_size = kHiddenSize;
  config.intermediate_size = kIntermediateSize;
  config.num_attention_heads = kNumAttentionHeads;
  config.num_kv_heads = kNumKvHeads;
  config.head_dim = kHeadDim;
  config.vocab_size = kVocabSize;
  config.max_context_length = kMaxContextLength;
  config.norm_eps = kNormEps;
  config.activation = "silu";
  config.tie_word_embeddings = true;
  config.dtype = at::ScalarType::BFloat16;
  config.rope.theta = kRopeTheta;
  config.rope.layout = RopeLayout::kNeox;
  config.rope.scaling_type = "none";
  config.rope.scaling_factor = 1.0;
  config.rope.original_max_position_embeddings = kMaxContextLength;
  return config;
}

// 构造融合前的原始算子。
FusedOp MakeFusedOp(std::string source_name, std::string op_kind) {
  FusedOp fused_op;
  fused_op.source_name = std::move(source_name);
  fused_op.op_kind = std::move(op_kind);
  return fused_op;
}

// 生成层内名称/权重引用前缀。
std::string LayerPrefix(int32_t layer) {
  return "layer." + std::to_string(layer);
}

// 生成实际权重 source_name。
std::string LayerSource(int32_t layer, std::string_view suffix) {
  return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

// Qwen3 参考图装配器。
//
// 使用固定顺序完成权重、节点、边添加，保证 Canonical JSON 与 Python 侧
// 参考构建器逐字节可比。
class Qwen3GraphAssembler {
 public:
  // 构建参考图。
  //
  // 返回：成功返回非空 Graph；失败抛 BuildFailure。
  std::unique_ptr<Graph> Build() {
    Require(builder_.SetModelConfig(MakeQwen3ModelConfig()));
    BuildWeights();
    BuildNodes();
    BuildEdges();
    Require(builder_.SetInputNodes(input_node_ids_));
    Require(builder_.SetOutputNodes(output_node_ids_));
    std::unique_ptr<Graph> graph;
    Require(builder_.Finalize(&graph));
    return graph;
  }

 private:
  // 添加权重并记录张量，供权重边共享存储。
  void AddWeight(const std::string& ref, const std::string& source_name,
                 const std::vector<int64_t>& shape) {
    AddWeightTensor(ref, source_name,
                    MakeMetaTensor(shape, at::ScalarType::BFloat16));
  }

  // 使用已有张量添加权重（用于 embedding/lm_head tie）。
  void AddWeightTensor(const std::string& ref, const std::string& source_name,
                       const at::Tensor& tensor) {
    weights_.emplace(ref, tensor);
    Require(builder_.AddWeight(WeightDesc(ref, source_name, tensor)));
  }

  // 添加节点并返回 NodeId。
  NodeId AddNode(std::string name, OpType op_type,
                 std::vector<FusedOp> fused_ops,
                 std::map<std::string, AttributeValue> attributes,
                 std::vector<std::string> input_edges,
                 std::vector<std::string> output_edges,
                 at::ScalarType precision, std::string kernel_key,
                 std::string fusion_rule_id = "", std::string binding_name = "",
                 CopyKind copy_kind = CopyKind::kNone,
                 std::optional<at::Tensor> host_tensor = std::nullopt) {
    NodeId node_id = kInvalidNodeId;
    Require(builder_.AddNode(
        Node(std::move(name), op_type, std::move(fused_ops),
             std::move(attributes), std::move(input_edges),
             std::move(output_edges), precision, std::move(kernel_key),
             std::move(fusion_rule_id), std::move(binding_name), copy_kind,
             std::move(host_tensor)),
        &node_id));
    return node_id;
  }

  // 添加激活边。
  void AddActivationEdge(const std::string& name, const SymbolicShape& shape,
                         at::ScalarType dtype) {
    const at::Tensor tensor = MakeMetaTensor(shape.ExampleShape(), dtype);
    Require(builder_.AddEdge(Edge(name, tensor, shape, std::nullopt, false, ""),
                             nullptr));
  }

  // 添加权重边，与权重共享同一 at::Tensor。
  void AddWeightEdge(const std::string& name, const std::string& weight_ref) {
    const at::Tensor tensor = weights_.at(weight_ref);
    Require(builder_.AddEdge(Edge(name, tensor, ConcreteSymbolicShape(tensor),
                                  std::nullopt, true, weight_ref),
                             nullptr));
  }

  // 添加全部权重，顺序固定。
  void BuildWeights() {
    AddWeight("embedding.weight", "model.embed_tokens.weight",
              {kVocabSize, kHiddenSize});
    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
      const std::string prefix = LayerPrefix(layer);
      AddWeight(prefix + ".attn_norm.weight",
                LayerSource(layer, "input_layernorm.weight"), {kHiddenSize});
      AddWeight(prefix + ".q.weight",
                LayerSource(layer, "self_attn.q_proj.weight"),
                {kNumAttentionHeads * kHeadDim, kHiddenSize});
      AddWeight(prefix + ".k.weight",
                LayerSource(layer, "self_attn.k_proj.weight"),
                {kNumKvHeads * kHeadDim, kHiddenSize});
      AddWeight(prefix + ".v.weight",
                LayerSource(layer, "self_attn.v_proj.weight"),
                {kNumKvHeads * kHeadDim, kHiddenSize});
      AddWeight(prefix + ".o.weight",
                LayerSource(layer, "self_attn.o_proj.weight"),
                {kHiddenSize, kNumAttentionHeads * kHeadDim});
      AddWeight(prefix + ".q_norm.weight",
                LayerSource(layer, "self_attn.q_norm.weight"), {kHeadDim});
      AddWeight(prefix + ".k_norm.weight",
                LayerSource(layer, "self_attn.k_norm.weight"), {kHeadDim});
      AddWeight(prefix + ".mlp_norm.weight",
                LayerSource(layer, "post_attention_layernorm.weight"),
                {kHiddenSize});
      AddWeight(prefix + ".gate.weight",
                LayerSource(layer, "mlp.gate_proj.weight"),
                {kIntermediateSize, kHiddenSize});
      AddWeight(prefix + ".up.weight", LayerSource(layer, "mlp.up_proj.weight"),
                {kIntermediateSize, kHiddenSize});
      AddWeight(prefix + ".down.weight",
                LayerSource(layer, "mlp.down_proj.weight"),
                {kHiddenSize, kIntermediateSize});
    }
    AddWeight("final_norm.weight", "model.norm.weight", {kHiddenSize});
    // tie_word_embeddings=true：LM Head 与 embedding 共享同一权重张量。
    AddWeightTensor("lm_head.weight", "lm_head.weight",
                    weights_.at("embedding.weight"));
  }

  // 添加全部节点，顺序固定。
  void BuildNodes() {
    input_node_ids_.push_back(
        AddNode("input.input_ids", OpType::kInput, {}, {}, {}, {"input_ids"},
                at::ScalarType::Long, "", "", "input_ids", CopyKind::kH2D,
                MakeCpuTensor({kBatchExample, kSequenceExample},
                              at::ScalarType::Long)));
    input_node_ids_.push_back(AddNode(
        "input.position_ids", OpType::kInput, {}, {}, {}, {"position_ids"},
        at::ScalarType::Long, "", "", "position_ids", CopyKind::kH2D,
        MakeCpuTensor({kBatchExample, kSequenceExample},
                      at::ScalarType::Long)));
    input_node_ids_.push_back(
        AddNode("input.last_token_indices", OpType::kInput, {}, {}, {},
                {"last_token_indices"}, at::ScalarType::Long, "", "",
                "last_token_indices", CopyKind::kH2D,
                MakeCpuTensor({kBatchExample}, at::ScalarType::Long)));

    AddNode("embedding", OpType::kEmbedding,
            {MakeFusedOp("model.embed_tokens", "embedding_lookup")}, {},
            {"input_ids", "weight.embedding"}, {"hidden"},
            at::ScalarType::BFloat16, "embedding");

    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
      const std::string prefix = LayerPrefix(layer);
      const std::string hidden_in =
          layer == 0 ? "hidden" : LayerPrefix(layer - 1) + ".hidden_out";
      const std::map<std::string, AttributeValue> norm_attributes = {
          {"eps", kNormEps}, {"precision", std::string("bfloat16")}};

      AddNode(prefix + ".attn_norm", OpType::kRmsNorm,
              {MakeFusedOp(LayerSource(layer, "input_layernorm"), "rmsnorm")},
              norm_attributes, {hidden_in, "weight." + prefix + ".attn_norm"},
              {prefix + ".attn_normed"}, at::ScalarType::BFloat16, "rms_norm",
              "qwen3.block.norm.attn.v1");

      AddNode(
          prefix + ".qkv", OpType::kQkvProjection,
          {MakeFusedOp(LayerSource(layer, "self_attn.q_proj"), "q_linear"),
           MakeFusedOp(LayerSource(layer, "self_attn.k_proj"), "k_linear"),
           MakeFusedOp(LayerSource(layer, "self_attn.v_proj"), "v_linear"),
           MakeFusedOp(LayerSource(layer, "self_attn.q_norm"), "qk_rmsnorm"),
           MakeFusedOp(LayerSource(layer, "self_attn.rotary_emb"), "rope")},
          {{"num_heads", static_cast<int64_t>(kNumAttentionHeads)},
           {"num_kv_heads", static_cast<int64_t>(kNumKvHeads)},
           {"head_dim", static_cast<int64_t>(kHeadDim)},
           {"rope_theta", kRopeTheta},
           {"rope_layout", std::string("neox")},
           {"has_qk_norm", true},
           {"qk_norm_eps", kNormEps},
           {"bias", false}},
          {prefix + ".attn_normed", "weight." + prefix + ".q",
           "weight." + prefix + ".k", "weight." + prefix + ".v",
           "weight." + prefix + ".q_norm", "weight." + prefix + ".k_norm",
           "position_ids"},
          {prefix + ".q", prefix + ".k", prefix + ".v"},
          at::ScalarType::BFloat16, "qkv_projection", "qwen3.block.qkv.v1");

      AddNode(
          prefix + ".attn", OpType::kAttention,
          {MakeFusedOp("kv_cache", "kv_cache_write"),
           MakeFusedOp(LayerSource(layer, "self_attn"), "qk_matmul"),
           MakeFusedOp(LayerSource(layer, "self_attn.softmax"), "softmax"),
           MakeFusedOp(LayerSource(layer, "self_attn"), "pv_matmul"),
           MakeFusedOp(LayerSource(layer, "self_attn.o_proj"),
                       "output_projection")},
          {{"causal", true},
           {"softmax_scale", 1.0 / std::sqrt(static_cast<double>(kHeadDim))},
           {"sliding_window", static_cast<int64_t>(0)},
           {"has_output_projection", true}},
          {prefix + ".q", prefix + ".k", prefix + ".v",
           "weight." + prefix + ".o"},
          {prefix + ".attn_out"}, at::ScalarType::BFloat16, "attention",
          "qwen3.block.attention.v1");

      AddNode(prefix + ".attn_residual", OpType::kResidualAdd,
              {MakeFusedOp(LayerPrefix(layer), "add")},
              {{"precision", std::string("bfloat16")}},
              {prefix + ".attn_out", hidden_in}, {prefix + ".hidden_mid"},
              at::ScalarType::BFloat16, "residual_add",
              "qwen3.block.residual.attn.v1");

      AddNode(prefix + ".mlp_norm", OpType::kRmsNorm,
              {MakeFusedOp(LayerSource(layer, "post_attention_layernorm"),
                           "rmsnorm")},
              norm_attributes,
              {prefix + ".hidden_mid", "weight." + prefix + ".mlp_norm"},
              {prefix + ".mlp_normed"}, at::ScalarType::BFloat16, "rms_norm",
              "qwen3.block.norm.mlp.v1");

      AddNode(prefix + ".mlp_gate_up", OpType::kMlpGateUpAct,
              {MakeFusedOp(LayerSource(layer, "mlp.gate_proj"), "gate_linear"),
               MakeFusedOp(LayerSource(layer, "mlp.up_proj"), "up_linear"),
               MakeFusedOp(LayerSource(layer, "mlp.act_fn"), "silu"),
               MakeFusedOp(LayerSource(layer, "mlp"), "mul")},
              {{"activation", std::string("silu")},
               {"intermediate_size", static_cast<int64_t>(kIntermediateSize)}},
              {prefix + ".mlp_normed", "weight." + prefix + ".gate",
               "weight." + prefix + ".up"},
              {prefix + ".mlp_act"}, at::ScalarType::BFloat16,
              "mlp_gate_up_act", "qwen3.block.gate_up.v1");

      AddNode(prefix + ".mlp_down_residual", OpType::kMlpDownResidual,
              {MakeFusedOp(LayerSource(layer, "mlp.down_proj"), "down_linear"),
               MakeFusedOp(LayerPrefix(layer), "add")},
              {{"precision", std::string("bfloat16")}},
              {prefix + ".mlp_act", "weight." + prefix + ".down",
               prefix + ".hidden_mid"},
              {prefix + ".hidden_out"}, at::ScalarType::BFloat16,
              "mlp_down_residual", "qwen3.block.down_residual.v1");
    }

    const std::string last_hidden = LayerPrefix(kNumLayers - 1) + ".hidden_out";
    AddNode("final_norm", OpType::kRmsNorm,
            {MakeFusedOp("model.norm", "rmsnorm")},
            {{"eps", kNormEps}, {"precision", std::string("bfloat16")}},
            {last_hidden, "weight.final_norm"}, {"final_normed"},
            at::ScalarType::BFloat16, "rms_norm");

    AddNode("lm_head", OpType::kLmHead,
            {MakeFusedOp("lm_head", "last_token_gather"),
             MakeFusedOp("lm_head", "linear")},
            {{"gather_last_token", true}},
            {"final_normed", "last_token_indices", "weight.lm_head"},
            {"logits"}, at::ScalarType::BFloat16, "lm_head");

    AddNode("sampling", OpType::kSampling,
            {MakeFusedOp("sampling", "sampling")}, {}, {"logits"},
            {"token_ids"}, at::ScalarType::Long, "sampling");

    output_node_ids_.push_back(
        AddNode("output.token_ids", OpType::kOutput, {}, {}, {"token_ids"}, {},
                at::ScalarType::Long, "", "", "token_ids", CopyKind::kD2H,
                MakeCpuTensor({kBatchExample}, at::ScalarType::Long)));
  }

  // 添加全部边，顺序固定。
  void BuildEdges() {
    AddActivationEdge("input_ids",
                      ConcreteShape({kBatchExample, kSequenceExample}),
                      at::ScalarType::Long);
    AddActivationEdge("position_ids",
                      ConcreteShape({kBatchExample, kSequenceExample}),
                      at::ScalarType::Long);
    AddActivationEdge("last_token_indices", ConcreteShape({kBatchExample}),
                      at::ScalarType::Long);
    AddWeightEdge("weight.embedding", "embedding.weight");
    AddActivationEdge("hidden", HiddenShape(), at::ScalarType::BFloat16);

    for (int32_t layer = 0; layer < kNumLayers; ++layer) {
      const std::string prefix = LayerPrefix(layer);
      AddWeightEdge("weight." + prefix + ".attn_norm",
                    prefix + ".attn_norm.weight");
      AddActivationEdge(prefix + ".attn_normed", HiddenShape(),
                        at::ScalarType::BFloat16);
      AddWeightEdge("weight." + prefix + ".q", prefix + ".q.weight");
      AddWeightEdge("weight." + prefix + ".k", prefix + ".k.weight");
      AddWeightEdge("weight." + prefix + ".v", prefix + ".v.weight");
      AddWeightEdge("weight." + prefix + ".q_norm", prefix + ".q_norm.weight");
      AddWeightEdge("weight." + prefix + ".k_norm", prefix + ".k_norm.weight");
      AddActivationEdge(prefix + ".q",
                        SequenceShape(kNumAttentionHeads * kHeadDim),
                        at::ScalarType::BFloat16);
      AddActivationEdge(prefix + ".k", SequenceShape(kNumKvHeads * kHeadDim),
                        at::ScalarType::BFloat16);
      AddActivationEdge(prefix + ".v", SequenceShape(kNumKvHeads * kHeadDim),
                        at::ScalarType::BFloat16);
      AddWeightEdge("weight." + prefix + ".o", prefix + ".o.weight");
      AddActivationEdge(prefix + ".attn_out", HiddenShape(),
                        at::ScalarType::BFloat16);
      AddActivationEdge(prefix + ".hidden_mid", HiddenShape(),
                        at::ScalarType::BFloat16);
      AddWeightEdge("weight." + prefix + ".mlp_norm",
                    prefix + ".mlp_norm.weight");
      AddActivationEdge(prefix + ".mlp_normed", HiddenShape(),
                        at::ScalarType::BFloat16);
      AddWeightEdge("weight." + prefix + ".gate", prefix + ".gate.weight");
      AddWeightEdge("weight." + prefix + ".up", prefix + ".up.weight");
      AddActivationEdge(prefix + ".mlp_act", SequenceShape(kIntermediateSize),
                        at::ScalarType::BFloat16);
      AddWeightEdge("weight." + prefix + ".down", prefix + ".down.weight");
      AddActivationEdge(prefix + ".hidden_out", HiddenShape(),
                        at::ScalarType::BFloat16);
    }

    AddWeightEdge("weight.final_norm", "final_norm.weight");
    AddActivationEdge("final_normed", HiddenShape(), at::ScalarType::BFloat16);
    AddWeightEdge("weight.lm_head", "lm_head.weight");
    AddActivationEdge("logits", LogitsShape(), at::ScalarType::BFloat16);
    AddActivationEdge("token_ids", ConcreteShape({kBatchExample}),
                      at::ScalarType::Long);
  }

  GraphBuilder builder_;
  std::map<std::string, at::Tensor> weights_;
  std::vector<NodeId> input_node_ids_;
  std::vector<NodeId> output_node_ids_;
};

}  // namespace

std::unique_ptr<Graph> BuildQwen3ReferenceGraph(Status* status) {
  try {
    Qwen3GraphAssembler assembler;
    std::unique_ptr<Graph> graph = assembler.Build();
    if (status != nullptr) {
      *status = Status::Ok();
    }
    return graph;
  } catch (const BuildFailure& failure) {
    if (status != nullptr) {
      *status = Status::Error(failure.code(), failure.what());
    }
    return nullptr;
  } catch (const std::exception& failure) {
    if (status != nullptr) {
      *status = Status::Error(
          ErrorCode::kInternal,
          std::string("Qwen3 fixture failure: ") + failure.what());
    }
    return nullptr;
  }
}

}  // namespace graph
}  // namespace mobilenet
