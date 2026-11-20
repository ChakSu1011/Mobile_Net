// OCL MegaAttention Kernel（P4.3 基线）。
//
// causal GQA：一个 work-item 处理一个 (batch, head, query)；
// FP32 online softmax；D ≤ 128；不物化 score 矩阵。
// offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mega_attention_causal(
    __global const half* q,
    const int q_offset,
    __global const half* k,
    const int k_offset,
    __global const half* v,
    const int v_offset,
    __global half* out,
    const int out_offset,
    const int batch,
    const int seq,
    const int heads,
    const int kv_heads,
    const int head_dim,
    const float scale,
    const int causal) {
  const int index = get_global_id(0);
  if (index >= batch * heads * seq) {
    return;
  }
  const int query = index % seq;
  const int head = (index / seq) % heads;
  const int batch_index = index / (seq * heads);

  const int group = heads / kv_heads;
  const int kv_head = head / group;
  const int q_base =
      q_offset + ((batch_index * seq + query) * heads + head) * head_dim;
  const int out_base =
      out_offset + ((batch_index * seq + query) * heads + head) * head_dim;

  float accumulator[128];
  for (int d = 0; d < head_dim; ++d) {
    accumulator[d] = 0.0f;
  }

  const int last_key = causal ? query : seq - 1;
  float max_score = -INFINITY;
  for (int key = 0; key <= last_key; ++key) {
    const int k_base =
        k_offset + ((batch_index * seq + key) * kv_heads + kv_head) *
                       head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += (float)q[q_base + d] * (float)k[k_base + d];
    }
    const float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  float sum = 0.0f;
  for (int key = 0; key <= last_key; ++key) {
    const int k_base =
        k_offset + ((batch_index * seq + key) * kv_heads + kv_head) *
                       head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += (float)q[q_base + d] * (float)k[k_base + d];
    }
    const float probability = exp(dot * scale - max_score);
    sum += probability;
    const int v_base =
        v_offset + ((batch_index * seq + key) * kv_heads + kv_head) *
                       head_dim;
    for (int d = 0; d < head_dim; ++d) {
      accumulator[d] += probability * (float)v[v_base + d];
    }
  }

  const float inverse = 1.0f / sum;
  for (int d = 0; d < head_dim; ++d) {
    out[out_base + d] = (half)(accumulator[d] * inverse);
  }
}
