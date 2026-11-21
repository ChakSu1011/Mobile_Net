// OCL MegaAttention Kernel（P4.3 基线）。
//
// causal GQA：一个 work-item 处理一个 (batch, head, query)；
// FP32 online softmax；D ≤ 128；不物化 score 矩阵。
// offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 全量 causal GQA：一个 work-item 处理一个 (batch, head, query)，
// 对 key 0..last_key 做 FP32 online softmax。
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
    const int seq_stride,
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
  // seq 是有效长度，seq_stride 是静态图的行跨度（含 padding）。
  const int q_base =
      q_offset + ((batch_index * seq_stride + query) * heads + head) *
                     head_dim;
  const int out_base =
      out_offset + ((batch_index * seq_stride + query) * heads + head) *
                       head_dim;

  float accumulator[128];
  for (int d = 0; d < head_dim; ++d) {
    accumulator[d] = 0.0f;
  }

  const int last_key = causal ? query : seq - 1;
  // 单遍 online softmax：新最大值出现时按 alpha 缩放历史累加，避免像
  // 两遍实现那样重复计算 score（P4.5）。
  float max_score = -INFINITY;
  float sum = 0.0f;
  for (int key = 0; key <= last_key; ++key) {
    const int k_base =
        k_offset + ((batch_index * seq_stride + key) * kv_heads + kv_head) *
                       head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += (float)q[q_base + d] * (float)k[k_base + d];
    }
    const float score = dot * scale;
    const int v_base =
        v_offset + ((batch_index * seq_stride + key) * kv_heads + kv_head) *
                       head_dim;
    if (score > max_score) {
      const float alpha = exp(max_score - score);
      sum *= alpha;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] *= alpha;
      }
      max_score = score;
      sum += 1.0f;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] += (float)v[v_base + d];
      }
    } else {
      const float probability = exp(score - max_score);
      sum += probability;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] += probability * (float)v[v_base + d];
      }
    }
  }

  const float inverse = 1.0f / sum;
  for (int d = 0; d < head_dim; ++d) {
    out[out_base + d] = (half)(accumulator[d] * inverse);
  }
}

// 把本步 K/V 写入 KV 池 slot。
//
// 一个 work-item 处理一个 (token, kv_head) 行；slot 是扁平 token 下标，
// 池布局为 NHD：slot * kv_heads * head_dim + kv_head * head_dim。
__kernel void paged_kv_store(
    __global const half* k,
    const int k_offset,
    __global const half* v,
    const int v_offset,
    __global half* k_pool,
    __global half* v_pool,
    __global const long* slots,
    const int slot_offset,
    const int token_count,
    const int kv_heads,
    const int head_dim) {
  const int index = get_global_id(0);
  if (index >= token_count * kv_heads) {
    return;
  }
  const int token = index / kv_heads;
  const int kv_head = index % kv_heads;
  const long slot = slots[slot_offset + token];
  const long pool_base =
      slot * (long)(kv_heads * head_dim) + (long)(kv_head * head_dim);
  const int input_base =
      (token * kv_heads + kv_head) * head_dim;
  for (int d = 0; d < head_dim; ++d) {
    k_pool[pool_base + d] = k[k_offset + input_base + d];
    v_pool[pool_base + d] = v[v_offset + input_base + d];
  }
}

// Paged causal GQA attention。
//
// 一个 work-item 处理一个 (token, q_head)；token 是 packed 下标，
// request_indices 给出所属请求，positions 给出绝对位置。
__kernel void mega_attention_paged(
    __global const half* q,
    const int q_offset,
    __global half* out,
    const int out_offset,
    __global const half* k_pool,
    __global const half* v_pool,
    __global const int* seq_lens,
    const int seq_offset,
    __global const int* query_lens,
    const int query_offset,
    __global const int* context_lens,
    const int context_offset,
    __global const int* request_indices,
    const int request_offset,
    __global const int* positions,
    const int position_offset,
    __global const int* block_table,
    const int block_table_offset,
    const int batch,
    const int token_count,
    const int q_stride,
    const int block_table_stride,
    const int heads,
    const int kv_heads,
    const int head_dim,
    const int kv_block_size,
    const float scale,
    const int causal) {
  const int index = get_global_id(0);
  if (index >= token_count * heads) {
    return;
  }
  const int token = index / heads;
  const int head = index % heads;
  if (token >= token_count || token >= q_stride) {
    return;
  }
  const int request = request_indices[request_offset + token];
  if (request < 0 || request >= batch) {
    return;
  }
  const int query_position = positions[position_offset + token];
  const int seq_len = seq_lens[seq_offset + request];
  const int query_len = query_lens[query_offset + request];
  const int context_len = context_lens[context_offset + request];
  if (seq_len != context_len + query_len || seq_len <= 0 ||
      query_position < 0 || query_position >= seq_len) {
    return;
  }

  const int group = heads / kv_heads;
  const int kv_head = head / group;
  const long q_base =
      (long)q_offset + (long)(token * heads + head) * head_dim;
  const long out_base =
      (long)out_offset + (long)(token * heads + head) * head_dim;

  float accumulator[128];
  for (int d = 0; d < head_dim; ++d) {
    accumulator[d] = 0.0f;
  }

  float max_score = -INFINITY;
  float sum = 0.0f;
  for (int kv_pos = 0; kv_pos < seq_len; ++kv_pos) {
    if (causal && kv_pos > query_position) {
      break;
    }
    const int block_index = kv_pos / kv_block_size;
    const int block_offset = kv_pos % kv_block_size;
    const int block =
        block_table[block_table_offset +
                    request * block_table_stride + block_index];
    if (block < 0) {
      return;
    }
    const long pool_slot =
        (long)block * kv_block_size + block_offset;
    const long pool_base =
        pool_slot * (long)(kv_heads * head_dim) +
        (long)(kv_head * head_dim);
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += (float)q[q_base + d] * (float)k_pool[pool_base + d];
    }
    const float score = dot * scale;
    if (score > max_score) {
      const float alpha = exp(max_score - score);
      sum *= alpha;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] *= alpha;
      }
      max_score = score;
      sum += 1.0f;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] += (float)v_pool[pool_base + d];
      }
    } else {
      const float probability = exp(score - max_score);
      sum += probability;
      for (int d = 0; d < head_dim; ++d) {
        accumulator[d] += probability * (float)v_pool[pool_base + d];
      }
    }
  }
  const float inverse = 1.0f / sum;
  for (int d = 0; d < head_dim; ++d) {
    out[out_base + d] = (half)(accumulator[d] * inverse);
  }
}
