// OCL LinearPerHeadRMSNormRope Kernel（P4.3 基线）。
//
// 一 (row, head) 一 workgroup，local size 128；lane 负责一个 head 维度：
// 点积 GEMM → per-head RMSNorm → NeoX RoPE。D ≤ 128；offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void per_head_rmsnorm_rope(
    __global const half* x,
    const int x_offset,
    __global const half* weight,
    const int weight_offset,
    __global const half* norm,
    const int norm_offset,
    __global const long* positions,
    const int position_offset,
    __global half* out,
    const int out_offset,
    const int rows,
    const int heads,
    const int head_dim,
    const int in_features,
    const float eps,
    const float theta) {
  const int local_size = get_local_size(0);
  const int query = get_global_id(0) / local_size;
  const int lane = get_local_id(0);
  const int row = query % rows;
  const int head = query / rows;
  if (head >= heads) {
    return;
  }

  __local float values[128];
  __local float squares[128];
  __local float scale_shared;

  float value = 0.0f;
  if (lane < head_dim) {
    const int column = head * head_dim + lane;
    const int x_base = x_offset + row * in_features;
    const int w_base = weight_offset + column * in_features;
    for (int i = 0; i < in_features; ++i) {
      value += (float)x[x_base + i] * (float)weight[w_base + i];
    }
    values[lane] = value;
    squares[lane] = value * value;
  } else {
    values[lane] = 0.0f;
    squares[lane] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lane == 0) {
    float sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      sum += squares[i];
    }
    scale_shared = rsqrt(sum / (float)head_dim + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lane < head_dim) {
    values[lane] =
        values[lane] * scale_shared * (float)norm[norm_offset + lane];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const int half_dim = head_dim >> 1;
  if (lane < half_dim) {
    const int position = (int)positions[position_offset + row];
    const float angle =
        (float)position *
        pow(theta, -2.0f * (float)lane / (float)head_dim);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const int second = lane + half_dim;
    const float first_value = values[lane];
    const float second_value = values[second];
    const int out_base =
        out_offset + row * heads * head_dim + head * head_dim;
    out[out_base + lane] = (half)(first_value * cosine -
                                  second_value * sine);
    out[out_base + second] = (half)(second_value * cosine +
                                    first_value * sine);
  }
}
