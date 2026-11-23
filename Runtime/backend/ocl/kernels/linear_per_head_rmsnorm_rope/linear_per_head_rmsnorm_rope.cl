// OCL LinearPerHeadRMSNormRope Kernel（P4.5，Intel 优先）。
//
// 一 (row, head) 一 workgroup，local size 128；lane 负责一个 head 维度：
// half8 装载 + FP16 FMA 点积 GEMM → FP32 per-head RMSNorm → FP32 NeoX RoPE。
// D ≤ 128；offsets 为元素偏移。

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

  const int vec_ok = ((in_features & 7) == 0) &&
                     ((x_offset & 7) == 0) && ((weight_offset & 7) == 0);
  half value = (half)0.0f;
  if (lane < head_dim) {
    const int column = head * head_dim + lane;
    const int x_base = x_offset + row * in_features;
    const int w_base = weight_offset + column * in_features;
    // K 维 half8 向量化：同址读 x 与 weight 各 8 个 half，FP16 FMA 累加。
    int i = 0;
    for (; i + 8 <= in_features; i += 8) {
      const __global half* x_ptr = x + x_base + i;
      const __global half* w_ptr = weight + w_base + i;
      const half8 xv = vec_ok ? *((const __global half8*)x_ptr)
                              : convert_half8(vload_half8(0, x_ptr));
      const half8 wv = vec_ok ? *((const __global half8*)w_ptr)
                              : convert_half8(vload_half8(0, w_ptr));
      value = fma(xv.s0, wv.s0, value);
      value = fma(xv.s1, wv.s1, value);
      value = fma(xv.s2, wv.s2, value);
      value = fma(xv.s3, wv.s3, value);
      value = fma(xv.s4, wv.s4, value);
      value = fma(xv.s5, wv.s5, value);
      value = fma(xv.s6, wv.s6, value);
      value = fma(xv.s7, wv.s7, value);
    }
    for (; i < in_features; ++i) {
      value = fma(x[x_base + i], weight[w_base + i], value);
    }
    const float dot = (float)value;
    values[lane] = dot;
    squares[lane] = dot * dot;
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
