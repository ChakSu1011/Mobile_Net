// OCL Sample / Argmax Kernel（P4.2 基线）。
//
// 一行一 workgroup，local size 固定 256；标量扫描 + local 树归约。
// offsets 为元素偏移；并列最大值取最小下标。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sample_argmax(
    __global const half* logits,
    const int logits_offset,
    __global long* out,
    const int out_offset,
    const int rows,
    const int vocab) {
  const int local_size = get_local_size(0);
  const int row = get_global_id(0) / local_size;
  const int tid = get_local_id(0);
  if (row >= rows) {
    return;
  }
  const int base = logits_offset + row * vocab;
  float best = -INFINITY;
  int best_index = 0;
  for (int i = tid; i < vocab; i += local_size) {
    const float value = (float)logits[base + i];
    if (value > best) {
      best = value;
      best_index = i;
    }
  }

  __local float best_values[256];
  __local int best_indices[256];
  best_values[tid] = best;
  best_indices[tid] = best_index;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = best_values[tid + stride];
      const int other_index = best_indices[tid + stride];
      if (other_value > best_values[tid] ||
          (other_value == best_values[tid] &&
           other_index < best_indices[tid])) {
        best_values[tid] = other_value;
        best_indices[tid] = other_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    out[out_offset + row] = (long)best_indices[0];
  }
}
