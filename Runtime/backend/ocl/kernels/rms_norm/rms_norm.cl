// OCL RMSNorm Kernel（P4.3 基线）。
//
// 一行一 workgroup，local size 128；FP32 平方和 + local 树归约；
// offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rms_norm_row(
    __global const half* x,
    const int x_offset,
    __global const half* w,
    const int w_offset,
    __global half* out,
    const int out_offset,
    const int rows,
    const int hidden,
    const float eps) {
  const int local_size = get_local_size(0);
  const int row = get_global_id(0) / local_size;
  const int tid = get_local_id(0);
  if (row >= rows) {
    return;
  }
  const int base = x_offset + row * hidden;
  float sum = 0.0f;
  for (int i = tid; i < hidden; i += local_size) {
    const float value = (float)x[base + i];
    sum += value * value;
  }

  __local float partial[128];
  partial[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  const float scale = rsqrt(partial[0] / (float)hidden + eps);
  const int out_base = out_offset + row * hidden;
  for (int i = tid; i < hidden; i += local_size) {
    const float value = (float)x[base + i];
    const float weight = (float)w[w_offset + i];
    out[out_base + i] = (half)(value * scale * weight);
  }
}
