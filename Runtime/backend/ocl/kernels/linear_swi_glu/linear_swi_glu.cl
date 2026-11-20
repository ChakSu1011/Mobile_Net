// OCL LinearSwiGlu Kernel（P4.3 基线）。
//
// 每 work-item 一个输出：gate/up 双点积 + SiLU epilogue。
// offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void linear_swi_glu_nt(
    __global const half* x,
    const int x_offset,
    __global const half* gate,
    const int gate_offset,
    __global const half* up,
    const int up_offset,
    __global half* out,
    const int out_offset,
    const int rows,
    const int hidden,
    const int intermediate) {
  const int index = get_global_id(0);
  if (index >= rows * intermediate) {
    return;
  }
  const int row = index / intermediate;
  const int col = index - row * intermediate;
  const int x_base = x_offset + row * hidden;
  const int w_base = col * hidden;

  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (int i = 0; i < hidden; ++i) {
    const float value = (float)x[x_base + i];
    gate_sum += value * (float)gate[gate_offset + w_base + i];
    up_sum += value * (float)up[up_offset + w_base + i];
  }
  const float activated = gate_sum / (1.0f + exp(-gate_sum));
  out[out_offset + row * intermediate + col] = (half)(activated * up_sum);
}
