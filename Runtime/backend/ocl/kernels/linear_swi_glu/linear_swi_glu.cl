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
  // K 维 half8 向量化：x/gate/up 同址各读 8 个 half，FP32 累加。
  int i = 0;
  for (; i + 8 <= hidden; i += 8) {
    const float8 xv = vload_half8(0, x + x_base + i);
    const float8 gv = vload_half8(0, gate + gate_offset + w_base + i);
    const float8 uv = vload_half8(0, up + up_offset + w_base + i);
    gate_sum += xv.s0 * gv.s0 + xv.s1 * gv.s1 + xv.s2 * gv.s2 +
                xv.s3 * gv.s3 + xv.s4 * gv.s4 + xv.s5 * gv.s5 +
                xv.s6 * gv.s6 + xv.s7 * gv.s7;
    up_sum += xv.s0 * uv.s0 + xv.s1 * uv.s1 + xv.s2 * uv.s2 +
              xv.s3 * uv.s3 + xv.s4 * uv.s4 + xv.s5 * uv.s5 +
              xv.s6 * uv.s6 + xv.s7 * uv.s7;
  }
  for (; i < hidden; ++i) {
    const float value = (float)x[x_base + i];
    gate_sum += value * (float)gate[gate_offset + w_base + i];
    up_sum += value * (float)up[up_offset + w_base + i];
  }
  const float activated = gate_sum / (1.0f + exp(-gate_sum));
  out[out_offset + row * intermediate + col] = (half)(activated * up_sum);
}
