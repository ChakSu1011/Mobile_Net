// OCL LinearSwiGlu Kernel（P4.5，Intel 优先）。
//
// 每 work-item 一个输出：gate/up 双点积 + SiLU epilogue。
// half8 装载 + FP16 FMA 累加；SiLU 在 FP32 中计算。
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
  const int vec_ok = ((hidden & 7) == 0) && ((x_offset & 7) == 0) &&
                     ((gate_offset & 7) == 0) && ((up_offset & 7) == 0);

  half gate_sum = (half)0.0f;
  half up_sum = (half)0.0f;
  // K 维 half8 向量化：x/gate/up 同址各读 8 个 half，FP16 FMA 累加。
  int i = 0;
  for (; i + 8 <= hidden; i += 8) {
    const __global half* x_ptr = x + x_base + i;
    const __global half* g_ptr = gate + gate_offset + w_base + i;
    const __global half* u_ptr = up + up_offset + w_base + i;
    const half8 xv = vec_ok ? *((const __global half8*)x_ptr)
                            : convert_half8(vload_half8(0, x_ptr));
    const half8 gv = vec_ok ? *((const __global half8*)g_ptr)
                            : convert_half8(vload_half8(0, g_ptr));
    const half8 uv = vec_ok ? *((const __global half8*)u_ptr)
                            : convert_half8(vload_half8(0, u_ptr));
    gate_sum = fma(xv.s0, gv.s0, gate_sum);
    gate_sum = fma(xv.s1, gv.s1, gate_sum);
    gate_sum = fma(xv.s2, gv.s2, gate_sum);
    gate_sum = fma(xv.s3, gv.s3, gate_sum);
    gate_sum = fma(xv.s4, gv.s4, gate_sum);
    gate_sum = fma(xv.s5, gv.s5, gate_sum);
    gate_sum = fma(xv.s6, gv.s6, gate_sum);
    gate_sum = fma(xv.s7, gv.s7, gate_sum);
    up_sum = fma(xv.s0, uv.s0, up_sum);
    up_sum = fma(xv.s1, uv.s1, up_sum);
    up_sum = fma(xv.s2, uv.s2, up_sum);
    up_sum = fma(xv.s3, uv.s3, up_sum);
    up_sum = fma(xv.s4, uv.s4, up_sum);
    up_sum = fma(xv.s5, uv.s5, up_sum);
    up_sum = fma(xv.s6, uv.s6, up_sum);
    up_sum = fma(xv.s7, uv.s7, up_sum);
  }
  for (; i < hidden; ++i) {
    const half value = x[x_base + i];
    gate_sum = fma(value, gate[gate_offset + w_base + i], gate_sum);
    up_sum = fma(value, up[up_offset + w_base + i], up_sum);
  }
  // SiLU 在 FP32 中计算，避免 exp 的 half 精度问题；累加值保持 FP16。
  const float gate_f = (float)gate_sum;
  const float activated = gate_f / (1.0f + exp(-gate_f));
  out[out_offset + row * intermediate + col] =
      (half)(activated * (float)up_sum);
}
