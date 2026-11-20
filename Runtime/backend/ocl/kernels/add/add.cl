// OCL Add Kernel（P4.2 基线）。
//
// 元素/偏移 8 对齐时使用 half8 + FP32 相加；否则标量。
// offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void add_half(
    __global const half* a,
    const int a_offset,
    __global const half* b,
    const int b_offset,
    __global half* out,
    const int out_offset,
    const int elements) {
  const int global_size = get_global_size(0);
  const int gid = get_global_id(0);
  const bool vectorized =
      ((elements & 7) == 0) && ((a_offset & 7) == 0) &&
      ((b_offset & 7) == 0) && ((out_offset & 7) == 0);

  if (vectorized) {
    for (int i = gid * 8; i < elements; i += global_size * 8) {
      half8 va = vload8(0, a + a_offset + i);
      half8 vb = vload8(0, b + b_offset + i);
      float8 sum = convert_float8(va) + convert_float8(vb);
      half8 vc = convert_half8(sum);
      vstore8(vc, 0, out + out_offset + i);
    }
  } else {
    for (int i = gid; i < elements; i += global_size) {
      out[out_offset + i] =
          (half)((float)a[a_offset + i] + (float)b[b_offset + i]);
    }
  }
}
