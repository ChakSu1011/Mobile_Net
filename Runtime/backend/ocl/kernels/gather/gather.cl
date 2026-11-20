// OCL Gather / token embedding Kernel（P4.2 基线）。
//
// 一个 workgroup 处理一个 token；hidden % 8 == 0 时用 half8，
// 否则走标量循环。offsets 均为元素偏移（不是字节）。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gather_embedding(
    __global const long* ids,
    const int ids_offset,
    __global const half* weight,
    const int weight_offset,
    __global half* out,
    const int out_offset,
    const int tokens,
    const int vocab,
    const int hidden) {
  const int local_size = get_local_size(0);
  const int token = get_global_id(0) / local_size;
  const int lane = get_local_id(0);
  if (token >= tokens) {
    return;
  }
  const long id = ids[ids_offset + token];
  if (id < 0 || id >= (long)vocab) {
    return;
  }
  const int out_base = out_offset + token * hidden;
  const int weight_base = weight_offset + (int)id * hidden;

  if ((hidden & 7) == 0) {
    const int step = local_size * 8;
    for (int i = lane * 8; i < hidden; i += step) {
      half8 value = vload8(0, weight + weight_base + i);
      vstore8(value, 0, out + out_base + i);
    }
  } else {
    for (int i = lane; i < hidden; i += local_size) {
      out[out_base + i] = weight[weight_base + i];
    }
  }
}
