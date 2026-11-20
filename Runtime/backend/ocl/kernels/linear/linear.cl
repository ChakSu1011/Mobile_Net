// OCL Linear / NT GEMM Kernel（P4.3 基线）。
//
// 每 work-item 计算 4×4 输出块；A=[M,K]，B=[N,K]，C=A·B^T。
// 标量 half 读、FP32 累加；offsets 为元素偏移。

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void linear_nt_4x4(
    __global const half* a,
    const int a_offset,
    __global const half* b,
    const int b_offset,
    __global half* c,
    const int c_offset,
    const int m,
    const int n,
    const int k) {
  const int tiles_n = (n + 3) >> 2;
  const int tiles_m = (m + 3) >> 2;
  const int tile = get_global_id(0);
  const int tile_y = tile / tiles_n;
  if (tile_y >= tiles_m) {
    return;
  }
  const int tile_x = tile - tile_y * tiles_n;
  const int row0 = tile_y << 2;
  const int col0 = tile_x << 2;

  float acc[4][4];
  for (int ti = 0; ti < 4; ++ti) {
    for (int tj = 0; tj < 4; ++tj) {
      acc[ti][tj] = 0.0f;
    }
  }

  for (int kk = 0; kk < k; ++kk) {
    float av[4];
    for (int ti = 0; ti < 4; ++ti) {
      const int row = row0 + ti;
      av[ti] = (row < m) ? (float)a[a_offset + row * k + kk] : 0.0f;
    }
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        const float bv = (float)b[b_offset + col * k + kk];
        for (int ti = 0; ti < 4; ++ti) {
          acc[ti][tj] += av[ti] * bv;
        }
      }
    }
  }

  for (int ti = 0; ti < 4; ++ti) {
    const int row = row0 + ti;
    if (row >= m) {
      continue;
    }
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row * n + col] = (half)acc[ti][tj];
      }
    }
  }
}
