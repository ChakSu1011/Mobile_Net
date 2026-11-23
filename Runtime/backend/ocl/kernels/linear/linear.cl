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

  // 单行尾部：M=1 或最后一个 tile 只剩 1 行时，避免计算 4 行 padding。
  if (row0 + 1 >= m) {
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int kk1 = 0;
    for (; kk1 + 8 <= k; kk1 += 8) {
      const float8 av =
          vload_half8(0, a + a_offset + row0 * k + kk1);
      float8 bv[4];
      for (int tj = 0; tj < 4; ++tj) {
        const int col = col0 + tj;
        const int safe_col = col < n ? col : 0;
        bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk1);
      }
      for (int tj = 0; tj < 4; ++tj) {
        acc1[tj] += av.s0 * bv[tj].s0 + av.s1 * bv[tj].s1 +
                    av.s2 * bv[tj].s2 + av.s3 * bv[tj].s3 +
                    av.s4 * bv[tj].s4 + av.s5 * bv[tj].s5 +
                    av.s6 * bv[tj].s6 + av.s7 * bv[tj].s7;
      }
    }
    for (; kk1 < k; ++kk1) {
      const float av = (float)a[a_offset + row0 * k + kk1];
      for (int tj = 0; tj < 4; ++tj) {
        const int col = col0 + tj;
        if (col < n) {
          acc1[tj] += av * (float)b[b_offset + col * k + kk1];
        }
      }
    }
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row0 * n + col] = (half)acc1[tj];
      }
    }
    return;
  }

  float acc[4][4];
  for (int ti = 0; ti < 4; ++ti) {
    for (int tj = 0; tj < 4; ++tj) {
      acc[ti][tj] = 0.0f;
    }
  }

  // K 维 half8 向量化：每个 k8 步加载 4 个 A 行向量与 4 个 B 行向量，
  // 完成 4×4×8 次 FMA。越界行/列回落到第 0 行/列读取，结果在写回时丢弃。
  int kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    // vload_half8 把 half 读成 float8，后续直接用 FP32 累加。
    float8 av[4];
    for (int ti = 0; ti < 4; ++ti) {
      const int row = row0 + ti;
      const int safe_row = row < m ? row : 0;
      av[ti] = vload_half8(0, a + a_offset + safe_row * k + kk);
    }
    float8 bv[4];
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      const int safe_col = col < n ? col : 0;
      bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk);
    }
    for (int ti = 0; ti < 4; ++ti) {
      for (int tj = 0; tj < 4; ++tj) {
        acc[ti][tj] += av[ti].s0 * bv[tj].s0 +
                       av[ti].s1 * bv[tj].s1 +
                       av[ti].s2 * bv[tj].s2 +
                       av[ti].s3 * bv[tj].s3 +
                       av[ti].s4 * bv[tj].s4 +
                       av[ti].s5 * bv[tj].s5 +
                       av[ti].s6 * bv[tj].s6 +
                       av[ti].s7 * bv[tj].s7;
      }
    }
  }

  // K 尾部（非 8 倍数）标量累加。
  for (; kk < k; ++kk) {
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

// 只计算 sample_rows 指定行的 NT GEMM（LM Head 优化）。
//
// 每个 work-item 处理一组最多 4 个 sample row × 4 个输出列；B 的 4 个
// 列向量只加载一次，供组内所有 row 复用。row_count==1 时走单行路径，
// 避免为 prefill 的单 token 输出计算 4 行。
__kernel void linear_nt_rows(
    __global const half* a,
    const int a_offset,
    __global const half* b,
    const int b_offset,
    __global half* c,
    const int c_offset,
    __global const int* rows,
    const int rows_offset,
    const int row_count,
    const int n,
    const int k) {
  const int tiles_n = (n + 3) >> 2;
  const int tile = get_global_id(0);
  const int group = tile / tiles_n;
  const int base_row = group << 2;
  if (base_row >= row_count) {
    return;
  }
  const int tile_x = tile - group * tiles_n;
  const int col0 = tile_x << 2;
  int row_ids[4];
  const int valid_rows = min(4, row_count - base_row);
  for (int ti = 0; ti < 4; ++ti) {
    const int index = base_row + (ti < valid_rows ? ti : 0);
    row_ids[ti] = rows[rows_offset + index];
    if (row_ids[ti] < 0) {
      return;
    }
  }

  if (valid_rows == 1) {
    const int row = row_ids[0];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
      const float8 av = vload_half8(0, a + a_offset + row * k + kk);
      float8 bv[4];
      for (int tj = 0; tj < 4; ++tj) {
        const int col = col0 + tj;
        const int safe_col = col < n ? col : 0;
        bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk);
      }
      for (int tj = 0; tj < 4; ++tj) {
        acc[tj] += av.s0 * bv[tj].s0 + av.s1 * bv[tj].s1 +
                   av.s2 * bv[tj].s2 + av.s3 * bv[tj].s3 +
                   av.s4 * bv[tj].s4 + av.s5 * bv[tj].s5 +
                   av.s6 * bv[tj].s6 + av.s7 * bv[tj].s7;
      }
    }
    for (; kk < k; ++kk) {
      const float av = (float)a[a_offset + row * k + kk];
      for (int tj = 0; tj < 4; ++tj) {
        const int col = col0 + tj;
        if (col < n) {
          acc[tj] += av * (float)b[b_offset + col * k + kk];
        }
      }
    }
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row * n + col] = (half)acc[tj];
      }
    }
    return;
  }

  float acc[4][4];
  for (int ti = 0; ti < 4; ++ti) {
    for (int tj = 0; tj < 4; ++tj) {
      acc[ti][tj] = 0.0f;
    }
  }
  int kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    float8 av[4];
    for (int ti = 0; ti < 4; ++ti) {
      av[ti] = vload_half8(0, a + a_offset + row_ids[ti] * k + kk);
    }
    float8 bv[4];
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      const int safe_col = col < n ? col : 0;
      bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk);
    }
    for (int ti = 0; ti < 4; ++ti) {
      for (int tj = 0; tj < 4; ++tj) {
        acc[ti][tj] += av[ti].s0 * bv[tj].s0 +
                       av[ti].s1 * bv[tj].s1 +
                       av[ti].s2 * bv[tj].s2 +
                       av[ti].s3 * bv[tj].s3 +
                       av[ti].s4 * bv[tj].s4 +
                       av[ti].s5 * bv[tj].s5 +
                       av[ti].s6 * bv[tj].s6 +
                       av[ti].s7 * bv[tj].s7;
      }
    }
  }
  for (; kk < k; ++kk) {
    float av[4];
    for (int ti = 0; ti < 4; ++ti) {
      av[ti] = (float)a[a_offset + row_ids[ti] * k + kk];
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
  for (int ti = 0; ti < valid_rows; ++ti) {
    for (int tj = 0; tj < 4; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row_ids[ti] * n + col] = (half)acc[ti][tj];
      }
    }
  }
}

// NT GEMM 4×8 分块：每个 work-item 计算 4 行 × 8 列。
// 相对 4×4，N 方向加倍，B 的 8 个列向量供 4 个 A 行复用。
__kernel void linear_nt_4x8(
    __global const half* a,
    const int a_offset,
    __global const half* b,
    const int b_offset,
    __global half* c,
    const int c_offset,
    const int m,
    const int n,
    const int k) {
  const int tiles_n = (n + 7) >> 3;
  const int tiles_m = (m + 3) >> 2;
  const int tile = get_global_id(0);
  const int tile_y = tile / tiles_n;
  if (tile_y >= tiles_m) {
    return;
  }
  const int tile_x = tile - tile_y * tiles_n;
  const int row0 = tile_y << 2;
  const int col0 = tile_x << 3;

  if (row0 + 1 >= m) {
    float acc[8];
    for (int tj = 0; tj < 8; ++tj) {
      acc[tj] = 0.0f;
    }
    int kk = 0;
    for (; kk + 8 <= k; kk += 8) {
      const float8 av =
          vload_half8(0, a + a_offset + row0 * k + kk);
      float8 bv[8];
      for (int tj = 0; tj < 8; ++tj) {
        const int col = col0 + tj;
        const int safe_col = col < n ? col : 0;
        bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk);
      }
      for (int tj = 0; tj < 8; ++tj) {
        acc[tj] += av.s0 * bv[tj].s0 + av.s1 * bv[tj].s1 +
                   av.s2 * bv[tj].s2 + av.s3 * bv[tj].s3 +
                   av.s4 * bv[tj].s4 + av.s5 * bv[tj].s5 +
                   av.s6 * bv[tj].s6 + av.s7 * bv[tj].s7;
      }
    }
    for (; kk < k; ++kk) {
      const float av = (float)a[a_offset + row0 * k + kk];
      for (int tj = 0; tj < 8; ++tj) {
        const int col = col0 + tj;
        if (col < n) {
          acc[tj] += av * (float)b[b_offset + col * k + kk];
        }
      }
    }
    for (int tj = 0; tj < 8; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row0 * n + col] = (half)acc[tj];
      }
    }
    return;
  }

  float acc[4][8];
  for (int ti = 0; ti < 4; ++ti) {
    for (int tj = 0; tj < 8; ++tj) {
      acc[ti][tj] = 0.0f;
    }
  }
  int kk = 0;
  for (; kk + 8 <= k; kk += 8) {
    float8 av[4];
    for (int ti = 0; ti < 4; ++ti) {
      const int row = row0 + ti;
      const int safe_row = row < m ? row : 0;
      av[ti] = vload_half8(0, a + a_offset + safe_row * k + kk);
    }
    float8 bv[8];
    for (int tj = 0; tj < 8; ++tj) {
      const int col = col0 + tj;
      const int safe_col = col < n ? col : 0;
      bv[tj] = vload_half8(0, b + b_offset + safe_col * k + kk);
    }
    for (int ti = 0; ti < 4; ++ti) {
      for (int tj = 0; tj < 8; ++tj) {
        acc[ti][tj] += av[ti].s0 * bv[tj].s0 +
                       av[ti].s1 * bv[tj].s1 +
                       av[ti].s2 * bv[tj].s2 +
                       av[ti].s3 * bv[tj].s3 +
                       av[ti].s4 * bv[tj].s4 +
                       av[ti].s5 * bv[tj].s5 +
                       av[ti].s6 * bv[tj].s6 +
                       av[ti].s7 * bv[tj].s7;
      }
    }
  }
  for (; kk < k; ++kk) {
    float av[4];
    for (int ti = 0; ti < 4; ++ti) {
      const int row = row0 + ti;
      av[ti] = row < m ? (float)a[a_offset + row * k + kk] : 0.0f;
    }
    for (int tj = 0; tj < 8; ++tj) {
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
    for (int tj = 0; tj < 8; ++tj) {
      const int col = col0 + tj;
      if (col < n) {
        c[c_offset + row * n + col] = (half)acc[ti][tj];
      }
    }
  }
}
