"""模块：重复 Layer 的内存布局识别。

职责：
  比较每层 Activation/权重的归一化布局签名，把连续相同层合并为区间，
  并标记与基准层不同的差异层。

边界：
  - 不比较绝对 offset，避免分配器状态造成误判；
  - 不修改模型图或显存计划；
  - 只处理 ``layer.<n>.`` 命名模式。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerRun:
    """一组连续且内存布局相同的 Layer。

    属性：
      start：起始层号；
      end：结束层号；
      count：层数；
      is_variant：是否与基准签名不同。
    """

    start: int
    end: int
    count: int
    is_variant: bool


def analyze_layer_runs(
    signatures: dict[int, tuple],
) -> list[LayerRun]:
    """按签名合并连续层。

    参数：
      signatures：层号到归一化布局签名的映射。

    返回：
      ``LayerRun`` 列表；基准为重复次数最多的区间，次数相同时取最早。
    """
    indices = sorted(signatures)
    if not indices:
        return []
    raw = []
    start = indices[0]
    previous = indices[0]
    signature = signatures[start]
    for index in indices[1:]:
        current = signatures[index]
        if index == previous + 1 and current == signature:
            previous = index
            continue
        raw.append((start, previous, signature))
        start = index
        previous = index
        signature = current
    raw.append((start, previous, signature))

    counts = [end - start + 1 for start, end, _ in raw]
    base_index = max(
        range(len(raw)),
        key=lambda index: (counts[index], -raw[index][0]),
    )
    base_signature = raw[base_index][2]
    return [
        LayerRun(
            start=start,
            end=end,
            count=count,
            is_variant=signature != base_signature,
        )
        for (start, end, signature), count in zip(raw, counts)
    ]
