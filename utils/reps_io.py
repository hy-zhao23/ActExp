"""
Activation 分片的统一读取入口。

历史上有两种在盘格式，两边都还在用：
  - **带 npy 头**：`np.save` 写的（extract_diversified.py）。头长可变（≥128B，随 dtype/shape
    的 dict 长度变化），不是固定 128。
  - **裸 memmap**：`np.memmap` 写的（cache_finetune_reps.py），无任何头。

三个加载器（qa_data / oracle_dataset / finetune_dataset）过去一律按裸格式 memmap。
读带头文件时，头部字节被当成数据，**每行整体错位 header_size/2 个 uint16**：
    读到的第 i 行 = 真实第 i-1 行的末 64 维 ++ 真实第 i 行的前 2496 维
即每个向量混入 2.5% 的邻居样本、且整体旋转。训练与推理用同一加载器，错位一致，
所以模型仍能学（旧 ckpt 就是这么训出来的），但信号是脏的。

本模块按文件头自动分派，两种格式都读对。修复后需重训依赖 rep 的模型
（反演器 ③ 等），旧 ckpt 与新读法不兼容。
"""

import os
from pathlib import Path

import numpy as np

_NPY_MAGIC = b"\x93NUMPY"


def open_reps_shard(path, n_rows: int, hidden_dim: int, dtype: str = "uint16"):
    """按 (n_rows, hidden_dim) 惰性打开一个 activation 分片。

    带头 → np.load(mmap_mode="r")（numpy 自己跳过头，头长多少都对）；
    裸格式 → np.memmap。两者都不把数据读进内存。

    形状/大小与预期不符时直接抛错——这类不一致过去是静默的。
    """
    path = Path(path)
    with path.open("rb") as f:
        has_header = f.read(6) == _NPY_MAGIC

    if has_header:
        arr = np.load(path, mmap_mode="r")
        if arr.shape != (n_rows, hidden_dim):
            raise ValueError(
                f"{path.name}: npy 形状 {arr.shape} != 预期 ({n_rows}, {hidden_dim})")
        return arr

    expect = n_rows * hidden_dim * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual != expect:
        raise ValueError(
            f"{path.name}: 裸 memmap 大小 {actual} != 预期 {expect} "
            f"({n_rows}×{hidden_dim}×{np.dtype(dtype).itemsize})")
    return np.memmap(path, dtype=dtype, mode="r", shape=(n_rows, hidden_dim))


# ── 分块写入（断点续传）→ npy 定稿 ────────────────────────────────────────────
# np.save 无法增量写，而缓存作业跑在可抢占队列上、必须能续跑。故：作业期间用
# memmap 裸写 <out>.building 工作文件，全部算完再一次性定稿为 npy。
# 这样续传能力与自描述格式兼得，且中途被抢占时 <out> 不会存在半成品。

_BUILD_SUFFIX = ".building"


def open_building_shard(out_path, n_rows: int, hidden_dim: int):
    """打开（或新建）分块写入的裸格式工作文件；已存在则续写。"""
    work = Path(str(out_path) + _BUILD_SUFFIX)
    mode = "r+" if work.exists() else "w+"
    return np.memmap(work, dtype="uint16", mode=mode, shape=(n_rows, hidden_dim))


def finalize_building(mmap_obj, out_path) -> tuple:
    """工作文件 → npy 定稿：临时文件 + 原子替换，成功后删除工作文件。"""
    out_path = Path(out_path)
    mmap_obj.flush()
    arr = np.asarray(mmap_obj)
    shape = arr.shape
    tmp = Path(str(out_path) + ".finalizing")
    try:
        with tmp.open("wb") as f:
            np.lib.format.write_array(f, arr, allow_pickle=False)
        os.replace(tmp, out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    Path(str(out_path) + _BUILD_SUFFIX).unlink(missing_ok=True)
    return shape
