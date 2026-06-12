"""
Step 1: Data Inventory
目的：摸清 ITW 和 Teleop 两种数据各自有哪些 channel、数据形状、基本统计。
不训练任何模型，只做"我手里有什么"的盘点。
"""

import zarr          # 读取 .zarr 格式数据
import numpy as np   # 数值计算
import os, glob      # 文件路径操作

# ── 数据根目录 ───────────────────────────────────────────────────────────────
BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
DATASETS = {
    "ITW":    "uncap_itw_xarm_7dof_05_18_2026",
    "Teleop": "uncap_teleop_xarm_7dof_05_18_2026",
}

# ════════════════════════════════════════════════════════════════════════════
# 第一部分：列出每个数据集 demo_0 里所有 channel 的 shape 和 dtype
# 为什么看 demo_0？因为所有 demo 的 channel 结构是一样的，只有时间轴长度不同。
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PART 1: Channel 列表（以 demo_0 为代表）")
print("═"*65)

all_keys = {}  # 用字典保存每个数据集的 key 列表，后面做对比用

for mode, dataset_name in DATASETS.items():
    demo0_path = os.path.join(BASE, dataset_name, "demo_0", "data.zarr")
    z = zarr.open(demo0_path, mode='r')          # 只读方式打开 zarr
    keys = sorted(z.keys())                       # 所有 channel 名称，排序方便阅读
    all_keys[mode] = set(keys)

    print(f"\n【{mode}】  路径: {dataset_name}/demo_0/data.zarr")
    print(f"  {'Channel 名':<35} {'Shape':<25} {'dtype'}")
    print(f"  {'-'*35} {'-'*25} {'-'*10}")

    for key in keys:
        arr = z[key]
        # arr.shape 是数组维度，例如 (217, 480, 640, 3)
        # 第 0 维永远是时间步数 T（帧数）
        # 后面的维度是每帧的数据维度
        shape_str = str(arr.shape)
        print(f"  {key:<35} {shape_str:<25} {arr.dtype}")

# ════════════════════════════════════════════════════════════════════════════
# 第二部分：对比 ITW 和 Teleop 的 channel 差异
# ════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*65)
print("  PART 2: Channel 差异对比")
print("═"*65)

shared      = all_keys["ITW"] & all_keys["Teleop"]   # 交集：两者都有
itw_only    = all_keys["ITW"] - all_keys["Teleop"]    # 差集：只有 ITW 有
teleop_only = all_keys["Teleop"] - all_keys["ITW"]    # 差集：只有 Teleop 有

print(f"\n  两者共有的 channel ({len(shared)} 个):")
for k in sorted(shared):
    print(f"    ✓  {k}")

print(f"\n  只有 ITW 有 ({len(itw_only)} 个):")
for k in sorted(itw_only):
    print(f"    →  {k}")

print(f"\n  只有 Teleop 有 ({len(teleop_only)} 个):")
for k in sorted(teleop_only):
    print(f"    →  {k}")

# ════════════════════════════════════════════════════════════════════════════
# 第三部分：全量统计——遍历所有 demo，统计时长、频率等
# ════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*65)
print("  PART 3: 全量 Demo 统计")
print("═"*65)

for mode, dataset_name in DATASETS.items():
    dataset_dir = os.path.join(BASE, dataset_name)
    # glob 找到所有 demo_* 文件夹，sorted 保证顺序
    demo_dirs = sorted(glob.glob(os.path.join(dataset_dir, "demo_*")))

    lengths   = []   # 每条 demo 的帧数
    durations = []   # 每条 demo 的时长（秒）
    freqs     = []   # 每条 demo 的控制频率（Hz）

    for d in demo_dirs:
        z  = zarr.open(os.path.join(d, "data.zarr"), mode='r')
        ts = z["timestamp"][:]       # 一维数组，每帧的 Unix 时间戳（秒）
        n  = len(ts)
        lengths.append(n)

        if n > 1:
            dt   = np.diff(ts)           # 相邻帧之间的时间差，单位：秒
            freq = 1.0 / np.median(dt)   # 中位数时间差取倒数 = 频率
            freqs.append(freq)
            durations.append(ts[-1] - ts[0])   # 最后一帧 - 第一帧 = 总时长

    lengths   = np.array(lengths)
    freqs     = np.array(freqs)
    durations = np.array(durations)

    print(f"\n【{mode}】  ({len(demo_dirs)} demos)")
    print(f"  控制频率 (Hz)    : mean={freqs.mean():.2f}  "
          f"min={freqs.min():.2f}  max={freqs.max():.2f}")
    print(f"  帧数 / demo      : mean={lengths.mean():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}  std={lengths.std():.1f}")
    print(f"  时长 / demo (s)  : mean={durations.mean():.1f}  "
          f"min={durations.min():.1f}  max={durations.max():.1f}")
    print(f"  总帧数           : {lengths.sum()}")
    print(f"  总时长 (s)       : {durations.sum():.1f}")

# ════════════════════════════════════════════════════════════════════════════
# 第四部分：关键信号检查——efforts 是否有真实数据？
# 这对研究极其重要：ITW 有没有力信号？
# ════════════════════════════════════════════════════════════════════════════
print("\n\n" + "═"*65)
print("  PART 4: 力信号检查（efforts 是否全为零？）")
print("═"*65)

for mode, dataset_name in DATASETS.items():
    demo_dirs = sorted(glob.glob(
        os.path.join(BASE, dataset_name, "demo_*")))

    follower_nonzero = 0
    leader_nonzero   = 0

    for d in demo_dirs:
        z = zarr.open(os.path.join(d, "data.zarr"), mode='r')
        if np.any(z["follower_joint_efforts"][:] != 0):
            follower_nonzero += 1
        if np.any(z["leader_joint_efforts"][:] != 0):
            leader_nonzero += 1

    total = len(demo_dirs)
    print(f"\n【{mode}】")
    print(f"  follower_joint_efforts 非零的 demo: {follower_nonzero}/{total}")
    print(f"  leader_joint_efforts   非零的 demo: {leader_nonzero}/{total}")

    # 如果有非零数据，再打印 demo_0 的统计作为样本
    z0 = zarr.open(
        os.path.join(BASE, dataset_name, "demo_0", "data.zarr"), mode='r')
    eff = z0["follower_joint_efforts"][:]
    if np.any(eff != 0):
        print(f"  follower_joint_efforts demo_0 样本 (mean/std per joint):")
        for j in range(eff.shape[1]):
            print(f"    joint {j}: mean={eff[:,j].mean():8.2f}  "
                  f"std={eff[:,j].std():7.2f}  "
                  f"range=[{eff[:,j].min():.1f}, {eff[:,j].max():.1f}]")
    else:
        print(f"  follower_joint_efforts: 全部为 0.0（无力信号）")

print("\n\n" + "═"*65)
print("  数据摸底完成")
print("═"*65 + "\n")
