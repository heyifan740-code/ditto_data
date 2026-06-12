"""
Step 0 收尾：验证 ITW arm_obs 能否当米制位姿算 v_ee
对应截图三件事的验证 2（单位/量级）和验证 3（分布 sanity check）。

源码已确认（验证 1、2 的定性部分）：
  in_the_wild_agent.py        -> arm_obs = 相对 episode 起点的 6D 位移，base 固定
  trakstar_arm_transform_utils -> 返回 [dx,dy,dz,drx,dry,drz]，单位 米 + 弧度
本脚本做数值验证：ITW 的 ‖Δp‖/dt 峰值是否落在人手合理区间 (~0.1–1.5 m/s)。
"""

import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
DATASETS = {"ITW": "uncap_itw_xarm_7dof_05_18_2026",
            "Teleop": "uncap_teleop_xarm_7dof_05_18_2026"}

def moving_average(x, w=5):
    """对每一列做 w 帧滑动平均，去高频噪声。x: (T, D) -> (T, D)
    用 uniform_filter1d(mode='nearest')：边界用最近邻值填充，
    避免 np.convolve('same') 的零填充在序列首尾制造假跳变。"""
    if len(x) < w:
        return x
    return uniform_filter1d(x, size=w, axis=0, mode='nearest')

def compute_vee_for_dataset(name):
    """遍历一个数据集所有 demo，返回所有帧的末端线速度大小 (m/s)。"""
    demo_dirs = sorted(glob.glob(os.path.join(BASE, name, "demo_*")))
    all_speeds = []
    peak_per_demo = []   # 每条 demo 的速度峰值（P99，避免单点 outlier）

    for d in demo_dirs:
        z  = zarr.open(os.path.join(d, "data.zarr"), 'r')
        pos = z["arm_obs"][:][:, :3]          # 取 dim0-2 = xyz 位置（米）
        ts  = z["timestamp"][:]
        if len(pos) < 6:
            continue
        pos_s = moving_average(pos, w=5)       # 先平滑
        dp = np.diff(pos_s, axis=0)            # 相邻帧位置差 (T-1, 3)
        dt = np.diff(ts)                       # 相邻帧时间差 (T-1,)
        dt = np.clip(dt, 1e-3, None)           # 防止除零
        speed = np.linalg.norm(dp, axis=1) / dt  # ‖Δp‖/dt -> m/s
        all_speeds.append(speed)
        peak_per_demo.append(np.percentile(speed, 99))

    return np.concatenate(all_speeds), np.array(peak_per_demo)

# ─── 计算两侧 ───────────────────────────────────────────────────────────────
results = {}
for mode, name in DATASETS.items():
    speeds, peaks = compute_vee_for_dataset(name)
    results[mode] = {"speeds": speeds, "peaks": peaks}

# ─── 打印验证 2：单位/量级 ───────────────────────────────────────────────────
print("="*68)
print("验证 2：v_ee = ‖Δp‖/dt 的量级（人手合理区间 ~0.1–1.5 m/s）")
print("="*68)
for mode in ["ITW", "Teleop"]:
    s = results[mode]["speeds"]
    p = results[mode]["peaks"]
    print(f"\n【{mode}】 共 {len(s)} 帧 speed 样本")
    print(f"  median      = {np.median(s):.3f} m/s")
    print(f"  mean        = {s.mean():.3f} m/s")
    print(f"  P95         = {np.percentile(s,95):.3f} m/s")
    print(f"  P99         = {np.percentile(s,99):.3f} m/s")
    print(f"  max         = {s.max():.3f} m/s")
    print(f"  每-demo 峰值(P99) 的中位数 = {np.median(p):.3f} m/s  "
          f"(范围 {p.min():.3f}–{p.max():.3f})")

# 判定
itw_peak = np.median(results["ITW"]["peaks"])
verdict = "通过 ✓" if 0.1 <= itw_peak <= 2.0 else "需复核 ✗"
print(f"\n  >>> ITW 每-demo 峰值中位数 = {itw_peak:.3f} m/s -> {verdict}")
print(f"      (人手操作峰值通常 0.1–1.5 m/s，放宽到 2.0 容忍快速接近动作)")

# ─── 验证 3：histogram 叠图 ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：线性坐标
ax = axes[0]
bins = np.linspace(0, max(np.percentile(results["ITW"]["speeds"],99.5),
                          np.percentile(results["Teleop"]["speeds"],99.5)), 60)
for mode, color in [("ITW", "tab:orange"), ("Teleop", "tab:blue")]:
    ax.hist(results[mode]["speeds"], bins=bins, alpha=0.55,
            density=True, label=mode, color=color)
ax.axvspan(0.1, 1.5, color='green', alpha=0.08, label="human hand 0.1-1.5 m/s")
ax.set_xlabel("end-effector speed |v_ee| (m/s)")
ax.set_ylabel("density")
ax.set_title("v_ee distribution (linear)")
ax.legend()
ax.grid(alpha=0.3)

# 右：log-y，看尾部
ax = axes[1]
for mode, color in [("ITW", "tab:orange"), ("Teleop", "tab:blue")]:
    ax.hist(results[mode]["speeds"], bins=bins, alpha=0.55,
            density=True, label=mode, color=color, log=True)
ax.set_xlabel("end-effector speed |v_ee| (m/s)")
ax.set_ylabel("density (log)")
ax.set_title("v_ee distribution (log-y, tail)")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Verification 3: ITW vs Teleop end-effector speed (sanity check)",
             fontsize=13)
fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "vee_verification")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "verify_vee_histogram.png")
fig.savefig(out, dpi=110)
print(f"\nhistogram 已保存: {out}")
