"""
Step 1: Phase 切分（free vs contact）
- v_ee  = ‖Δ(arm_obs[:3])‖/dt    末端线速度（手臂移动快慢）
- a_hand= ‖Δ(follower_joint_states)‖/dt  手指角速度范数（手指活动强度）
- 状态机：阈值 = 该轨迹 v_ee 的 25th percentile，带迟滞 + 最小段长
- 输出：单条 demo 的时序图（曲线 + phase 着色）+ 切换点抽帧核对

用法: python 05_phase_segmentation.py teleop 0
"""

import zarr, numpy as np, os, sys
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mode    = sys.argv[1] if len(sys.argv) > 1 else "teleop"
demo_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
DATASET = {"itw": "uncap_itw_xarm_7dof_05_18_2026",
           "teleop": "uncap_teleop_xarm_7dof_05_18_2026"}[mode]
zpath = os.path.join(BASE, DATASET, f"demo_{demo_id}", "data.zarr")
z = zarr.open(zpath, 'r')

# ─── 读数据 ──────────────────────────────────────────────────────────────────
pos  = z["arm_obs"][:][:, :3]              # 末端位置 (T,3) 米
hand = z["follower_joint_states"][:]       # 手指关节角 (T,7)
ts   = z["timestamp"][:]
ts0  = ts - ts[0]
T    = len(ts)

# ─── 计算 v_ee 和 a_hand（统一用 uniform_filter1d 平滑，避免边界伪影） ────────
def speed_norm(x, ts, smooth=5):
    """对多维信号 x(T,D) 算逐帧速度范数 ‖Δx‖/dt。返回长度 T（首帧补第二帧的值）。"""
    xs = uniform_filter1d(x, size=smooth, axis=0, mode='nearest')   # 先平滑
    dx = np.diff(xs, axis=0)                                         # (T-1, D)
    dt = np.clip(np.diff(ts), 1e-3, None)                           # (T-1,)
    spd = np.linalg.norm(dx, axis=1) / dt                           # (T-1,)
    return np.concatenate([[spd[0]], spd])                          # 对齐回 T

v_ee   = speed_norm(pos,  ts, smooth=5)    # (T,) m/s
a_hand = speed_norm(hand, ts, smooth=5)    # (T,) rad/s（手指活动度）

# ─── 状态机切 phase ──────────────────────────────────────────────────────────
# 阈值：该轨迹 v_ee 的 25th percentile
thr = np.percentile(v_ee, 25)
# 迟滞：上阈值 = thr，下阈值略低，避免在阈值附近反复横跳
hi = thr * 1.15
lo = thr * 0.85

# phase 标签：0=contact（慢，末端静止）, 1=free（快，末端移动）
# 初始状态按第一帧定
labels = np.zeros(T, dtype=int)
state = 1 if v_ee[0] > thr else 0
for i in range(T):
    if state == 0 and v_ee[i] > hi:        # 当前 contact，速度升过上阈值 -> 转 free
        state = 1
    elif state == 1 and v_ee[i] < lo:      # 当前 free，速度降到下阈值 -> 转 contact
        state = 0
    labels[i] = state

# 最小段长过滤：把短于 min_len 帧的碎段并入相邻段（去抖）
min_len = 5   # 5 帧 = 0.25s
def remove_short_segments(labels, min_len):
    lab = labels.copy()
    # 找出所有连续段 [start, end, value]
    segs = []
    s = 0
    for i in range(1, len(lab)+1):
        if i == len(lab) or lab[i] != lab[s]:
            segs.append([s, i, lab[s]]); s = i
    # 把过短的段翻转为相邻段的值（用前一段的值填充）
    for k, (a, b, v) in enumerate(segs):
        if b - a < min_len and k > 0:
            lab[a:b] = segs[k-1][2]
    return lab
labels = remove_short_segments(labels, min_len)

# 统计 phase 占比
free_ratio    = np.mean(labels == 1) * 100
contact_ratio = np.mean(labels == 0) * 100
n_transitions = np.sum(np.diff(labels) != 0)

print(f"{mode} demo_{demo_id}: T={T}, {ts0[-1]:.1f}s")
print(f"  v_ee 25th-pct 阈值 = {thr:.4f} m/s")
print(f"  free  (末端移动) 占比 = {free_ratio:.0f}%")
print(f"  contact(末端静止) 占比 = {contact_ratio:.0f}%")
print(f"  phase 切换次数 = {n_transitions}")

# ─── 找 free->contact 切换点，用于抽帧核对 ───────────────────────────────────
transitions = np.where(np.diff(labels) != 0)[0]
f2c = [i for i in transitions if labels[i] == 1 and labels[i+1] == 0]  # free转contact
print(f"  free->contact 切换点帧号: {f2c}")

# ─── 画时序图 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

def shade_phases(ax):
    """背景按 phase 着色：free=蓝, contact=红"""
    s = 0
    for i in range(1, T+1):
        if i == T or labels[i] != labels[s]:
            color = 'tab:blue' if labels[s] == 1 else 'tab:red'
            ax.axvspan(ts0[s], ts0[min(i, T-1)], color=color, alpha=0.10)
            s = i

# 上图：v_ee
ax = axes[0]
ax.plot(ts0, v_ee, color='black', lw=1.3, label='v_ee (end-effector speed)')
ax.axhline(thr, color='green', ls='--', lw=1, label=f'25th-pct threshold={thr:.3f}')
shade_phases(ax)
ax.set_ylabel("v_ee (m/s)")
ax.set_title(f"{mode} demo_{demo_id}  |  blue=free move, red=contact  "
             f"(free {free_ratio:.0f}% / contact {contact_ratio:.0f}%)")
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

# 下图：a_hand
ax = axes[1]
ax.plot(ts0, a_hand, color='purple', lw=1.3, label='a_hand (finger joint speed)')
shade_phases(ax)
# 标注 free->contact 切换点
for i in f2c:
    ax.axvline(ts0[i], color='orange', ls=':', lw=1.5)
    ax.text(ts0[i], ax.get_ylim()[1]*0.9, f'f{i}', color='orange', fontsize=8)
ax.set_ylabel("a_hand (rad/s)")
ax.set_xlabel("time (s)")
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "phase_segmentation")
os.makedirs(OUT, exist_ok=True)
out_fig = os.path.join(OUT, f"phase_{mode}_demo{demo_id}.png")
fig.savefig(out_fig, dpi=110)
print(f"  时序图: {out_fig}")

# ─── 抽帧核对：在第一个 free->contact 切换点前后各抽一帧 side_rgb ──────────────
if f2c:
    cam_key = "side_rgb"
    ti = f2c[0]
    idxs = [max(0, ti-10), ti, min(T-1, ti+10)]   # 切换前、切换点、切换后
    fig2, axs2 = plt.subplots(1, 3, figsize=(13, 4))
    for k, fi in enumerate(idxs):
        axs2[k].imshow(z[cam_key][fi])
        ph = 'free' if labels[fi]==1 else 'contact'
        axs2[k].set_title(f"frame {fi}  t={ts0[fi]:.2f}s  [{ph}]\n"
                          f"v_ee={v_ee[fi]:.3f} a_hand={a_hand[fi]:.3f}", fontsize=10)
        axs2[k].axis('off')
    fig2.suptitle(f"{mode} demo_{demo_id}: frames around first free->contact transition (frame {ti})")
    fig2.tight_layout()
    out_fr = os.path.join(OUT, f"phase_{mode}_demo{demo_id}_frames.png")
    fig2.savefig(out_fr, dpi=110)
    print(f"  抽帧核对图: {out_fr}")
