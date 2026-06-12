"""
Step 3: 数据可视化 —— 逐帧对比图像 + 力/电流 + 动作
把一条 demo 渲染成 mp4：左侧相机画面，右侧同步滚动的信号曲线。

用法:
    python 03_visualize_demo.py teleop 0     # 可视化 teleop 的 demo_0
    python 03_visualize_demo.py itw 5        # 可视化 itw 的 demo_5
"""

import zarr
import numpy as np
import cv2                              # 写视频
import matplotlib
matplotlib.use("Agg")                  # 无显示器模式，只渲染到内存不弹窗
import matplotlib.pyplot as plt
import sys, os

# ─── 命令行参数 ──────────────────────────────────────────────────────────────
mode    = sys.argv[1] if len(sys.argv) > 1 else "teleop"   # itw 或 teleop
demo_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # demo 编号

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
DATASET = {
    "itw":    "uncap_itw_xarm_7dof_05_18_2026",
    "teleop": "uncap_teleop_xarm_7dof_05_18_2026",
}[mode]

zarr_path = os.path.join(BASE, DATASET, f"demo_{demo_id}", "data.zarr")
z = zarr.open(zarr_path, mode='r')

# ─── 读取数据到内存 ──────────────────────────────────────────────────────────
# 图像：side_rgb 两种模式都有，作为主画面
side   = z["side_rgb"][:]                          # (T, 480, 640, 3) uint8
# 第二个相机：itw 是 leader_rgb，teleop 是 palm_rgb
cam2_key = "palm_rgb" if "palm_rgb" in z else "leader_rgb"
cam2   = z[cam2_key][:]                             # (T, 480, 640, 3)

effort = z["follower_joint_efforts"][:]            # (T, 7) 力/电流信号
action = z["hand_action"][:]                       # (T, 7) 手部动作指令
ts     = z["timestamp"][:]                         # (T,)
ts     = ts - ts[0]                                # 从 0 开始的相对时间（秒）

T = side.shape[0]
print(f"可视化 {mode} demo_{demo_id}: {T} 帧, {ts[-1]:.1f}s")

# ─── 预先算好 y 轴范围，让曲线在整段视频里刻度固定（不跳动） ──────────────────
eff_min, eff_max = effort.min(), effort.max()
act_min, act_max = action.min(), action.max()
# 留 5% 边距，避免曲线贴边
eff_pad = (eff_max - eff_min) * 0.05 + 1e-6
act_pad = (act_max - act_min) * 0.05 + 1e-6

# ─── 7 个关节用 7 种颜色 ─────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 1, 7))

# ─── 输出视频设置 ────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "videos")
os.makedirs(OUT, exist_ok=True)
out_path = os.path.abspath(os.path.join(OUT, f"viz_{mode}_demo{demo_id}.mp4"))

# 先渲染第一帧确定画布尺寸
def render_frame(i):
    """渲染第 i 帧为一张 BGR 图像（给 cv2 写视频用）"""
    # 画布：2 行 2 列。左列两个相机，右列两个信号图
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=80)

    # ── 左上：side 相机 ──
    axes[0, 0].imshow(side[i])
    axes[0, 0].set_title(f"side_rgb   |   frame {i}/{T-1}   t={ts[i]:.2f}s",
                         fontsize=11)
    axes[0, 0].axis("off")

    # ── 左下：第二个相机 ──
    axes[1, 0].imshow(cam2[i])
    axes[1, 0].set_title(cam2_key, fontsize=11)
    axes[1, 0].axis("off")

    # ── 右上：力/电流曲线，画到当前帧为止 ──
    ax_eff = axes[0, 1]
    for j in range(7):
        # 画从开头到第 i 帧的曲线
        ax_eff.plot(ts[:i+1], effort[:i+1, j],
                    color=colors[j], lw=1.2, label=f"J{j}")
        # 当前帧位置画一个圆点，强调"现在"
        ax_eff.plot(ts[i], effort[i, j], 'o',
                    color=colors[j], markersize=5)
    ax_eff.set_xlim(0, ts[-1])
    ax_eff.set_ylim(eff_min - eff_pad, eff_max + eff_pad)
    ax_eff.set_title("follower_joint_efforts (motor current -> contact force proxy)",
                     fontsize=11)
    ax_eff.set_xlabel("time (s)")
    ax_eff.set_ylabel("effort")
    ax_eff.legend(loc="upper right", fontsize=7, ncol=2)
    ax_eff.grid(alpha=0.3)

    # ── 右下：手部动作曲线 ──
    ax_act = axes[1, 1]
    for j in range(7):
        ax_act.plot(ts[:i+1], action[:i+1, j], color=colors[j], lw=1.2)
        ax_act.plot(ts[i], action[i, j], 'o', color=colors[j], markersize=5)
    ax_act.set_xlim(0, ts[-1])
    ax_act.set_ylim(act_min - act_pad, act_max + act_pad)
    ax_act.set_title("hand_action (joint position command)", fontsize=11)
    ax_act.set_xlabel("time (s)")
    ax_act.set_ylabel("action")
    ax_act.grid(alpha=0.3)

    fig.tight_layout()

    # 把 matplotlib 画布转成 numpy 数组
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[:, :, :3]   # 去掉 alpha 通道
    plt.close(fig)                          # 关闭释放内存，否则会爆
    return img

# ─── 渲染第一帧，拿到尺寸，初始化视频写入器 ──────────────────────────────────
first = render_frame(0)
h, w = first.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))   # 20 fps = 原始控制频率

# 写第一帧（RGB → BGR）
writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))

# ─── 逐帧渲染并写入 ──────────────────────────────────────────────────────────
for i in range(1, T):
    frame = render_frame(i)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if i % 50 == 0:
        print(f"  已渲染 {i}/{T} 帧")

writer.release()
print(f"\n视频已保存: {out_path}")
print(f"播放速度 20fps = 实时速度")
