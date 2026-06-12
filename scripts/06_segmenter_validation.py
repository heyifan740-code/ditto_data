"""
Step 1 验证：用 teleop 的 force 真值客观评判 a_hand vs v_ee segmenter
思路：teleop 独有 force(efforts)，是 contact 的物理真值。用它标定哪个运动信号
      (a_hand 手指速度 / v_ee 末端速度) 切出的 contact 更接近真实接触。
      在 teleop 上锁定最优规则，再迁移到无 force 的 ITW。

输出：
  (1) efforts_histogram.png   ‖efforts‖ 双峰 + θ_e 谷底 + contact 真值占比
  (2) F1 对照表（打印）        a_hand vs v_ee segmenter 的 帧级 P/R/F1
  (3) ahand_f1_heatmap.png    a_hand 的 θ_high×θ_low grid F1 热力图 + 敏感度
"""

import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
TELEOP = f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "force_analysis")
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════
def speed_norm(x, ts, smooth=5):
    """逐帧速度范数 ‖Δx‖/dt，对齐回长度 T。x:(T,D)"""
    xs = uniform_filter1d(x, size=smooth, axis=0, mode='nearest')
    dx = np.diff(xs, axis=0)
    dt = np.clip(np.diff(ts), 1e-3, None)
    spd = np.linalg.norm(dx, axis=1) / dt
    return np.concatenate([[spd[0]], spd])

def hysteresis_segment(sig, th_high, th_low, k, contact_when='high'):
    """带迟滞的状态机 + 最短段长过滤。返回 0/1 标签（1=contact）。
    contact_when='high': sig 越大越 contact（a_hand）—— 升过 th_high 进，降过 th_low 出
    contact_when='low' : sig 越小越 contact（v_ee）—— 降过 th_low 进，升过 th_high 出
    """
    T = len(sig)
    lab = np.zeros(T, dtype=int)
    if contact_when == 'high':
        state = 1 if sig[0] > th_high else 0
        for i in range(T):
            if state == 0 and sig[i] > th_high: state = 1
            elif state == 1 and sig[i] < th_low: state = 0
            lab[i] = state
    else:  # 'low'
        state = 1 if sig[0] < th_low else 0
        for i in range(T):
            if state == 0 and sig[i] < th_low: state = 1
            elif state == 1 and sig[i] > th_high: state = 0
            lab[i] = state
    # 最短段长 k：短段并入前一段
    segs, s = [], 0
    for i in range(1, T+1):
        if i == T or lab[i] != lab[s]:
            segs.append([s, i, lab[s]]); s = i
    for idx, (a, b, v) in enumerate(segs):
        if b - a < k and idx > 0:
            lab[a:b] = segs[idx-1][2]
    return lab

def prf1(pred, gt):
    """帧级 precision/recall/F1，正类=contact=1"""
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    P = tp / (tp + fp) if (tp+fp) else 0.0
    R = tp / (tp + fn) if (tp+fn) else 0.0
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return P, R, F

# ════════════════════════════════════════════════════════════════════════════
# 1. 加载全量 teleop，per-demo 计算信号（差分不跨 demo 边界）
# ════════════════════════════════════════════════════════════════════════════
demo_dirs = sorted(glob.glob(f"{TELEOP}/demo_*"))
demos = []   # 每条: dict(eff_norm, v_ee, a_hand)
for d in demo_dirs:
    z = zarr.open(os.path.join(d, "data.zarr"), 'r')
    ts   = z["timestamp"][:]
    eff  = uniform_filter1d(z["follower_joint_efforts"][:], 5, axis=0, mode='nearest')
    demos.append({
        "eff_norm": np.linalg.norm(eff, axis=1),          # contact 真值用
        "v_ee":     speed_norm(z["arm_obs"][:][:, :3], ts),
        "a_hand":   speed_norm(z["follower_joint_states"][:], ts),
    })

eff_all = np.concatenate([d["eff_norm"] for d in demos])

# ════════════════════════════════════════════════════════════════════════════
# 2. KDE 找双峰谷底 θ_e，生成帧级 contact 真值
# ════════════════════════════════════════════════════════════════════════════
kde = gaussian_kde(eff_all)
xs  = np.linspace(eff_all.min(), np.percentile(eff_all, 99), 600)
dens = kde(xs)
# 谷底：在主峰右侧 [140,300] 区间找密度最小点
mask = (xs >= 140) & (xs <= 300)
theta_e = xs[mask][np.argmin(dens[mask])]

# 生成真值
for d in demos:
    d["gt"] = (d["eff_norm"] > theta_e).astype(int)
gt_all = np.concatenate([d["gt"] for d in demos])
contact_ratio = gt_all.mean() * 100

print("="*64)
print(f"contact 真值: θ_e (谷底) = {theta_e:.1f}")
print(f"  全量 contact 帧占比 = {contact_ratio:.1f}%  "
      f"(free {100-contact_ratio:.1f}%)")
print("="*64)

# 画 efforts histogram + θ_e
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, np.percentile(eff_all, 99.5), 80)
ax.hist(eff_all, bins=bins, density=True, color='tab:blue', alpha=0.55, label='||efforts||')
ax.plot(xs, dens, 'k-', lw=1.5, label='KDE')
ax.axvline(theta_e, color='red', ls='--', lw=2, label=f'θ_e (valley) = {theta_e:.0f}')
ax.set_xlabel("||follower_joint_efforts|| (7-dim L2 norm)")
ax.set_ylabel("density")
ax.set_title(f"teleop ||efforts|| bimodal distribution  |  "
             f"contact = {contact_ratio:.0f}% of frames")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/efforts_histogram.png", dpi=110)
print(f"saved efforts_histogram.png")

# ════════════════════════════════════════════════════════════════════════════
# 3. 两个 segmenter 默认参数下的 F1 对照
#    a_hand: contact 当 a_hand 高;  v_ee: contact 当 v_ee 低
# ════════════════════════════════════════════════════════════════════════════
def eval_segmenter(signal_key, th_high, th_low, k, contact_when):
    preds = []
    for d in demos:
        preds.append(hysteresis_segment(d[signal_key], th_high, th_low, k, contact_when))
    pred_all = np.concatenate(preds)
    return prf1(pred_all, gt_all)

# 数据驱动的默认阈值（用全量分位数，稳健）
ah_all = np.concatenate([d["a_hand"] for d in demos])
ve_all = np.concatenate([d["v_ee"]   for d in demos])

# a_hand 默认：高=contact。用中位附近做迟滞带
ah_hi_def, ah_lo_def = np.percentile(ah_all, 55), np.percentile(ah_all, 40)
# v_ee 默认：低=contact。
ve_lo_def, ve_hi_def = np.percentile(ve_all, 40), np.percentile(ve_all, 55)

P_a, R_a, F_a = eval_segmenter("a_hand", ah_hi_def, ah_lo_def, 5, 'high')
P_v, R_v, F_v = eval_segmenter("v_ee",   ve_hi_def, ve_lo_def, 5, 'low')

print("\n" + "="*64)
print("Segmenter F1 对照（默认参数, k=5, 以 force 真值为准）")
print("="*64)
print(f"{'segmenter':<12}{'θ_high':>9}{'θ_low':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}")
print("-"*64)
print(f"{'a_hand':<12}{ah_hi_def:>9.3f}{ah_lo_def:>9.3f}{P_a:>11.3f}{R_a:>9.3f}{F_a:>8.3f}")
print(f"{'v_ee':<12}{ve_hi_def:>9.4f}{ve_lo_def:>9.4f}{P_v:>11.3f}{R_v:>9.3f}{F_v:>8.3f}")

# ════════════════════════════════════════════════════════════════════════════
# 4. a_hand 的 grid search: θ_high × θ_low × k，找最优 F1 + 敏感度
# ════════════════════════════════════════════════════════════════════════════
# grid 范围围绕 a_hand 分布设定
th_high_grid = np.round(np.percentile(ah_all, [45,50,55,60,65,70]), 3)
th_low_grid  = np.round(np.percentile(ah_all, [25,30,35,40,45,50]), 3)
k_grid = [3, 5, 7]

best = (-1, None)
records = []
for k in k_grid:
    for thi in th_high_grid:
        for tlo in th_low_grid:
            if tlo >= thi:           # 迟滞要求 low<high
                continue
            P, R, F = eval_segmenter("a_hand", thi, tlo, k, 'high')
            records.append((k, thi, tlo, P, R, F))
            if F > best[0]:
                best = (F, (k, thi, tlo, P, R))

bestF, (bk, bthi, btlo, bP, bR) = best
print("\n" + "="*64)
print("a_hand grid search 最优")
print("="*64)
print(f"  最优: k={bk}, θ_high={bthi:.3f}, θ_low={btlo:.3f}")
print(f"        Precision={bP:.3f}  Recall={bR:.3f}  F1={bestF:.3f}")

# 敏感度：所有合法组合的 F1 分布
allF = np.array([r[5] for r in records])
print(f"  敏感度: F1 在整个 grid 上 mean={allF.mean():.3f} "
      f"std={allF.std():.3f} min={allF.min():.3f} max={allF.max():.3f}")
print(f"          (std 小 => 对参数不敏感, 切分稳健)")

# 热力图：固定最优 k，画 θ_high(y) × θ_low(x) 的 F1
F_mat = np.full((len(th_high_grid), len(th_low_grid)), np.nan)
for (k, thi, tlo, P, R, F) in records:
    if k != bk: continue
    i = list(th_high_grid).index(thi); j = list(th_low_grid).index(tlo)
    F_mat[i, j] = F

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(F_mat, origin='lower', aspect='auto', cmap='viridis')
ax.set_xticks(range(len(th_low_grid)));  ax.set_xticklabels(th_low_grid)
ax.set_yticks(range(len(th_high_grid))); ax.set_yticklabels(th_high_grid)
ax.set_xlabel("θ_low (a_hand)"); ax.set_ylabel("θ_high (a_hand)")
ax.set_title(f"a_hand segmenter F1 heatmap (k={bk}, vs force ground truth)")
for i in range(len(th_high_grid)):
    for j in range(len(th_low_grid)):
        if not np.isnan(F_mat[i, j]):
            ax.text(j, i, f"{F_mat[i,j]:.2f}", ha='center', va='center',
                    color='white' if F_mat[i,j] < bestF*0.9 else 'red', fontsize=8)
fig.colorbar(im, label='F1')
fig.tight_layout(); fig.savefig(f"{OUT}/ahand_f1_heatmap.png", dpi=110)
print(f"saved ahand_f1_heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
# 结论
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("结论")
print("="*64)
winner = "a_hand" if F_a > F_v else "v_ee"
margin = abs(F_a - F_v)
print(f"  默认参数: a_hand F1={F_a:.3f}  vs  v_ee F1={F_v:.3f}  "
      f"=> {winner} 胜 (差 {margin:.3f})")
print(f"  a_hand 调优后最优 F1={bestF:.3f}")
if F_a > F_v and margin > 0.05:
    print(f"  => a_hand 明显胜出，建议锁定 a_hand 规则迁移到 ITW，进 Step 2")
elif margin <= 0.05:
    print(f"  => 两者接近，需进一步看 PR 权衡或联合判据")
