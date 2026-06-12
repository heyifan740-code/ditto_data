"""
Step 1 深化：把"运动学能否预测 force-contact"做到监督学习上限
(1) a_hand manipulation 段对 force-contact 的覆盖率（按 episode）
(2) LR + 小 MLP，0.5s 滑窗 [q,Δq,v_ee] 特征预测 frame 级 force-contact，报 AUC
    train/val 严格按 episode 切；额外对照"按帧随机切"展示泄漏导致的 AUC 虚高
(3) manipulation 段内 force 微观结构图（高力 micro-episode 着色）
"""
import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
TELEOP = f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "force_analysis")
os.makedirs(OUT, exist_ok=True)
WIN = 10   # 0.5s @ 20Hz = 10 帧滑窗

def speed_norm(x, ts, s=5):
    xs = uniform_filter1d(x, s, axis=0, mode='nearest'); dx = np.diff(xs, axis=0)
    dt = np.clip(np.diff(ts), 1e-3, None); sp = np.linalg.norm(dx, axis=1)/dt
    return np.concatenate([[sp[0]], sp])

def hysteresis(sig, th_high, th_low, k):
    """a_hand 高=manipulation(1)。迟滞 + 最短段长 k"""
    T = len(sig); lab = np.zeros(T, int); state = 1 if sig[0] > th_high else 0
    for i in range(T):
        if state == 0 and sig[i] > th_high: state = 1
        elif state == 1 and sig[i] < th_low: state = 0
        lab[i] = state
    segs, s = [], 0
    for i in range(1, T+1):
        if i == T or lab[i] != lab[s]: segs.append([s, i, lab[s]]); s = i
    for idx, (a, b, v) in enumerate(segs):
        if b - a < k and idx > 0: lab[a:b] = segs[idx-1][2]
    return lab

# ════════════ 加载 + per-episode 信号/特征/标签 ════════════════════════════
demo_dirs = sorted(glob.glob(f"{TELEOP}/demo_*"))
eps = []
for d in demo_dirs:
    z = zarr.open(os.path.join(d, "data.zarr"), 'r'); ts = z["timestamp"][:]
    q   = z["follower_joint_states"][:]                       # (T,7)
    dq  = np.vstack([np.zeros((1,7)), np.diff(q, axis=0)])    # (T,7) 帧间差
    vee = speed_norm(z["arm_obs"][:][:, :3], ts)              # (T,)
    ah  = speed_norm(q, ts)                                   # (T,) a_hand
    eff = np.linalg.norm(uniform_filter1d(z["follower_joint_efforts"][:],5,axis=0,mode='nearest'), axis=1)
    feat_frame = np.hstack([q, dq, vee[:, None]])             # (T,15) 每帧特征
    eps.append(dict(q=q, dq=dq, vee=vee, ah=ah, eff=eff, feat=feat_frame))

# θ_e (force 谷底) -> contact 真值
eff_all = np.concatenate([e["eff"] for e in eps])
kde = gaussian_kde(eff_all); xs = np.linspace(eff_all.min(), np.percentile(eff_all,99), 600)
dens = kde(xs); m = (xs>=140)&(xs<=300); theta_e = xs[m][np.argmin(dens[m])]
for e in eps: e["gt"] = (e["eff"] > theta_e).astype(int)
print(f"θ_e={theta_e:.1f}, 全量 contact={np.concatenate([e['gt'] for e in eps]).mean()*100:.1f}%\n")

# ════════════ 任务1: manipulation 段覆盖率 ═══════════════════════════════════
ah_all = np.concatenate([e["ah"] for e in eps])
TH_HI, TH_LO, K = np.percentile(ah_all,55), np.percentile(ah_all,40), 5
cov_list, manip_ratio_list, purity_list = [], [], []
for e in eps:
    manip = hysteresis(e["ah"], TH_HI, TH_LO, K)   # 1=manipulation
    gt = e["gt"]
    n_contact = gt.sum()
    if n_contact == 0:
        continue
    coverage = np.sum((gt==1)&(manip==1)) / n_contact          # force-contact 落在段内比例
    manip_ratio = manip.mean()                                  # 段占整条比例
    purity = (np.sum((gt==1)&(manip==1)) / manip.sum()) if manip.sum() else 0  # 段内 contact 密度
    cov_list.append(coverage); manip_ratio_list.append(manip_ratio); purity_list.append(purity)
cov = np.array(cov_list); mr = np.array(manip_ratio_list); pu = np.array(purity_list)

print("="*64)
print("任务1: a_hand manipulation 段 对 force-contact 的覆盖率（按 episode）")
print("="*64)
print(f"  manipulation 段参数: θ_high={TH_HI:.3f} θ_low={TH_LO:.3f} k={K}")
print(f"  覆盖率 coverage (force-contact 落在段内): mean={cov.mean():.3f} "
      f"std={cov.std():.3f} min={cov.min():.3f} max={cov.max():.3f}")
print(f"  段占比 manip_ratio:                       mean={mr.mean():.3f} std={mr.std():.3f}")
print(f"  段内纯度 purity (段内是 contact 的比例):    mean={pu.mean():.3f} std={pu.std():.3f}")
print(f"  解读: 覆盖率高+纯度低 => 段是 contact 的'容器'但不精确(段内混大量非接触帧)")

# ════════════ 任务2: 监督学习预测 force-contact (AUC) ════════════════════════
def build_windows(ep_indices):
    """对指定 episode 构造滑窗特征 X(N,WIN*15) 和标签 y(N,)，窗口不跨 episode"""
    X, y = [], []
    for i in ep_indices:
        e = eps[i]; F = e["feat"]; g = e["gt"]; T = len(g)
        for t in range(WIN-1, T):                  # 因果窗口 [t-9, t]，预测第 t 帧
            X.append(F[t-WIN+1:t+1].ravel()); y.append(g[t])
    return np.array(X), np.array(y)

# --- 按 episode 切（正确做法）---
rng = np.random.default_rng(42)
order = rng.permutation(len(eps))
n_tr = int(len(eps)*0.7)
tr_ep, va_ep = order[:n_tr], order[n_tr:]
Xtr, ytr = build_windows(tr_ep); Xva, yva = build_windows(va_ep)
sc = StandardScaler().fit(Xtr); Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

def fit_report(name, clf):
    clf.fit(Xtr_s, ytr)
    auc_tr = roc_auc_score(ytr, clf.predict_proba(Xtr_s)[:,1])
    auc_va = roc_auc_score(yva, clf.predict_proba(Xva_s)[:,1])
    return name, auc_tr, auc_va

results_ep = [
    fit_report("LogReg", LogisticRegression(max_iter=1000, C=1.0)),
    fit_report("MLP(64,16)", MLPClassifier(hidden_layer_sizes=(64,16), max_iter=400,
                                           early_stopping=True, random_state=0)),
]

# --- 故意按帧随机切（错误做法，演示泄漏）---
Xall, yall = build_windows(np.arange(len(eps)))
sca = StandardScaler().fit(Xall); Xall_s = sca.transform(Xall)
idx = rng.permutation(len(yall)); cut = int(len(yall)*0.7)
fi_tr, fi_va = idx[:cut], idx[cut:]
clf_leak = MLPClassifier(hidden_layer_sizes=(64,16), max_iter=400, early_stopping=True, random_state=0)
clf_leak.fit(Xall_s[fi_tr], yall[fi_tr])
auc_leak = roc_auc_score(yall[fi_va], clf_leak.predict_proba(Xall_s[fi_va])[:,1])

print("\n" + "="*64)
print("任务2: 运动学序列特征预测 force-contact 的 AUC 上限")
print("="*64)
print(f"  特征: 0.5s滑窗(10帧)×[q(7),Δq(7),v_ee(1)]=150维")
print(f"  train/val: 按 episode 切 ({n_tr} train / {len(eps)-n_tr} val demos)")
print(f"  {'model':<14}{'train AUC':>11}{'val AUC':>10}")
print("  " + "-"*35)
for name, atr, ava in results_ep:
    print(f"  {name:<14}{atr:>11.3f}{ava:>10.3f}")
print(f"\n  [对照] 同 MLP 故意按帧随机切 val AUC = {auc_leak:.3f}  <- 泄漏虚高")
best_ep_auc = max(a for _,_,a in results_ep)
print(f"  泄漏抬高量: {auc_leak - best_ep_auc:+.3f}")

# ════════════ 任务3: manipulation 段内 force 微观结构 ════════════════════════
demo_id = 0; e = eps[demo_id]
z = zarr.open(f"{TELEOP}/demo_{demo_id}/data.zarr",'r'); ts0 = z["timestamp"][:]-z["timestamp"][0]
manip = hysteresis(e["ah"], TH_HI, TH_LO, K)
fig, ax = plt.subplots(figsize=(13,4.5))
ax.plot(ts0, e["eff"], color='black', lw=1.2, label='||efforts||')
ax.axhline(theta_e, color='red', ls='--', lw=1, label=f'θ_e={theta_e:.0f}')
# manipulation 段底色（蓝）
s=0
for i in range(1,len(manip)+1):
    if i==len(manip) or manip[i]!=manip[s]:
        if manip[s]==1: ax.axvspan(ts0[s], ts0[min(i,len(manip)-1)], color='tab:blue', alpha=0.08)
        s=i
# 高力 micro-episode 着色（红）：连续 eff>θ_e 段
hi = (e["eff"]>theta_e).astype(int); s=0
for i in range(1,len(hi)+1):
    if i==len(hi) or hi[i]!=hi[s]:
        if hi[s]==1: ax.axvspan(ts0[s], ts0[min(i,len(hi)-1)], color='red', alpha=0.22)
        s=i
ax.set_xlabel("time (s)"); ax.set_ylabel("||efforts||")
ax.set_title(f"teleop demo_{demo_id}: force micro-structure  "
             f"(blue=a_hand manipulation seg, red=high-force micro-episode)")
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/force_microstructure.png", dpi=110)
print(f"\nsaved force_microstructure.png")

# 覆盖率直方图
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.hist(cov, bins=20, color='tab:green', alpha=0.7)
ax2.axvline(cov.mean(), color='red', ls='--', label=f'mean={cov.mean():.2f}')
ax2.set_xlabel("coverage (force-contact frames inside manipulation seg)")
ax2.set_ylabel("# episodes"); ax2.set_title("Task1: per-episode coverage")
ax2.legend(); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(f"{OUT}/coverage_per_episode.png", dpi=110)
print(f"saved coverage_per_episode.png")
