"""
归因：AUC 0.98 到底来自哪个特征？
对比 只用q / 只用速度类(Δq+v_ee) / 单帧q / 全部，定位 force 信息载体。
若 q 携带而速度类不携带 => force 与"手构型"相关、与"运动快慢"无关。
"""
import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
TELEOP = f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"
WIN = 10

def speed_norm(x, ts, s=5):
    xs = uniform_filter1d(x, s, axis=0, mode='nearest'); dx = np.diff(xs, axis=0)
    dt = np.clip(np.diff(ts), 1e-3, None); sp = np.linalg.norm(dx, axis=1)/dt
    return np.concatenate([[sp[0]], sp])

eps = []
for d in sorted(glob.glob(f"{TELEOP}/demo_*")):
    z = zarr.open(os.path.join(d, "data.zarr"), 'r'); ts = z["timestamp"][:]
    q  = z["follower_joint_states"][:]
    dq = np.vstack([np.zeros((1,7)), np.diff(q, axis=0)])
    vee = speed_norm(z["arm_obs"][:][:, :3], ts)
    eff = np.linalg.norm(uniform_filter1d(z["follower_joint_efforts"][:],5,axis=0,mode='nearest'),axis=1)
    eps.append(dict(feat=np.hstack([q, dq, vee[:,None]]), eff=eff))   # 列: q0-6, dq7-13, vee14

eff_all = np.concatenate([e["eff"] for e in eps])
kde = gaussian_kde(eff_all); xs = np.linspace(eff_all.min(), np.percentile(eff_all,99),600)
dens = kde(xs); m=(xs>=140)&(xs<=300); theta_e = xs[m][np.argmin(dens[m])]
for e in eps: e["gt"] = (e["eff"]>theta_e).astype(int)

rng = np.random.default_rng(42); order = rng.permutation(len(eps))
tr_ep, va_ep = order[:35], order[35:]

def build(ep_idx, cols, win):
    X,y=[],[]
    for i in ep_idx:
        F=eps[i]["feat"][:,cols]; g=eps[i]["gt"]; T=len(g)
        for t in range(win-1,T):
            X.append(F[t-win+1:t+1].ravel()); y.append(g[t])
    return np.array(X), np.array(y)

def run(cols, win, tag):
    Xtr,ytr=build(tr_ep,cols,win); Xva,yva=build(va_ep,cols,win)
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=1000).fit(sc.transform(Xtr),ytr)
    auc=roc_auc_score(yva,clf.predict_proba(sc.transform(Xva))[:,1])
    print(f"  {tag:<32}{Xtr.shape[1]:>5}d   val AUC={auc:.3f}")

print(f"θ_e={theta_e:.1f}  (LogReg, episode-split val AUC)\n")
print("="*60)
print("特征消融：force-contact 的信息藏在哪个特征里？")
print("="*60)
run(list(range(0,7)),  WIN, "q only (0.5s window)")
run(list(range(7,15)), WIN, "speed only (Δq+v_ee, 0.5s win)")
run(list(range(0,7)),  1,   "q only (single frame)")
run(list(range(7,15)), 1,   "speed only (single frame)")
run(list(range(0,15)), WIN, "ALL (q+Δq+v_ee, 0.5s win)")
