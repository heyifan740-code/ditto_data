"""
机制判定：q 预测 force 是"接触偏移(A)"还是"构型/进度(B)"？
关键对比：teleop follower_q(真实,含接触偏移) vs leader_q(指令,无接触偏移)
  leader_q 性质 ≈ ITW 的无接触标称构型。
  follower 明显 > leader => 机制A(接触偏移携带额外force信息) => ITW 论点成立。
辅证：tracking_err=‖follower_q-leader_q‖ 与 force 的相关；ITW 的 tracking_err≈0。
"""
import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
TELEOP = f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"
ITW = f"{BASE}/uncap_itw_xarm_7dof_05_18_2026"

def load(folder):
    eps=[]
    for d in sorted(glob.glob(f"{folder}/demo_*")):
        z=zarr.open(os.path.join(d,"data.zarr"),'r')
        fq=z["follower_joint_states"][:]; lq=z["leader_joint_states"][:]
        eff=np.linalg.norm(uniform_filter1d(z["follower_joint_efforts"][:],5,axis=0,mode='nearest'),axis=1)
        eps.append(dict(fq=fq, lq=lq, eff=eff, terr=np.linalg.norm(fq-lq,axis=1)))
    return eps

tel = load(TELEOP)
eff_all=np.concatenate([e["eff"] for e in tel])
kde=gaussian_kde(eff_all); xs=np.linspace(eff_all.min(),np.percentile(eff_all,99),600)
dens=kde(xs); m=(xs>=140)&(xs<=300); theta_e=xs[m][np.argmin(dens[m])]
for e in tel: e["gt"]=(e["eff"]>theta_e).astype(int)

rng=np.random.default_rng(42); order=rng.permutation(len(tel))
tr,va=order[:35],order[35:]
def build(idx,key):
    X,y=[],[]
    for i in idx:
        F=tel[i][key]; g=tel[i]["gt"]
        for t in range(len(g)): X.append(F[t]); y.append(g[t])
    return np.array(X),np.array(y)
def auc_of(key):
    Xtr,ytr=build(tr,key); Xva,yva=build(va,key)
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=1000).fit(sc.transform(Xtr),ytr)
    return roc_auc_score(yva,clf.predict_proba(sc.transform(Xva))[:,1])

print(f"θ_e={theta_e:.1f}  (single-frame q, episode-split val AUC)\n")
print("="*58)
print("机制判定: follower_q vs leader_q 预测 force-contact")
print("="*58)
a_fq=auc_of("fq"); a_lq=auc_of("lq")
print(f"  follower_q (真实,含接触偏移)  val AUC = {a_fq:.3f}")
print(f"  leader_q   (指令,无接触偏移)  val AUC = {a_lq:.3f}")
print(f"  差值 (follower - leader)             = {a_fq-a_lq:+.3f}")
verdict = "机制A(接触偏移): follower 的接触卡位携带额外 force 信息" if a_fq-a_lq>0.03 \
          else "机制B(构型/进度): 两者接近, q 主要编码任务构型而非接触偏移"
print(f"  => {verdict}")

print("\n" + "="*58)
print("辅证: 跟踪误差 ‖follower_q - leader_q‖ 与 force 的关系")
print("="*58)
terr_all=np.concatenate([e["terr"] for e in tel]); gt_all=np.concatenate([e["gt"] for e in tel])
rho=spearmanr(terr_all,eff_all).statistic
# tracking_err 单变量预测 force-contact 的 AUC
pos=terr_all[gt_all==1]; neg=terr_all[gt_all==0]
rng2=np.random.default_rng(0); a=rng2.choice(pos,50000); b=rng2.choice(neg,50000)
auc_terr=np.mean(a>b)+0.5*np.mean(a==b)
print(f"  teleop: Spearman(terr, ||efforts||) = {rho:+.3f}")
print(f"  teleop: tracking_err 单变量预测 force-contact AUC = {auc_terr:.3f}")
print(f"  teleop: terr 均值 contact={terr_all[gt_all==1].mean():.4f} "
      f"free={terr_all[gt_all==0].mean():.4f} "
      f"比={terr_all[gt_all==1].mean()/terr_all[gt_all==0].mean():.2f}x")

itw=load(ITW)
itw_terr=np.concatenate([e["terr"] for e in itw])
print(f"\n  ITW: tracking_err 均值={itw_terr.mean():.6f} max={itw_terr.max():.6f}")
print(f"  ITW: follower==leader 完全镜像 => tracking_err 恒 0 => 无接触偏移编码")
print(f"\n结论: 若 follower_q>leader_q 且 terr 与 force 正相关,")
print(f"      则 teleop 的 q 通过'position控制接触偏移'编码 force;")
print(f"      ITW 镜像 q 不含此编码 => 'ITW 缺 force'论点成立(机制性).")
