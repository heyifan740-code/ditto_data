"""
Step 0-4 闭环：q-classifier 切 phase + 跨模式 Δq gap + tracking_err 缺失通道
(1) teleop force真值训 q(7)->contact 的 LR(episode-split报AUC)，切 teleop+ITW 两侧
(2) ITW 被判 contact 的帧抽15个导 side_rgb 缩略图(ITW无palm_rgb)做人工 spot check
(3) per-phase ITW vs teleop Δq 分布距离(z-score + per-dim W1 取平均, 标 n)
(4) tracking_err: teleop contact 段内分布; ITW 恒为0(结构性)
"""
import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
TELEOP = f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"
ITW    = f"{BASE}/uncap_itw_xarm_7dof_05_18_2026"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "closure")
os.makedirs(OUT, exist_ok=True)

def load(folder):
    eps=[]
    for d in sorted(glob.glob(f"{folder}/demo_*")):
        z=zarr.open(os.path.join(d,"data.zarr"),'r')
        q=z["follower_joint_states"][:]
        dq=np.vstack([np.zeros((1,7)), np.diff(q,axis=0)])
        terr=np.linalg.norm(q - z["leader_joint_states"][:], axis=1)
        eff=np.linalg.norm(uniform_filter1d(z["follower_joint_efforts"][:],5,axis=0,mode='nearest'),axis=1)
        eps.append(dict(path=d, q=q, dq=dq, terr=terr, eff=eff, T=len(q)))
    return eps

tel=load(TELEOP); itw=load(ITW)

# ── force 真值 θ_e ──
eff_all=np.concatenate([e["eff"] for e in tel])
kde=gaussian_kde(eff_all); xs=np.linspace(eff_all.min(),np.percentile(eff_all,99),600)
dens=kde(xs); m=(xs>=140)&(xs<=300); theta_e=xs[m][np.argmin(dens[m])]
for e in tel: e["gt"]=(e["eff"]>theta_e).astype(int)

# ════════ (1) 训 q->contact LR ════════════════════════════════════════════
rng=np.random.default_rng(42); order=rng.permutation(len(tel))
tr,va=order[:35],order[35:]
def stackq(idx):
    X=np.vstack([tel[i]["q"] for i in idx]); y=np.concatenate([tel[i]["gt"] for i in idx]); return X,y
Xtr,ytr=stackq(tr); Xva,yva=stackq(va)
sc_cv=StandardScaler().fit(Xtr)
lr_cv=LogisticRegression(max_iter=1000).fit(sc_cv.transform(Xtr),ytr)
val_auc=roc_auc_score(yva, lr_cv.predict_proba(sc_cv.transform(Xva))[:,1])

# 最终切分模型：全 teleop 训
Xall=np.vstack([e["q"] for e in tel]); yall=np.concatenate([e["gt"] for e in tel])
sc=StandardScaler().fit(Xall)
lr=LogisticRegression(max_iter=1000).fit(sc.transform(Xall),yall)

def hyst_prob(p, hi=0.6, lo=0.4, k=5):
    T=len(p); lab=np.zeros(T,int); s=1 if p[0]>hi else 0
    for i in range(T):
        if s==0 and p[i]>hi: s=1
        elif s==1 and p[i]<lo: s=0
        lab[i]=s
    segs,st=[],0
    for i in range(1,T+1):
        if i==T or lab[i]!=lab[st]: segs.append([st,i,lab[st]]); st=i
    for idx,(a,b,v) in enumerate(segs):
        if b-a<k and idx>0: lab[a:b]=segs[idx-1][2]
    return lab

def segment(eps):
    for e in eps:
        p=lr.predict_proba(sc.transform(e["q"]))[:,1]
        e["prob"]=p; e["phase"]=hyst_prob(p)   # 1=contact,0=free
segment(tel); segment(itw)

tel_c=np.mean(np.concatenate([e["phase"] for e in tel]))*100
itw_c=np.mean(np.concatenate([e["phase"] for e in itw]))*100
print("="*60)
print("(1) q->contact LR phase 切分")
print("="*60)
print(f"  episode-split val AUC = {val_auc:.3f}")
print(f"  teleop contact 占比 = {tel_c:.1f}%")
print(f"  ITW    contact 占比 = {itw_c:.1f}%  (构型推断,非真实力接触)")

# ════════ (2) ITW contact 帧 spot check (side_rgb) ════════════════════════
contact_frames=[]
for ei,e in enumerate(itw):
    idx=np.where(e["phase"]==1)[0]
    for fi in idx: contact_frames.append((ei,fi))
rng2=np.random.default_rng(7)
pick=[contact_frames[i] for i in rng2.choice(len(contact_frames),15,replace=False)]
fig,axs=plt.subplots(3,5,figsize=(15,9))
for ax,(ei,fi) in zip(axs.ravel(),pick):
    z=zarr.open(os.path.join(itw[ei]["path"],"data.zarr"),'r')
    ax.imshow(z["side_rgb"][fi]); ax.axis('off')
    ax.set_title(f"d{ei} f{fi} p={itw[ei]['prob'][fi]:.2f}",fontsize=8)
fig.suptitle("(2) ITW frames judged CONTACT by q-classifier  [side_rgb; ITW has no palm_rgb]",fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/itw_contact_spotcheck.png",dpi=100)
print(f"\n(2) saved itw_contact_spotcheck.png (15 ITW contact frames)")

# ════════ (3) per-phase Δq 分布距离 (z-score + per-dim W1) ═════════════════
# 收集每 phase 每侧的 Δq (跳过每条 demo 首帧 Δq=0)
def collect_dq(eps, phase_val):
    out=[]
    for e in eps:
        mask=(e["phase"]==phase_val); mask[0]=False
        out.append(e["dq"][mask])
    return np.vstack(out)

print("\n"+"="*60); print("(3) per-phase ITW vs teleop Δq 分布距离"); print("="*60)
phase_names={1:"contact",0:"free"}
gap_by_phase={}; n_by_phase={}
for pv,pname in phase_names.items():
    itw_dq=collect_dq(itw,pv); tel_dq=collect_dq(tel,pv)
    # z-score: 每维用两侧并集 mean/std
    both=np.vstack([itw_dq,tel_dq])
    mu,sd=both.mean(0),both.std(0)+1e-9
    iz=(itw_dq-mu)/sd; tz=(tel_dq-mu)/sd
    w1s=[wasserstein_distance(iz[:,d],tz[:,d]) for d in range(7)]
    gap_by_phase[pname]=np.mean(w1s); n_by_phase[pname]=(len(itw_dq),len(tel_dq))
    print(f"  [{pname}] mean W1(7dim)={np.mean(w1s):.3f}  "
          f"n_itw={len(itw_dq)} n_tel={len(tel_dq)}")
    print(f"      per-dim W1: {np.round(w1s,3)}")

fig,ax=plt.subplots(figsize=(7,5))
names=list(gap_by_phase.keys()); vals=[gap_by_phase[n] for n in names]
bars=ax.bar(names,vals,color=['tab:red','tab:blue'],alpha=0.75)
for b,nm in zip(bars,names):
    ni,nt=n_by_phase[nm]
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f"W1={b.get_height():.3f}\nn_itw={ni}\nn_tel={nt}",ha='center',fontsize=9)
ax.set_ylabel("mean per-dim Wasserstein-1 (z-scored Δq)")
ax.set_title("(3) ITW vs Teleop hand-action gap, by phase")
ax.grid(alpha=0.3,axis='y')
fig.tight_layout(); fig.savefig(f"{OUT}/gap_by_phase.png",dpi=110)
print(f"  saved gap_by_phase.png")

# ════════ (4) tracking_err 缺失通道 ════════════════════════════════════════
tel_terr_contact=np.concatenate([e["terr"][e["phase"]==1] for e in tel])
tel_terr_free=np.concatenate([e["terr"][e["phase"]==0] for e in tel])
itw_terr_all=np.concatenate([e["terr"] for e in itw])
print("\n"+"="*60); print("(4) tracking_err 缺失通道量化"); print("="*60)
print(f"  teleop contact 段 tracking_err: mean={tel_terr_contact.mean():.4f} "
      f"median={np.median(tel_terr_contact):.4f} P90={np.percentile(tel_terr_contact,90):.4f}")
print(f"  teleop free    段 tracking_err: mean={tel_terr_free.mean():.4f}")
print(f"  ITW (全部)     tracking_err: mean={itw_terr_all.mean():.6f} max={itw_terr_all.max():.6f}  => 恒为0(结构性)")

fig,ax=plt.subplots(figsize=(9,5))
bins=np.linspace(0,np.percentile(tel_terr_contact,99),60)
ax.hist(tel_terr_free,bins=bins,density=True,alpha=0.5,color='tab:blue',label='teleop free')
ax.hist(tel_terr_contact,bins=bins,density=True,alpha=0.6,color='tab:red',label='teleop contact')
ax.axvline(0,color='black',lw=3)
ax.text(0.001,ax.get_ylim()[1]*0.5,"ITW ≡ 0\n(structural,\nfollower=leader mirror)",
        fontsize=11,color='darkred',fontweight='bold')
ax.set_xlabel("tracking_err = ||follower_q - leader_q||")
ax.set_ylabel("density")
ax.set_title("(4) Contact signal (tracking_err): teleop encodes it, ITW structurally 0")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/tracking_err_missing.png",dpi=110)
print(f"  saved tracking_err_missing.png")

print("\n"+"="*60)
print("Step 0-4 闭环完成。三图: itw_contact_spotcheck / gap_by_phase / tracking_err_missing")
print("="*60)
