"""
gap_by_phase robustness: 下采样到同量级 + bootstrap CI
确认 contact gap >> free gap 不是样本量 artifact。
"""
import zarr, numpy as np, glob, os
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE="/home/fa_team/roamlab/finger_aloha/software/ditto_data/traning_data"
def load(folder):
    eps=[]
    for d in sorted(glob.glob(f"{folder}/demo_*")):
        z=zarr.open(os.path.join(d,"data.zarr"),'r'); q=z["follower_joint_states"][:]
        dq=np.vstack([np.zeros((1,7)),np.diff(q,axis=0)])
        eff=np.linalg.norm(uniform_filter1d(z["follower_joint_efforts"][:],5,axis=0,mode='nearest'),axis=1)
        eps.append(dict(q=q,dq=dq,eff=eff))
    return eps
tel=load(f"{BASE}/uncap_teleop_xarm_7dof_05_18_2026"); itw=load(f"{BASE}/uncap_itw_xarm_7dof_05_18_2026")
eff_all=np.concatenate([e["eff"] for e in tel]); kde=gaussian_kde(eff_all)
xs=np.linspace(eff_all.min(),np.percentile(eff_all,99),600); dens=kde(xs)
m=(xs>=140)&(xs<=300); theta_e=xs[m][np.argmin(dens[m])]
for e in tel: e["gt"]=(e["eff"]>theta_e).astype(int)
Xall=np.vstack([e["q"] for e in tel]); yall=np.concatenate([e["gt"] for e in tel])
sc=StandardScaler().fit(Xall); lr=LogisticRegression(max_iter=1000).fit(sc.transform(Xall),yall)
def hyst(p,hi=0.6,lo=0.4,k=5):
    T=len(p);lab=np.zeros(T,int);s=1 if p[0]>hi else 0
    for i in range(T):
        if s==0 and p[i]>hi:s=1
        elif s==1 and p[i]<lo:s=0
        lab[i]=s
    segs,st=[],0
    for i in range(1,T+1):
        if i==T or lab[i]!=lab[st]:segs.append([st,i,lab[st]]);st=i
    for idx,(a,b,v) in enumerate(segs):
        if b-a<k and idx>0:lab[a:b]=segs[idx-1][2]
    return lab
for eps in (tel,itw):
    for e in eps: e["phase"]=hyst(lr.predict_proba(sc.transform(e["q"]))[:,1])
def collect(eps,pv):
    out=[]
    for e in eps:
        mk=(e["phase"]==pv);mk[0]=False;out.append(e["dq"][mk])
    return np.vstack(out)

rng=np.random.default_rng(0); B=200
print("="*60); print("gap robustness: 下采样到 min(n) + bootstrap (B=200)"); print("="*60)
for pv,pname in [(1,"contact"),(0,"free")]:
    idq=collect(itw,pv); tdq=collect(tel,pv)
    both=np.vstack([idq,tdq]); mu,sd=both.mean(0),both.std(0)+1e-9
    iz=(idq-mu)/sd; tz=(tdq-mu)/sd
    n=min(len(iz),len(tz))
    boot=[]
    for _ in range(B):
        ii=rng.choice(len(iz),n,replace=True); tt=rng.choice(len(tz),n,replace=True)
        boot.append(np.mean([wasserstein_distance(iz[ii,d],tz[tt,d]) for d in range(7)]))
    boot=np.array(boot)
    print(f"  [{pname}] n_each={n}  W1 mean={boot.mean():.3f}  "
          f"std={boot.std():.3f}  95%CI=[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")
print("\n  若 contact CI 下界 > free CI 上界 => gap 真实, 非样本量 artifact")
