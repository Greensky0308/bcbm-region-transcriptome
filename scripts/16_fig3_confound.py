#!/usr/bin/env python3
"""16 — Fig3「脑区混杂」：在真实 GSE322943 上量化——
 (a) 脑区是 BCBM 转录组的主导变异轴（PCA）；
 (b) 脑区对比产生的 DE 远超随机分组（效应量对比）；
 (c) 组间脑区构成越不平衡，'组间签名'越等于真实脑区效应（r 随 Δ 上升）。
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from setup_style import setup_style
from export_figure import export_figure

B=ROOT
OUT=os.path.join(ROOT, "out"); FIG=f"{OUT}/figdata"; IMG=os.path.join(ROOT, "figures")
setup_style(journal="general",lang="en",use_sciplots=False)
mpl.rcParams.update({"font.family":"Arial","font.size":7,"axes.linewidth":0.6,
    "xtick.major.width":0.6,"ytick.major.width":0.6,"legend.frameon":False,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
C_CB,C_CBR="#D55E00","#0072B2"
cpm=pd.read_csv(f"{FIG}/bulk_cpm.csv",index_col=0)
cpm.index=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0).index.values
bm=pd.read_csv(f"{FIG}/bulk_meta.csv")
M=cpm.values.astype(float); genes=np.array(cpm.index)
loc=bm["location"].values; iscb=(loc=="cerebellum")
cb_idx=np.where(iscb)[0]; cbr_idx=np.where(~iscb)[0]; N=len(loc)

def de(maskA):
    t,p=ttest_ind(M[:,maskA],M[:,~maskA],axis=1,equal_var=False)
    q=multipletests(p,method="fdr_bh")[1]
    return int((q<0.05).sum()), M[:,maskA].mean(1)-M[:,~maskA].mean(1)

# ---------- (a) split-half 复现性（全基因 vs 脑区DE基因）----------
rng2=np.random.default_rng(2)
cbs=rng2.permutation(cb_idx); cbrs=rng2.permutation(cbr_idx)
cbA,cbB=cbs[:len(cbs)//2],cbs[len(cbs)//2:]; cbA2,cbB2=cbrs[:len(cbrs)//2],cbrs[len(cbrs)//2:]
lfc1=M[:,cbA].mean(1)-M[:,cbA2].mean(1); lfc2=M[:,cbB].mean(1)-M[:,cbB2].mean(1)
t0,p0=ttest_ind(M[:,iscb],M[:,~iscb],axis=1,equal_var=False)
sigregion=multipletests(p0,method="fdr_bh")[1]<0.05
r_half=np.corrcoef(lfc1,lfc2)[0,1]
r_half_reg=np.corrcoef(lfc1[sigregion],lfc2[sigregion])[0,1]

# ---------- (b) 脑区 vs 随机分组 DE ----------
n_reg,_=de(iscb)
rng=np.random.default_rng(1); rnd=[]
for _ in range(30):
    m=np.zeros(N,bool); m[rng.choice(N,N//2,replace=False)]=True
    rnd.append(de(m)[0])
rnd_mean,rnd_sd=np.mean(rnd),np.std(rnd)

# ---------- (c) r 随不平衡度 ----------
reg_lfc=M[:,cb_idx].mean(1)-M[:,cbr_idx].mean(1)
nA=N//2; rows=[]
for k in range(17,35,2):
    rng=np.random.default_rng(0); rs=[]
    for _ in range(20):
        A=np.concatenate([rng.choice(cb_idx,k,replace=False),rng.choice(cbr_idx,nA-k,replace=False)])
        m=np.zeros(N,bool); m[A]=True
        rs.append(np.corrcoef(de(m)[1],reg_lfc)[0,1])
    d=(k/nA)-((len(cb_idx)-k)/(N-nA)); rows.append((d,np.mean(rs),np.std(rs)))
sim=pd.DataFrame(rows,columns=["delta","r","sd"]); sim.to_csv(f"{OUT}/16_sim_imbalance.csv",index=False)
print(f"(a) split-half r all={r_half:.3f} | region-DE r={r_half_reg:.3f}")
print(f"(b) region DE={n_reg} | random split DE={rnd_mean:.1f}±{rnd_sd:.1f}")
print("(c)", " ".join(f"{d:+.2f}:{r:+.2f}" for d,r,_ in rows))
json.dump({"region_de":int(n_reg),"random_de":float(rnd_mean),"random_sd":float(rnd_sd),
           "split_half_r_all":float(r_half),"split_half_r_region":float(r_half_reg)},open(f"{OUT}/16_summary.json","w"))

# ---------------- plot ----------------
fig=plt.figure(figsize=(7.09,2.2)); gs=fig.add_gridspec(1,3,wspace=0.55)
def L(ax,l,dx=-0.24,dy=1.0): ax.text(dx,dy,l,transform=ax.transAxes,fontsize=9,va="bottom",ha="left")

ax=fig.add_subplot(gs[0,0])
ax.scatter(lfc1[~sigregion],lfc2[~sigregion],s=2,c="0.82",lw=0,rasterized=True)
ax.scatter(lfc1[sigregion],lfc2[sigregion],s=6,c=C_CB,lw=0,rasterized=True,zorder=3)
ax.axhline(0,c="0.88",lw=0.5); ax.axvline(0,c="0.88",lw=0.5)
ax.text(0.03,0.96,f"all genes: $r$ = {r_half:.2f}",transform=ax.transAxes,fontsize=6,va="top",color="0.5")
ax.text(0.03,0.87,f"region-DE genes: $r$ = {r_half_reg:.2f}",transform=ax.transAxes,fontsize=6,va="top",color=C_CB)
ax.set_xlabel("log$_2$ FC (cerebellum vs. cerebrum)\n[split half A]")
ax.set_ylabel("log$_2$ FC\n[split half B]")
L(ax,"a")

ax=fig.add_subplot(gs[0,1])
ax.bar([0,1],[n_reg,rnd_mean],yerr=[0,rnd_sd],width=0.55,
       color=[C_CB,C_CBR],edgecolor="k",linewidth=0.4,capsize=3)
for x,v in zip([0,1],[n_reg,rnd_mean]): ax.text(x,v+3,f"{v:.0f}",ha="center",va="bottom",fontsize=6.5)
ax.set_xticks([0,1]); ax.set_xticklabels(["region\n(cb vs. cbr)","balanced\nrandom split"],fontsize=6.5)
ax.set_ylabel("DE genes (FDR < 0.05)"); ax.set_ylim(0,n_reg*1.22)
L(ax,"b",dx=-0.28)

ax=fig.add_subplot(gs[0,2])
ax.errorbar(sim["delta"],sim["r"],yerr=sim["sd"],fmt="-o",ms=4,lw=1.2,c=C_CB,capsize=2)
ax.set_xlabel("brain-region imbalance between groups\n(Δ cerebellum fraction)")
ax.set_ylabel("$r$  (group signature vs. true region effect)")
ax.set_ylim(-0.1,1.05); ax.axhline(0,c="0.85",lw=0.5)
L(ax,"c")
export_figure(fig,basename=f"{IMG}/Fig3_region_confound",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,2.2),grayscale_preview=True)
print("Fig3 done")
