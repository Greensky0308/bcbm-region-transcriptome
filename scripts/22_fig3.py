#!/usr/bin/env python3
"""22 — Fig3（2×2）：脑区混杂的量化与危害。
 a split-half 复现性 | b 零分布：随机分组 vs 按脑区分组 | c r 随不平衡上升 | d DE 数随不平衡（不控 vs 控脑区）
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
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
mpl.rcParams.update({"font.family":"Arial","font.size":7,
    "axes.labelsize":7,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,
    "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
    "legend.frameon":False,"axes.spines.top":False,"axes.spines.right":False,
    "figure.constrained_layout.use":False,
    "pdf.fonttype":42,"ps.fonttype":42})
C_CB,C_CBR="#D55E00","#0072B2"; C_GRN="#009E73"
cpm=pd.read_csv(f"{FIG}/bulk_cpm.csv",index_col=0)
cpm.index=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0).index.values
M=cpm.values.astype(float); genes=np.array(cpm.index)
bm=pd.read_csv(f"{FIG}/bulk_meta.csv"); iscb=(bm["location"].values=="cerebellum")
cb=np.where(iscb)[0]; cbr=np.where(~iscb)[0]; N=M.shape[1]
z=(iscb.astype(float)-iscb.mean()); R=M-np.outer((M@z)/(z@z), z)   # 逐基因残差化掉 region
reg_lfc=M[:,cb].mean(1)-M[:,cbr].mean(1)

def _lab_x(a):
    fig.canvas.draw(); inv=a.transAxes.inverted(); lb=a.yaxis.label
    if lb.get_text():
        bb=lb.get_window_extent(); return inv.transform(((bb.x0+bb.x1)/2,0))[0]
    return None
def L(a,letter,ref=None,dy=1.03):
    x=_lab_x(a)
    if x is None: x=_lab_x(ref) if ref is not None else None
    if x is None: x=-0.12
    a.text(x,dy,letter,transform=a.transAxes,fontsize=9,va="bottom",ha="center")

def nde(X,mA):
    t,p=ttest_ind(X[:,mA],X[:,~mA],axis=1,equal_var=False)
    return int((multipletests(p,method="fdr_bh")[1]<0.05).sum())

# (a) split-half
rng=np.random.default_rng(2); cbs=rng.permutation(cb); cbrs=rng.permutation(cbr)
lfc1=M[:,cbs[:len(cbs)//2]].mean(1)-M[:,cbrs[:len(cbrs)//2]].mean(1)
lfc2=M[:,cbs[len(cbs)//2:]].mean(1)-M[:,cbrs[len(cbrs)//2:]].mean(1)
sig=multipletests(ttest_ind(M[:,iscb],M[:,~iscb],axis=1,equal_var=False)[1],method="fdr_bh")[1]<0.05
r_all=np.corrcoef(lfc1,lfc2)[0,1]; r_reg=np.corrcoef(lfc1[sig],lfc2[sig])[0,1]
n_reg=nde(M,iscb)

# (b) 零分布
rr=np.random.default_rng(1)
rnd=[nde(M,np.isin(np.arange(N),rr.choice(N,N//2,replace=False))) for _ in range(200)]

# (c) r 随不平衡
rows=[]
for k in range(17,35,2):
    g=np.random.default_rng(0); rs=[]
    for _ in range(20):
        A=np.concatenate([g.choice(cb,k,replace=False),g.choice(cbr,N//2-k,replace=False)])
        mA=np.isin(np.arange(N),A); lfc=M[:,mA].mean(1)-M[:,~mA].mean(1)
        rs.append(np.corrcoef(lfc,reg_lfc)[0,1])
    rows.append((k/N-(len(cb)-k)/(N-N//2), np.mean(rs), np.std(rs)))
sim=pd.DataFrame(rows,columns=["delta","r","sd"])

# (d) DE 数 随不平衡（固定组大小 n=40）
nA=40; dd,na,nas,ad,ads=[],[],[],[],[]
for k in [14,18,22,26,30,34]:
    g=np.random.default_rng(0); a=[];b=[]
    for _ in range(30):
        A=np.concatenate([g.choice(cb,k,replace=False),g.choice(cbr,nA-k,replace=False)])
        mA=np.isin(np.arange(N),A); a.append(nde(M,mA)); b.append(nde(R,mA))
    dd.append(k/nA-(len(cb)-k)/(N-nA)); na.append(np.mean(a)); nas.append(np.std(a))
    ad.append(np.mean(b)); ads.append(np.std(b))
    print(f"  k={k} Δ={dd[-1]:+.2f} 朴素{na[-1]:.1f}±{nas[-1]:.1f} 校正{ad[-1]:.1f}±{ads[-1]:.1f}")

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,wspace=0.20,hspace=0.26,left=0.075,right=0.985,top=0.95,bottom=0.11)

# (a)
ax=fig.add_subplot(gs[0,0]); ax1=ax
ax.scatter(lfc1[~sig],lfc2[~sig],s=2,c="0.82",lw=0,rasterized=True)
ax.scatter(lfc1[sig],lfc2[sig],s=6,c=C_CB,lw=0,rasterized=True,zorder=3)
_lo=min(lfc1.min(),lfc2.min()); _hi=max(lfc1.max(),lfc2.max())
ax.plot([_lo,_hi],[_lo,_hi],ls="--",lw=0.6,c="0.6",zorder=1)
ax.axhline(0,c="0.88",lw=0.5); ax.axvline(0,c="0.88",lw=0.5); ax.margins(x=0.10,y=0.32)
ax.text(0.03,0.97,f"all genes: $r$ = {r_all:.2f}",transform=ax.transAxes,fontsize=6,va="top",ha="left",color="0.5")
ax.text(0.03,0.89,f"region-DE genes: $r$ = {r_reg:.2f}",transform=ax.transAxes,fontsize=6,va="top",ha="left",color=C_CB)
ax.set_xlabel("log$_2$ FC [split half A]"); ax.set_ylabel("log$_2$ FC [split half B]")
ax.legend(handles=[plt.Line2D([],[],marker="o",ls="",c=C_CB,label="region-DE genes"),
                   plt.Line2D([],[],marker="o",ls="",c="0.72",label="other genes")],
          fontsize=6,loc="lower right",handletextpad=0.2)

# (b) 零分布：按脑区分组 vs 200 次平衡随机分组
ax=fig.add_subplot(gs[0,1]); ax2=ax
_rndm=float(np.mean(rnd)); _rnds=float(np.std(rnd)); _rndmax=int(np.max(rnd))
ax.bar([0],[n_reg],width=0.6,color=C_CB,linewidth=0)
ax.bar([1],[_rndm],width=0.6,color="0.72",linewidth=0)
ax.errorbar([1],[_rndm],yerr=[_rnds],fmt="none",ecolor="0.35",capsize=3,lw=0.8)
ax.plot([1],[_rndmax],marker="_",ms=9,mew=1.2,c="0.25")
ax.text(0,n_reg+1.2,f"{n_reg}",ha="center",va="bottom",fontsize=6.5)
ax.text(1,_rndm+1.2,f"{_rndm:.1f}",ha="center",va="bottom",fontsize=6.5)
ax.text(1.07,_rndmax,"max single\ndraw %d"%_rndmax,va="center",ha="left",fontsize=6,color="0.35")
ax.set_xticks([0,1]); ax.set_xticklabels(["by region\n(cb vs. cbr)","balanced random\nsplit (x200)"],fontsize=6.5)
ax.set_ylabel("genes at FDR < 0.05"); ax.set_xlim(-0.6,1.9); ax.set_ylim(0,n_reg*1.30)
# (c)
ax=fig.add_subplot(gs[1,0]); ax3=ax
ax.errorbar(sim["delta"],sim["r"],yerr=sim["sd"],fmt="-o",ms=4,lw=1.2,c=C_CBR,capsize=2)
ax.set_xlabel("brain-region imbalance between groups\n(Δ cerebellum fraction)")
ax.set_ylabel("$r$ (group signature vs. region effect)")
ax.set_ylim(-0.1,1.05); ax.axhline(0,c="0.85",lw=0.5)

# (d) 模拟对比中 DE 基因数：不适定 vs 校正脑区（30 次抽样）
ax=fig.add_subplot(gs[1,1]); ax4=ax
_dd=np.array(dd); _x=np.arange(len(_dd)); _w=0.36
ax.bar(_x-_w/2,na,width=_w,color="#8B0000",linewidth=0,label="ignoring region")
ax.bar(_x+_w/2,ad,width=_w,color="#009E73",linewidth=0,label="adjusted for region")
ax.errorbar(_x-_w/2,na,yerr=nas,fmt="none",ecolor="0.35",capsize=2,lw=0.7)
ax.errorbar(_x+_w/2,ad,yerr=ads,fmt="none",ecolor="0.35",capsize=2,lw=0.7)
ax.set_xticks(_x); ax.set_xticklabels(["%.2f"%v for v in _dd],fontsize=6.5)
ax.set_xlabel("brain-region imbalance (Δ)"); ax.set_ylabel("genes at FDR < 0.05")
ax.legend(fontsize=6,loc="upper left",handletextpad=0.3)
L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/Fig3_region_confound",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")
print("Fig3 done")
