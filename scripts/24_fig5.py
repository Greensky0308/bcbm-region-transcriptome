#!/usr/bin/env python3
"""24 — Fig5（2×2）：双混杂反例（GSE325196 多原发 snRNA）。
 a region↔原发 完全混杂 | b "神经元样"簇=纯 SCLC 样本 | c 那些细胞是 SCLC 瘤(神经内分泌) | d 修正前后
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
import matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
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
C_ART="#2E4057"; C_GOOD="#52B788"; C_GREY="0.72"
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

with h5py.File(f"{OUT}/03d_sc_adata.h5ad","r") as h:
    cg=h["layers/counts"]; X=sp.csr_matrix((cg["data"][:],cg["indices"][:],cg["indptr"][:]),shape=tuple(cg.attrs["shape"]))
    feat=np.array([x.decode() if isinstance(x,bytes) else x for x in h["var/_index"][:]])
    def cat(k):
        o=h["obs/"+k]; c=[x.decode() if isinstance(x,bytes) else x for x in o["categories"][:]]
        return np.array(c)[o["codes"][:]]
    leiden=cat("leiden"); gsm=cat("gsm"); prim=cat("primary"); regn=cat("region")
X=X.tocsr(); lib=np.asarray(X.sum(1)).ravel(); lib[lib==0]=1
Xl=X.multiply(1e4/lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data)
g2i={g:i for i,g in enumerate(feat)}

NEUR_CL=[0,11,13,21,22,23,28,29]
NET=["ASCL1","INSM1","CHGA","SYP"]
MK=["ASCL1","INSM1","CHGA","SYP","RELN","BARHL2","GABRA6","CBLN3"]
iscb=(regn=="Cerebellum"); isbrain=(regn=="Brain")

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,wspace=0.36,hspace=0.38,left=0.075,right=0.985,top=0.95,bottom=0.11)

# (a) region × primary 构成（每行一个低饱和色系，行内深浅=原发）
ax=fig.add_subplot(gs[0,0]); ax1=ax
REG_ORD=["Cerebellum","Brain","Frontal lobe","Parietal lobe","Temporal lobe","Occipital lobe"]
ct=pd.crosstab(regn,prim).reindex(REG_ORD)
_SHORT={"ER+/PR-/HER2+ BC":"ER+/HER2+","ER-/PR-/HER2+ BC":"ER-/HER2+","ER-/PR-/HER2- BC":"TNBC",
        "Melanoma":"Mel","NSCLC":"NSCLC","SCLC":"SCLC","Sarcoma":"Sarc"}
frac=ct.div(ct.sum(1),axis=0)*100
BASE=["#4C72B0","#55A868","#8172B2","#C44E52","#CCB974","#64B5CD"]
def _shades(base,n):
    b=np.array(mpl.colors.to_rgb(base))
    return [tuple(1-(1-b)*f) for f in np.linspace(0.30,1.0,n)]
_y=np.arange(len(REG_ORD)); 
for _i,(reg,base) in enumerate(zip(REG_ORD,BASE)):
    _left=0.0; _sh=_shades(base,len(ct.columns))
    for _j,_p in enumerate(ct.columns):
        _v=frac.loc[reg,_p]
        ax.barh(_i,_v,left=_left,color=_sh[_j],height=0.72,linewidth=0)
        _left+=_v
    _dom=frac.loc[reg].idxmax()
    ax.text(3,_i,_SHORT.get(_dom,_dom),va="center",ha="left",fontsize=5.5,color="k")
ax.set_yticks(_y); ax.set_yticklabels(REG_ORD,fontsize=6.3); ax.invert_yaxis()
ax.set_xlim(0,100); ax.set_xlabel("% of cells")

# (b) 点阵图：簇 × 原发（点大小 = 占比）
ax=fig.add_subplot(gs[0,1]); ax2=ax
_leid=leiden.astype(int)
_ncl=sorted([cl for cl in NEUR_CL if (_leid==cl).sum()>0])
_PORD=[p for p in ["SCLC","NSCLC","ER-/PR-/HER2- BC","Melanoma","ER+/PR-/HER2+ BC","ER-/PR-/HER2+ BC","Sarcoma"] if p in set(prim)]
_frac=pd.DataFrame({p:[(prim[_leid==cl]==p).mean()*100 for cl in _ncl] for p in _PORD},index=_ncl)
_PBASE=["#4C72B0","#55A868","#8172B2","#C44E52","#CCB974","#64B5CD","#8C8C8C"][:len(_PORD)]
for _i,_cl in enumerate(_ncl):
    for _j,_p in enumerate(_PORD):
        _v=_frac.loc[_cl,_p]
        if _v<=0: continue
        ax.scatter(_j,_i,s=_v*1.5+3,c=_PBASE[_j],alpha=0.9,lw=0,zorder=3)
ax.set_xticks(range(len(_PORD))); ax.set_xticklabels([_SHORT.get(p,p) for p in _PORD],rotation=40,ha="right",fontsize=6)
ax.set_yticks(range(len(_ncl))); ax.set_yticklabels([str(c) for c in _ncl],fontsize=6)
ax.set_xlim(-0.6,len(_PORD)-0.4); ax.set_ylim(len(_ncl)-0.4,-0.6)
ax.set_ylabel("neuron-like cluster")

# (c) marker 点图：神经内分泌程序 vs 神经元程序（在这些"神经元样"簇内）
ax=fig.add_subplot(gs[1,0]); ax3=ax
MK_C=["ASCL1","INSM1","CHGA","SYP","RBFOX3","SYT1","MAP2","SNAP25"]
MK_C=[g for g in MK_C if g in g2i]
_ncl=sorted([cl for cl in NEUR_CL if (_leid==cl).sum()>0])
_ss=[]; _cs=[]; _xs=[]; _ys=[]
_vals=[]
for _i,g in enumerate(MK_C):
    _j=g2i[g]
    for _k,cl in enumerate(_ncl):
        m=(_leid==cl); col=np.asarray(Xl[m][:,_j].todense()).ravel()
        _vals.append(col.mean())
        _xs.append(_k); _ys.append(_i); _ss.append((col>0).mean()*100); _cs.append(col.mean())
_vmax=np.percentile(_vals,98)
_sc=ax.scatter(_xs,_ys,s=np.array(_ss)*0.9+3,c=_cs,cmap="Blues",vmin=0,vmax=_vmax,edgecolor="k",linewidth=0.15)
ax.axhline(3.5,c="0.6",lw=0.6)
ax.set_xticks(range(len(_ncl))); ax.set_xticklabels([str(c) for c in _ncl],fontsize=6)
ax.set_yticks(range(len(MK_C))); ax.set_yticklabels(MK_C,fontsize=6)
ax.set_xlim(-0.6,len(_ncl)-0.4); ax.set_ylim(len(MK_C)-0.4,-0.6)
ax.set_xlabel("neuron-like cluster")
cb=fig.colorbar(_sc,ax=ax,fraction=0.030,pad=0.012); cb.set_label("mean\nexpr.",fontsize=5,labelpad=1); cb.ax.tick_params(labelsize=5)

# (d) 修正前后（哑铃图）
ax=fig.add_subplot(gs[1,1]); ax4=ax
mk=[g for g in MK if g in g2i]; mki=[g2i[g] for g in mk]
def pv(mask):
    sub=Xl[mask][:,mki]
    v=np.asarray(sub.todense()) if sp.issparse(sub) else np.asarray(sub)
    cb=(mask&iscb)[mask]; lb=(mask&~iscb&~isbrain)[mask]
    return [mannwhitneyu(v[:,j][cb],v[:,j][lb])[1] if cb.sum()>1 and lb.sum()>1 else np.nan for j in range(len(mk))]
mNeu=np.isin(leiden.astype(int),NEUR_CL)
p_all=pv(mNeu); p_noSCLC=pv(mNeu & (prim!="SCLC"))
_yv=np.arange(len(mk))
_l1=[-np.log10(max(p,1e-16)) if not np.isnan(p) else 0 for p in p_all]
_l2=[-np.log10(max(p,1e-16)) if not np.isnan(p) else 0 for p in p_noSCLC]
for _i in _yv: ax.plot([_l1[_i],_l2[_i]],[_i,_i],c="0.85",lw=1.2,zorder=1)
ax.scatter(_l1,_yv,s=26,c=C_ART,zorder=3,label="all samples")
ax.scatter(_l2,_yv,s=26,c=C_GOOD,zorder=3,label="SCLC excluded")
ax.axvline(-np.log10(0.05),c="0.5",ls="--",lw=0.7)
ax.set_yticks(_yv); ax.set_yticklabels(mk,fontsize=6.3); ax.set_ylim(len(mk)+0.2,-1.6)
ax.set_xlim(-0.8,27)
ax.set_xlabel("$-\\log_{10}$ $p$ (cerebellum vs. lobe)")
ax.legend(fontsize=6,loc="upper right",handletextpad=0.2)

L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/Fig5_dual_confound",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
# 预览 PNG 不留在 图/（QA 用，挪到 /tmp）
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")

print("Fig5 done")
