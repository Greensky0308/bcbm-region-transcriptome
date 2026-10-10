#!/usr/bin/env python3
"""25 — 补充图 S1（2×2）：数据质控。
 a bulk 文库大小 | b bulk region×batch(证不混杂) | c snRNA 每样本核数 | d snRNA 质控指标
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
import matplotlib as mpl, matplotlib.pyplot as plt
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
C_CB,C_CBR="#A0522D","#0072B2"
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}
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

cnt=pd.read_excel(os.path.join(ROOT, "data", "GSE322943", "counts.xlsx"),index_col=0)
cnt=cnt.rename(columns=lambda c:str(c).replace("Exp_",""))
bm=pd.read_csv(f"{FIG}/bulk_meta.csv"); iscb=(bm["location"].values=="cerebellum")
libsz=cnt[bm["sample"].values].sum(0).values/1e6

meta=pd.read_csv(f"{OUT}/GSE324453/meta.csv",index_col=0)
meta["region"]=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,wspace=0.20,hspace=0.26,left=0.075,right=0.985,top=0.95,bottom=0.11)
ax=fig.add_subplot(gs[0,0]); ax1=ax
for mask,cc,lab in [(~iscb,C_CBR,"cerebrum"),(iscb,C_CB,"cerebellum")]:
    ax.hist(libsz[mask],bins=15,color=cc,alpha=0.9,label=lab,histtype="step",lw=1.3)
ax.set_xlabel("library size (million reads)"); ax.set_ylabel("# samples")
_yl=ax.get_ylim(); ax.set_ylim(_yl[0],_yl[1]*1.30)
ax.legend(fontsize=6,loc="upper right"); L(ax,"a")

ax=fig.add_subplot(gs[0,1]); ax2=ax
_bs=pd.crosstab(bm["batch"],bm["location"])
_frac=_bs.div(_bs.sum(1),axis=0)*100
_y=np.arange(len(_bs)); _left=np.zeros(len(_bs))
for _p,_c in [("cerebrum",C_CBR),("cerebellum",C_CB)]:
    _v=_frac[_p].values
    ax.barh(_y,_v,left=_left,color=_c,height=0.62,linewidth=0)
    for _i in range(len(_bs)):
        ax.text(_left[_i]+_v[_i]/2,_i,f"{_v[_i]:.0f}%",ha="center",va="center",fontsize=5.5,color="w")
    _left+=_v
ax.set_yticks(_y); ax.set_yticklabels([str(b) for b in _bs.index],fontsize=6.3)
ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel("% of samples"); ax.set_ylabel("batch")

ax=fig.add_subplot(gs[1,0]); ax3=ax
nuc=meta.groupby(["sampleDemuxIDstrict","region"]).size().reset_index(name="n")
nuc=nuc.sort_values(["region","n"])
col=[C_CB if r=="cerebellum" else C_CBR for r in nuc.region]
ax.bar(np.arange(len(nuc)),nuc.n.values/1000,color=col,edgecolor="k",linewidth=0.3)
ax.set_xticks([]); ax.set_ylabel("nuclei (thousands)"); ax.set_xlabel("GSE324453 samples (n=28)"); ax.set_ylim(0,8.6)
ax.legend(handles=[plt.Line2D([],[],marker="s",ls="",c=C_CB,label="cerebellum"),
                   plt.Line2D([],[],marker="s",ls="",c=C_CBR,label="cerebrum")],fontsize=6,loc="upper right",handletextpad=0.2)

ax=fig.add_subplot(gs[1,1]); ax4=ax
_x=meta["nCount_RNA"].values; _y=meta["percent.mt"].values; _cb=(meta["region"].values=="cerebellum")
hb=ax.hexbin(_x,_y,xscale="log",gridsize=45,cmap="Blues",bins="log",mincnt=1,linewidths=0)
cb2=fig.colorbar(hb,ax=ax,fraction=0.045,pad=0.02); cb2.set_label("# nuclei",fontsize=6); cb2.ax.tick_params(labelsize=6)
ax.set_xscale("log")
ax.set_xlabel("UMI per nucleus"); ax.set_ylabel("% mitochondrial")
ax.axhline(10,c="0.6",ls="--",lw=0.6)

L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/FigS1_QC",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
# move QA preview PNGs out of the figure directory
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")

print("S1 done")
