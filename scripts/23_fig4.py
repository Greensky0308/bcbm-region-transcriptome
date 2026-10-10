#!/usr/bin/env python3
"""23 — Fig4（2×2）：脑区签名 = 正常脑组织印记（GTEx 正交印证）。
 a BCBM vs GTEx正常脑 脑区效应相关 | b 方向一致性(86% vs 50%) | c 签名集 vs 背景 GTEx log2FC | d 跨脑亚区稳健
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, gzip
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import binomtest, mannwhitneyu
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
C_ACC="#1F6F6B"; C_GREY="0.75"
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

# GTEx
d=pd.read_csv(os.path.join(ROOT, "data", "GTEx", "gtex_median_tpm.gct.gz"),sep="\t",skiprows=2).rename(columns={"Name":"ensg","Description":"sym"})
d=d.dropna(subset=["sym"]); d=d[~d.sym.duplicated()].set_index("sym")
def gtex_lfc(reg):
    e=d[["Brain - Cerebellum",reg]]; e=e[e.sum(1)>1]
    return np.log2(e["Brain - Cerebellum"]+1)-np.log2(e[reg]+1)
gl=gtex_lfc("Brain - Cortex")

de=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0)
bcbm=-de["log2FC"]                     # cerebellum - cerebrum
common=[g for g in de.index if g in gl.index]
sig=[g for g in common if de.loc[g,"q"]<0.05]
SET=[g for g in de.query("q<0.05").query("log2FC<0").index if g in gl.index]   # 小脑↑集
r_all=np.corrcoef(bcbm.loc[common],gl.loc[common])[0,1]
r_sig=np.corrcoef(bcbm.loc[sig],gl.loc[sig])[0,1]
conc=int((gl.loc[SET]>0).sum()); n=len(SET); pconc=binomtest(conc,n,0.5,alternative="greater").pvalue
mw=mannwhitneyu(gl.loc[SET],gl.drop(index=SET),alternative="greater")
print(f"共同基因 {len(common)} | region-DE {len(sig)} | r_all={r_all:.3f} r_sig={r_sig:.3f}")
print(f"小脑↑集 {conc}/{n}={100*conc/n:.0f}% 在GTEx也小脑↑ binomial p={pconc:.2e}")
print(f"GTEx log2FC 签名集 {gl.loc[SET].mean():.3f} vs 其余 {gl.drop(index=SET).mean():.3f} MWU p={mw.pvalue:.2e}")

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,wspace=0.20,hspace=0.38,left=0.075,right=0.985,top=0.95,bottom=0.11)

# (a) 相关
ax=fig.add_subplot(gs[0,0]); ax1=ax
ax.hexbin(bcbm.loc[common],gl.loc[common],gridsize=55,cmap="Greys",bins="log",mincnt=1,linewidths=0)
ax.scatter(bcbm.loc[sig],gl.loc[sig],s=5,c=C_ACC,lw=0,rasterized=True,zorder=3)
ax.axhline(0,c="0.88",lw=0.5); ax.axvline(0,c="0.88",lw=0.5)
ax.plot([-3,3],[-3,3],ls="--",lw=0.6,c="0.6")
ax.set_xlim(-3.2,3.2); ax.set_ylim(-4,6)
ax.text(0.03,0.96,f"region-DE genes: $r$ = {r_sig:.2f}",transform=ax.transAxes,fontsize=6.5,va="top")
ax.set_xlabel("log$_2$ FC in BCBM (cerebellum vs. cerebrum)")
ax.set_ylabel("log$_2$ FC in normal brain\n(GTEx cerebellum vs. cortex)")

# (b) 排序瀑布图：signature 基因在 GTEx 的效应
ax=fig.add_subplot(gs[0,1]); ax2=ax
_v=gl.loc[SET].sort_values(ascending=False).values
_col=[C_ACC if x>0 else C_GREY for x in _v]
ax.bar(np.arange(len(_v)),_v,width=1.0,color=_col,linewidth=0)
ax.axhline(0,c="0.2",lw=0.6)
ax.set_xlim(-1,len(_v)); ax.set_ylim(min(_v)*1.15,max(_v)*1.15)
ax.set_xlabel("cerebellum-up signature genes (n=%d, ranked)"%n)
ax.set_ylabel("log$_2$ FC in GTEx\n(cerebellum vs. cortex)")
ax.text(0.44,0.62,f"{conc}/{n} > 0",transform=ax.transAxes,fontsize=6.5,ha="left",va="center",color=C_ACC)
ax.text(0.80,0.06,f"{n-conc}/{n} < 0",transform=ax.transAxes,fontsize=6.5,ha="right",va="bottom",color="0.45")

# (c) 分布
ax=fig.add_subplot(gs[1,0]); ax3=ax
_vv=[gl.loc[SET].values,gl.drop(index=SET).values]
_vp=ax.violinplot(_vv,positions=[0,1],widths=0.85,showextrema=False)
for _b,_c in zip(_vp["bodies"],[C_ACC,C_GREY]):
    _b.set_facecolor(_c); _b.set_alpha(0.75); _b.set_edgecolor("k"); _b.set_linewidth(0.4)
for _i,_v in enumerate(_vv): ax.plot([_i],[np.median(_v)],marker="o",ms=3,c="k",zorder=4)
ax.margins(y=0.26)
ax.axhline(0,c="0.85",lw=0.5)
ax.set_xticks([0,1]); ax.set_xticklabels(["cerebellum-up\nsignature (n=%d)"%n,"other genes"],fontsize=6.5)
ax.set_ylabel("GTEx log$_2$ FC (cerebellum vs. cortex)")
ax.text(0.97,0.96,f"MWU $p$ = {mw.pvalue:.1e}",transform=ax.transAxes,fontsize=6.5,va="top",ha="right")


# (d) 跨脑亚区稳健
ax=fig.add_subplot(gs[1,1]); ax4=ax
REG=["Brain - Cortex","Brain - Frontal Cortex (BA9)","Brain - Anterior cingulate cortex (BA24)",
     "Brain - Hippocampus","Brain - Amygdala","Brain - Putamen (basal ganglia)"]
short=["Cortex","Frontal ctx","ACC","Hippocampus","Amygdala","Putamen"]
vals=[gtex_lfc(r).reindex(SET).mean() for r in REG]
_y=np.arange(len(REG))
_PAL=["#1f77b4","#2ca02c","#d62728","#8c564b","#e377c2","#17becf"]
for _i,(_v,_c) in enumerate(zip(vals,_PAL)):
    ax.hlines(_i,0,_v,color=_c,lw=1.2)
    ax.scatter([_v],[_i],s=26,c=_c,zorder=3)
ax.set_yticks(_y); ax.set_yticklabels(short,fontsize=6.5); ax.set_ylim(len(REG)-0.5,-0.7)
ax.set_xlabel("mean log$_2$ FC of signature in GTEx\n(cerebellum vs. each region)")
for i,v in enumerate(vals): ax.text(v+0.10,i,f"{v:.1f}",va="center",fontsize=6)

L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/Fig4_normal_brain_anchor",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
# 预览 PNG 不留在 图/（QA 用，挪到 /tmp）
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")

print("Fig4 done")
