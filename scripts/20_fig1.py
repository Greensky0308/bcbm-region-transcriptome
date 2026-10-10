#!/usr/bin/env python3
"""20 — Fig1（2×2）：脑区在 BCBM bulk 转录组里的系统性差异（GSE322943, n=99）。
 a volcano | b top region-DE 热图 | c 脑区标志逐基因箱线 | d 纯度校正稳健性
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd
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
C_CB,C_CBR="#D55E00","#0072B2"

cpm=pd.read_csv(f"{FIG}/bulk_cpm.csv",index_col=0)
cpm.index=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0).index.values
bm=pd.read_csv(f"{FIG}/bulk_meta.csv"); iscb=(bm["location"].values=="cerebellum")
de=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0)
adj=pd.read_csv(f"{OUT}/03c_region_adjusted.csv",index_col=0)

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

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,wspace=0.20,hspace=0.38,left=0.075,right=0.985,top=0.95,bottom=0.11)

# (a) volcano
ax=fig.add_subplot(gs[0,0]); ax1=ax
y=-np.log10(de["p"].clip(lower=1e-300)); sig=de["q"]<0.05
ax.scatter(de["log2FC"][~sig],y[~sig],s=2,c="0.80",lw=0,rasterized=True)
ax.scatter(de["log2FC"][sig&(de["log2FC"]>0)],y[sig&(de["log2FC"]>0)],s=3,c=C_CB,lw=0,rasterized=True)
ax.scatter(de["log2FC"][sig&(de["log2FC"]<0)],y[sig&(de["log2FC"]<0)],s=3,c=C_CBR,lw=0,rasterized=True)
for g,(dx,dy,ha) in {"GABRA6":(5,3,"left"),"BARHL2":(4,-4,"left"),"NEUROD1":(4,4,"left"),
                     "RELN":(-4,-1,"right"),"FOXG1":(4,-1,"left")}.items():
    if g in de.index:
        xi=de.loc[g,"log2FC"]; yi=-np.log10(max(de.loc[g,"p"],1e-300))
        ax.annotate(g,(xi,yi),textcoords="offset points",xytext=(dx,dy),fontsize=6.5,ha=ha)
ax.set_xlabel("log$_2$ FC  (cerebellum vs. cerebrum)"); ax.set_ylabel("$-$log$_{10}$ $p$")
ax.set_xlim(-3.4,2.6); ax.set_ylim(-1,22)
ax.text(0.03,0.97,f"{(de['q']<0.05).sum()} DE genes",transform=ax.transAxes,fontsize=6.5,va="top")

# (b) heatmap
ax=fig.add_subplot(gs[0,1]); ax2=ax
cand=de[de["q"]<0.05]; cand=cand[[not g.startswith("ENSG") for g in cand.index]]
top=cand.reindex(cand["log2FC"].abs().sort_values(ascending=False).index).head(20).index
order=np.argsort(iscb)
sub=cpm.loc[top].iloc[:,order]
z=sub.sub(sub.mean(1),axis=0).div(sub.std(1)+1e-9,axis=0).clip(-2.5,2.5)
im=ax.imshow(z.values,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top,fontsize=6.5)
ax.set_xticks([]); ax.set_xlabel("BCBM samples",labelpad=10)
axb=ax.inset_axes([0,-0.055,1,0.062])
axb.imshow(np.where(iscb[order],0,1)[None,:],aspect="auto",cmap=mpl.colors.ListedColormap([C_CB,C_CBR]))
axb.set_xticks([]); axb.set_yticks([])
for s in axb.spines.values(): s.set_visible(False)
_mask=iscb[order]; _N=len(_mask); _k=int((~_mask).sum())   # 左段=大脑
axb.text(_k/2/_N, 0.5, "cerebrum", color="w", fontsize=5, ha="center", va="center", transform=axb.transAxes)
axb.text((_k+_N)/2/_N, 0.5, "cerebellum", color="w", fontsize=5, ha="center", va="center", transform=axb.transAxes)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03); cb.set_label("row $z$",fontsize=6.5); cb.ax.tick_params(labelsize=6.5)

# (c) 脑区标志逐基因箱线
ax=fig.add_subplot(gs[1,0]); ax3=ax
MK=["RELN","GABRA6","BARHL2","CALB2","CBLN3","FOXG1"]
MK=[g for g in MK if g in cpm.index]
pos=np.arange(len(MK))
for i,g in enumerate(MK):
    v=cpm.loc[g].values
    for k,(mask,cc) in enumerate([(iscb,C_CB),(~iscb,C_CBR)]):
        bp=ax.boxplot([v[mask]],positions=[i+(0.19 if k else -0.19)],widths=0.30,
                      patch_artist=True,showfliers=False,medianprops=dict(color="k",lw=0.8),
                      boxprops=dict(facecolor=cc,alpha=0.55,linewidth=0.4),
                      whiskerprops=dict(lw=0.5),capprops=dict(lw=0.5))
ax.set_xticks(pos); ax.set_xticklabels(MK,fontsize=6.5)
ax.set_ylabel("log$_2$ CPM")
ax.legend(handles=[plt.Line2D([],[],marker="s",ls="",mfc=C_CB,mec=C_CB,label="cerebellum",alpha=0.7),
                   plt.Line2D([],[],marker="s",ls="",mfc=C_CBR,mec=C_CBR,label="cerebrum",alpha=0.7)],
          fontsize=6.5,loc="upper right",handletextpad=0.2)

# (d) 纯度校正
ax=fig.add_subplot(gs[1,1]); ax4=ax
m=adj.reindex(de.index).dropna(subset=["log2FC_adj"])
ax.scatter(de.loc[m.index,"log2FC"],m["log2FC_adj"],s=2,c="0.65",lw=0,rasterized=True)
for g,(dx,dy,ha) in {"OLIG2":(5,4,"left"),"FOXG1":(5,-8,"left"),
                     "RELN":(3,6,"left"),"GABRA6":(-3,-9,"right"),"BARHL2":(5,-2,"left")}.items():
    if g in m.index:
        ax.scatter([de.loc[g,"log2FC"]],[m.loc[g,"log2FC_adj"]],s=16,
                   c=(C_CB if m.loc[g,"log2FC_adj"]>0 else C_CBR),zorder=3,edgecolor="k",linewidth=0.3)
        ax.annotate(g,(de.loc[g,"log2FC"],m.loc[g,"log2FC_adj"]),textcoords="offset points",
                    xytext=(dx,dy),fontsize=6.5,ha=ha)
ax.axhline(0,c="0.7",lw=0.5); ax.axvline(0,c="0.7",lw=0.5)
ax.plot([-3,3],[-3,3],ls="--",lw=0.6,c="0.6",zorder=1)   # y=x 参考
r=np.corrcoef(de.loc[m.index,"log2FC"],m["log2FC_adj"])[0,1]
ax.set_xlabel("log$_2$ FC (raw)\n(cerebellum vs. cerebrum)"); ax.set_ylabel("log$_2$ FC (adjusted)\n(cerebellum vs. cerebrum)")
ax.text(0.03,0.95,f"$r$ = {r:.2f}",transform=ax.transAxes,fontsize=6.5,va="top")
ax.set_xlim(-3.0,2.5)

L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/Fig1_bulk_region",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
# move QA preview PNGs out of the figure directory
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")

print("Fig1 done")
