#!/usr/bin/env python3
"""21 — Fig2（2×2）：信号来自宿主脑组织（GSE324453, n=28, 同队列）。
 a 逐样本细胞组成 | b marker 验证 | c 各区室×脑区 签名分 | d 留一法
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np, pandas as pd
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
C_CB,C_CBR="#D55E00","#0072B2"
pb=pd.read_csv(f"{FIG}/snrna_pb.csv")
mk=pd.read_csv(f"{FIG}/snrna_marker_by_ct.csv")
comp=pd.read_csv(f"{FIG}/snrna_composition.csv",index_col=0)
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

fig=plt.figure(figsize=(7.09,5.0)); gs=fig.add_gridspec(2,2,width_ratios=[1,1.28],wspace=0.20,hspace=0.42,left=0.075,right=0.985,top=0.95,bottom=0.11)

# (a) 组成
ax=fig.add_subplot(gs[0,0]); ax1=ax
cts=[c for c in comp.columns if c!="region"]
cts=[c for c in cts if comp[c].sum()>0]
frac=comp[cts].div(comp[cts].sum(1),axis=0)
order=np.argsort((comp["region"]=="cerebellum").values)  # cerebellum 后
frac=frac.iloc[order]; rreg=comp["region"].values[order]
C_TUM="#1F6F6B"
TUMC=["Tumor cells","Tumor cells (prol.)"]
pal=mpl.colors.ListedColormap(plt.get_cmap("tab20").colors[:len(cts)])
bottom=np.zeros(len(frac))
for j,c in enumerate(cts):
    col=(C_TUM if c in TUMC else None)
    ax.bar(np.arange(len(frac)),frac[c].values,bottom=bottom,width=1.0,
           color=col if col else pal(j),linewidth=0)
    bottom+=frac[c].values
ax.set_xlim(-0.5,len(frac)-0.5); ax.set_ylim(0,1.32)
ax.set_xticks([]); ax.set_ylabel("cell fraction")
ax.set_xlabel("BCBM samples (ordered by region)",labelpad=17)
axb=ax.inset_axes([0,-0.062,1,0.042])
axb.imshow(np.where(rreg=="cerebellum",0,1)[None,:],aspect="auto",cmap=mpl.colors.ListedColormap([C_CB,C_CBR]))
axb.set_xticks([]); axb.set_yticks([])
for s in axb.spines.values(): s.set_visible(False)
_mr=(rreg=="cerebellum"); _N=len(_mr); _k=int((~_mr).sum())
axb.text(_k/2/_N,0.5,"cerebrum",color="w",fontsize=5,ha="center",va="center",transform=axb.transAxes)
axb.text((_k+_N)/2/_N,0.5,"cerebellum",color="w",fontsize=5,ha="center",va="center",transform=axb.transAxes)
ax.legend(handles=[plt.Line2D([],[],marker="s",ls="",c=C_TUM,label="tumor")],
          fontsize=6.5,loc="upper right",handletextpad=0.2)

# (b) marker 验证 —— dot plot
ax=fig.add_subplot(gs[0,1]); ax2=ax
dot=pd.read_csv(f"{FIG}/dotplot_markers.csv")
GORD=["EPCAM","KRT8","TRPS1","ERBB2","RBFOX3","SYT1","SNAP25","GFAP","AQP4","MBP","PLP1",
      "CD68","CD163","CD3D","NKG7","MS4A1","MZB1","PECAM1","RGS5","COL1A1","DCN","MKI67"]
CTORD=["Tumor cells","Tumor cells (prol.)","Neurons","Astrocytes","ODG","Myeloid cells",
       "T/NK cells","B cells","Plasma cells","Endothelial cells","Pericytes","Stromal cells"]
CTSHORT=["Tumor","Tumor(prol.)","Neuron","Astro","ODG","Myeloid","T/NK","B","Plasma","Endo","Pericyte","Stromal"]
GORD=[g for g in GORD if g in set(dot.gene)]
P=dot.pivot(index="gene",columns="celltype",values="pct").reindex(index=GORD,columns=CTORD)
M=dot.pivot(index="gene",columns="celltype",values="mean").reindex(index=GORD,columns=CTORD)
xs,ys,ss,cs=[],[],[],[]
for i,g in enumerate(GORD):
    for j,c in enumerate(CTORD):
        v=P.loc[g,c]; m=M.loc[g,c]
        if pd.isna(v): continue
        xs.append(j); ys.append(i); ss.append(v); cs.append(m)
vmax=np.nanpercentile(M.values,97)
sc=ax.scatter(xs,ys,s=np.array(ss)*0.22+1,c=cs,cmap="viridis",vmin=0,vmax=vmax,
              edgecolor="k",linewidth=0.12)
ax.set_xticks(range(len(CTORD))); ax.set_xticklabels(CTSHORT,rotation=40,ha="right",fontsize=6.5)
ax.set_yticks(range(len(GORD))); ax.set_yticklabels(GORD,fontsize=6)
ax.set_ylim(len(GORD)-0.5,-0.5); ax.set_xlim(-0.6,len(CTORD)-0.4)
cb=fig.colorbar(sc,ax=ax,fraction=0.045,pad=0.02); cb.set_label("mean expr.",fontsize=6); cb.ax.tick_params(labelsize=6)

# (c) 各区室 × 脑区
ax=fig.add_subplot(gs[1,0]); ax3=ax
CMPS=[("Tumor","Tumor"),("Host(all)","Host"),("HostAstro","Astro"),("HostODG","ODG"),("Immune","Immune")]
rng=np.random.default_rng(0)
for i,(cp,_) in enumerate(CMPS):
    d=pb[pb.comp==cp]
    for reg,cc,off in [("cerebrum",C_CBR,-0.18),("cerebellum",C_CB,0.18)]:
        v=d[d.region==reg]["sig_CBL"].values
        ax.scatter(np.full(len(v),i+off)+rng.uniform(-0.045,0.045,len(v)),v,s=7,c=cc,alpha=0.85,lw=0,zorder=3)
        if len(v): ax.plot([i+off-0.10,i+off+0.10],[v.mean()]*2,c=cc,lw=1.3,zorder=4)
    cbv=d[d.region=="cerebellum"]["sig_CBL"].values; lv=d[d.region=="cerebrum"]["sig_CBL"].values
    if len(cbv)>=2 and len(lv)>=2:
        p=mannwhitneyu(cbv,lv)[1]; s="$p$<0.001" if p<0.001 else f"$p$={p:.2f}"
        ax.text(i,0.70,s,ha="center",va="center",fontsize=5.5)
ax.set_xticks(range(len(CMPS))); ax.set_xticklabels([t for _,t in CMPS],fontsize=6.5,rotation=35,ha="right")
ax.set_ylabel("Cerebellum-up signature score\n(bulk 106-gene set)"); ax.set_ylim(-0.05,0.95); ax.set_xlim(-0.55,len(CMPS)-0.4)
ax.legend(handles=[plt.Line2D([],[],marker="o",ls="",c=C_CBR,label="cerebrum"),
                   plt.Line2D([],[],marker="o",ls="",c=C_CB,label="cerebellum")],
          fontsize=6,loc="upper right",handletextpad=0.2)

# (d) 留一法
ax=fig.add_subplot(gs[1,1]); ax4=ax
loo=pd.read_csv(f"{OUT}/13_GSE324453_LODO.csv")
for nm,col,lab in [("TUMOR","#6A3D9A","Tumor"),("HOST","#009E73","Host")]:
    d=loo[loo.compartment==nm].reset_index(drop=True)
    ax.plot(np.arange(len(d)),d["p"],"-o",ms=3.5,lw=1,c=col,label=lab)
ax.axhline(0.05,c="0.75",ls="--",lw=0.7)
ax.text(4.05,0.06,"$p$ = 0.05",fontsize=6.5,c="0.35",ha="right")
ax.set_xticks(np.arange(5)); ax.set_xticklabels(["all","$-$1098","$-$1367","$-$5119","$-$6239"],fontsize=6.5)
ax.set_ylabel("$p$ (cerebellum vs. cerebrum)"); ax.set_ylim(-0.04,1.05)
ax.legend(fontsize=6.5,loc="upper right")

L(ax1,"a",ax3); L(ax2,"b",ax4); L(ax3,"c",ax1); L(ax4,"d",ax2)

export_figure(fig,basename=f"{IMG}/Fig2_snRNA_host",formats=["pdf","tif"],dpi=600,
              size_inches=(7.09,5.0),grayscale_preview=True)
# 预览 PNG 不留在 图/（QA 用，挪到 /tmp）
import glob, shutil
os.makedirs(os.path.join(OUT,"fig_qa"), exist_ok=True)
for _p in glob.glob(f"{IMG}/*.png"):
    shutil.move(_p, f"{os.path.join(OUT,'fig_qa')}/{os.path.basename(_p)}")

print("Fig2 done")
