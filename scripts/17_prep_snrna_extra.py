#!/usr/bin/env python3
"""17 — 缓存 snRNA 额外绘图数据：各区室 marker 均值 + 逐样本细胞组成。"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
B=ROOT
OUT=os.path.join(ROOT, "out"); IN=f"{OUT}/GSE324453"; FIG=f"{OUT}/figdata"
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}
meta=pd.read_csv(f"{IN}/meta.csv",index_col=0); feat=[l.strip() for l in open(f"{IN}/features.tsv")]
X=sp.csr_matrix(mmread(f"{IN}/counts.mtx")).T.tocsr()
lib=np.asarray(X.sum(1)).ravel(); lib[lib==0]=1
Xl=X.multiply(1e4/lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data); meta=meta.reset_index(drop=True)
meta["region"]=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")
MK=["EPCAM","KRT8","KRT18","TRPS1","ERBB2","MLANA","MITF","RBFOX3","SYT1","MBP","PLP1","GFAP","AQP4",
    "PTPRC","CD68","CD163","CD3D","MS4A1","PECAM1","VWF","COL1A1","DCN","MKI67","TOP2A","ASCL1","INSM1","CHGA","SYP"]
g2i={g:i for i,g in enumerate(feat)}
CTS=["Tumor cells","Tumor cells (prol.)","Neurons","Astrocytes","ODG","Myeloid cells","T/NK cells",
     "B cells","Plasma cells","Endothelial cells","Pericytes","Stromal cells"]
rows=[]
for ct in CTS:
    m=(meta["common_cell_clusters"]==ct).values
    if m.sum()==0: continue
    rows.append([ct,m.sum()]+[float(np.asarray(Xl[m,g2i[g]].mean())) if g in g2i else np.nan for g in MK])
pd.DataFrame(rows,columns=["celltype","n"]+MK).to_csv(f"{FIG}/snrna_marker_by_ct.csv",index=False)
comp=pd.crosstab(meta["sampleDemuxIDstrict"],meta["common_cell_clusters"])
comp["region"]=meta.groupby("sampleDemuxIDstrict")["region"].first()
comp.to_csv(f"{FIG}/snrna_composition.csv")
print("落盘 snrna_marker_by_ct.csv / snrna_composition.csv |", comp.shape)
print(pd.DataFrame(rows,columns=["celltype","n"]+MK)[["celltype","n","EPCAM","RBFOX3","MBP","PTPRC","COL1A1"]].to_string(index=False))

# ---- dot plot 数据：每细胞类型 × marker 的 表达比例 + 平均表达 ----
DOT=["EPCAM","KRT8","TRPS1","ERBB2","RBFOX3","SYT1","SNAP25","GFAP","AQP4","MBP","PLP1",
     "CD68","CD163","CD3D","NKG7","MS4A1","MZB1","PECAM1","RGS5","COL1A1","DCN","MKI67"]
rows=[]
for ct in CTS:
    m=(meta["common_cell_clusters"]==ct).values
    if m.sum()==0: continue
    sub=X[m]; nn=sub.shape[0]
    for g in DOT:
        if g not in g2i: continue
        j=g2i[g]
        col=np.asarray(sub[:,j].todense()).ravel()
        rows.append((ct,g,float((col>0).mean()*100),float(np.log1p(col).mean())))
pd.DataFrame(rows,columns=["celltype","gene","pct","mean"]).to_csv(f"{FIG}/dotplot_markers.csv",index=False)
print("落盘 dotplot_markers.csv |", len(rows), "行")
