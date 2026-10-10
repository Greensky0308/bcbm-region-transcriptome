#!/usr/bin/env python3
"""③h 重注释 v2：原发特异肿瘤程序 vs 宿主程序。

理由：03f 的通用 panel argmax 会把「像神经元的 SCLC」「像成纤维的 TNBC」判错；
且簇高度原发特异（簇4/1/34/35/37/31=纯黑色素瘤，簇9=纯TNBC，簇23=纯SCLC）。
做法：每簇按 (a)原发组成 (b)各原发肿瘤程序分 (c)宿主程序分 → 判定 tumor/host。
瘦身加载（同 03g）。
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
import scanpy as sc, anndata as ad

B=ROOT
OUT=os.path.join(ROOT, "out")

TUMOR={
 "BC":   ["EPCAM","KRT8","KRT18","KRT19","KRT7","KRT17","ELF3","CDH1","MUC1","GATA3","TRPS1","FOXA1","ERBB2","XBP1","KRT15"],
 "Mel":  ["MLANA","PMEL","MITF","TYR","DCT","TYRP1","S100B","PAX3","SOX10"],
 "SCLC": ["ASCL1","INSM1","CHGA","SYP","NCAM1","DLL3","NKX2-1","GRP","BCL11A"],
 "NSCLC":["NKX2-1","NAPSA","SFTPB","SFTPC","SCGB3A2","KRT7","MUC1"],
}
HOST={
 "neuron":["RBFOX3","SYT1","SNAP25","NEFL","NEFM","MAP2","SLC17A7","GAD1","GAD2","KCNIP4","DPP6","STMN2"],
 "oligo": ["MBP","PLP1","MOBP","MAG","MOG","CNP","CLDN11","OPALIN"],
 "astro": ["GFAP","AQP4","SLC1A2","SLC1A3","ALDH1L1","S100B"],
 "myeloid":["PTPRC","CD14","CD68","AIF1","CX3CR1","CD163","MRC1","CSF1R","C1QA","C1QB"],
 "tcell": ["CD3D","CD3E","CD8A","CD4","NKG7","KLRD1","IL7R","CCL5"],
 "bcell": ["MS4A1","CD79A","MZB1","JCHAIN","IGKC"],
 "endo":  ["PECAM1","VWF","CLDN5","FLT1","CD34"],
 "fibro": ["COL1A1","COL1A2","DCN","LUM","PDGFRA","COL3A1"],
 "cycling":["MKI67","TOP2A","CENPF","NUF2","BIRC5"],
}

def cat(h,k):
    o=h["obs/"+k]; c=[x.decode() if isinstance(x,bytes) else x for x in o["categories"][:]]
    return pd.Categorical.from_codes(o["codes"][:],c)

with h5py.File(f"{OUT}/03d_sc_adata.h5ad","r") as h:
    cg=h["layers/counts"]; shape=tuple(cg.attrs["shape"])
    X=sp.csr_matrix((cg["data"][:],cg["indices"][:],cg["indptr"][:]),shape=shape)
    obs=pd.DataFrame({k:cat(h,k) for k in ["leiden","gsm","primary","region"]})
    obs.index=[x.decode() if isinstance(x,bytes) else x for x in h["obs/_index"][:]]
    var=pd.DataFrame(index=[x.decode() if isinstance(x,bytes) else x for x in h["var/_index"][:]])
a=ad.AnnData(X=X,obs=obs,var=var)
sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)

PROG={**{f"T_{k}":v for k,v in TUMOR.items()}, **{f"H_{k}":v for k,v in HOST.items()}}
for k,genes in PROG.items():
    sc.tl.score_genes(a,[x for x in genes if x in a.var_names],score_name=k)
cols=list(PROG)
cl=a.obs.groupby("leiden",observed=True)[cols].mean()
comp=pd.crosstab(a.obs["leiden"],a.obs["primary"],normalize="index")*100

print("簇 | n | 主原发 | 最强肿瘤程序 | 最强宿主程序 | 判定")
rows=[]
for c in cl.index:
    n=(a.obs["leiden"]==c).sum()
    dom=comp.loc[c].sort_values(ascending=False)
    domp=dom.index[0]
    tcols=[f"T_{k}" for k in TUMOR if f"T_{k}" in cols]
    hcols=[f"H_{k}" for k in HOST if f"H_{k}" in cols]
    bt=cl.loc[c,tcols].idxmax()[2:]; bt_v=cl.loc[c,tcols].max()
    bh=cl.loc[c,hcols].idxmax()[2:]; bh_v=cl.loc[c,hcols].max()
    verdict="TUMOR" if bt_v>bh_v else "host"
    rows.append((c,n,f"{domp}({dom.iloc[0]:.0f}%)",f"{bt}:{bt_v:.2f}",f"{bh}:{bh_v:.2f}",verdict))
    print(f"{c:>3} | {n:5d} | {domp[:16]:<16}({dom.iloc[0]:3.0f}%) | {bt:<6}{bt_v:+.2f} | {bh:<8}{bh_v:+.2f} | {verdict}")

pd.DataFrame(rows,columns=["leiden","n","dom_primary","best_tumor","best_host","verdict"]).to_csv(f"{OUT}/03h_cluster_verdict.csv",index=False)
cl.to_csv(f"{OUT}/03h_cluster_scores.csv")
print("\n落盘: 03h_cluster_verdict.csv / 03h_cluster_scores.csv")
