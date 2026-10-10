#!/usr/bin/env python3
"""13 — GSE324453 定论 + 留一法：bulk(GSE322943) 脑区 DE 集在【宿主 vs 肿瘤】的方向与稳健性。

结论：bulk 小脑↑集在宿主细胞显著(cb 0.228 vs cbr 0.058, p<1e-3)、留一法去任一小脑样本仍 p=0.001；
      在肿瘤细胞无(cb 0.096 vs cbr 0.084, p=0.505)。→ 脑区信号=宿主脑组织印记，不在肿瘤内。
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import mannwhitneyu
B=ROOT
IN=os.path.join(ROOT, "out", "GSE324453"); OUT=os.path.join(ROOT, "out")
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}
HOST=["Neurons","Astrocytes","ODG","Myeloid cells","T/NK cells","B cells","Plasma cells","Endothelial cells","Pericytes","Stromal cells"]
TUM=["Tumor cells","Tumor cells (prol.)"]
meta=pd.read_csv(f"{IN}/meta.csv",index_col=0); feat=[l.strip() for l in open(f"{IN}/features.tsv")]
X=sp.csr_matrix(mmread(f"{IN}/counts.mtx")).T.tocsr(); lib=np.asarray(X.sum(1)).ravel(); lib[lib==0]=1
Xl=X.multiply(1e4/lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data); meta=meta.reset_index(drop=True)
meta["region"]=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")
g2i={g:i for i,g in enumerate(feat)}
bk=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0); bk=bk[bk["q"]<0.05]
SET=[g for g in bk.index[bk["log2FC"]>0] if g in g2i]; cols=[g2i[g] for g in SET]
def pb(mask):
    return {(s,r):np.asarray(Xl[meta.index.get_indexer(idx)].mean(0)).ravel()
            for (s,r),idx in meta[mask].groupby(["sampleDemuxIDstrict","region"]).groups.items()}
def test(d,drop=None):
    ks=[k for k in d if k[0]!=drop]
    cb=np.array([d[k][cols].mean() for k in ks if k[1]=="cerebellum"])
    lb=np.array([d[k][cols].mean() for k in ks if k[1]=="cerebrum"])
    return cb.mean(),lb.mean(),mannwhitneyu(cb,lb)[1],len(cb)
rows=[]
for nm,cc in [("HOST",HOST),("TUMOR",TUM)]:
    d=pb(meta["common_cell_clusters"].isin(cc).values)
    m,l,p,n=test(d); rows.append((nm,"all",m,l,p,n))
    print(f"{nm:6s} all         cb=%.3f cbr=%.3f p=%.4f n_cb=%d"%(m,l,p,n))
    for drop in sorted(CEREBELLUM):
        m,l,p,n=test(d,drop); rows.append((nm,f"drop_{drop}",m,l,p,n))
        print(f"{nm:6s} drop_{drop} cb=%.3f cbr=%.3f p=%.4f n_cb=%d"%(m,l,p,n))
pd.DataFrame(rows,columns=["compartment","test","cb","cbr","p","n_cb"]).to_csv(f"{OUT}/13_GSE324453_LODO.csv",index=False)
print("\n落盘: 13_GSE324453_LODO.csv")
