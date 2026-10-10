#!/usr/bin/env python3
"""12 — GSE324453 定向检验：脑区标志在【肿瘤 vs 宿主】各区室的 region 效应。
并用 bulk(GSE322943) 的脑区 DE 基因集验证方向是否一致 → bulk↔snRNA 互证。
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

B=ROOT
IN=os.path.join(ROOT, "out", "GSE324453"); OUT=os.path.join(ROOT, "out")
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}
CBL_MK=["RELN","GABRA6","BARHL2","CALB2","CBLN3","PAX3","NEUROD1","GRIN2B","CBLN1","GABRA5"]
CBR_MK=["FOXG1","SIX3","OTX2","LHX2","EMX1","EMX2","NEUROD6"]

meta=pd.read_csv(f"{IN}/meta.csv",index_col=0)
feat=[l.strip() for l in open(f"{IN}/features.tsv")]
X=sp.csr_matrix(mmread(f"{IN}/counts.mtx")).T.tocsr()
lib=np.asarray(X.sum(1)).ravel(); lib[lib==0]=1
Xl=X.multiply(1e4/lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data)
meta=meta.reset_index(drop=True)
meta["region"]=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")

COMP={"Tumor":["Tumor cells","Tumor cells (prol.)"],"HostNeuron":["Neurons"],
      "HostAstro":["Astrocytes"],"HostODG":["ODG"],
      "Immune":["Myeloid cells","T/NK cells","B cells","Plasma cells"],
      "Vascular":["Endothelial cells","Pericytes"],"Stromal":["Stromal cells"]}
meta["comp"]="Other"
for k,v in COMP.items(): meta.loc[meta["common_cell_clusters"].isin(v),"comp"]=k
g2i={g:i for i,g in enumerate(feat)}

# 全队列 pseudobulk (sample × comp)
rows=[];key=[]
for (s,cp),idx in meta.groupby(["sampleDemuxIDstrict","comp"]).groups.items():
    if cp=="Other": continue
    ii=meta.index.get_indexer(idx)
    rows.append(np.asarray(Xl[ii].mean(0)).ravel()); key.append((s,cp,meta.loc[ii,"region"].iloc[0]))
pb=pd.DataFrame(np.vstack(rows),index=pd.MultiIndex.from_tuples(key,names=["sample","comp","region"]),columns=feat)

# bulk 脑区 DE 集（log2FC<0 = 小脑↑）
bk=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0)
bk=bk[bk["q"]<0.05]
CBL_SET=[g for g in bk.index[bk["log2FC"]>0] if g in g2i]
CBR_SET=[g for g in bk.index[bk["log2FC"]<0] if g in g2i]
print(f"bulk 小脑↑集 n={len(CBL_SET)} | 大脑↑集 n={len(CBR_SET)}")

def mwu_test(sub, gene_list, label):
    m=np.asarray(sub.index.get_level_values("region")=="cerebellum")
    if m.sum()<2 or (~m).sum()<2: return None
    V=sub[gene_list].values
    sc=V.mean(1)
    p=mannwhitneyu(sc[m],sc[~m])[1]
    return sc[m].mean(), sc[~m].mean(), p

print(f"\n{'区室':<12}{'n_cb':>5}{'n_cbr':>6} | {'小脑标志 cb':>10}{'cbr':>8}{'p':>8} | {'大脑标志 cb':>10}{'cbr':>8}{'p':>8} | {'bulk小脑↑ cb':>11}{'cbr':>8}{'p':>8}")
res=[]
for cp in COMP:
    sub=pb[pb.index.get_level_values("comp")==cp]
    if len(sub)==0: continue
    ncb=(sub.index.get_level_values("region")=="cerebellum").sum(); ncbr=len(sub)-ncb
    a=mwu_test(sub,[g for g in CBL_MK if g in sub.columns],"cbl")
    b=mwu_test(sub,[g for g in CBR_MK if g in sub.columns],"cbr")
    c=mwu_test(sub,CBL_SET,"bulkCBL")
    d=mwu_test(sub,CBR_SET,"bulkCBR")
    def f(x): return f"{x[0]:>10.3f}{x[1]:>8.3f}{x[2]:>8.3f}" if x else f"{'--':>26}"
    print(f"{cp:<12}{ncb:>5}{ncbr:>6} | {f(a) if a else '--':>26} | {f(b) if b else '--':>26} | {f(c) if c else '--':>26}")
    if a and c: res.append((cp,ncb,ncbr,a,b,c,d))
    # 逐基因小脑标志
    if a:
        print("     小脑标志逐基因(p):", " ".join(f"{g}:{mannwhitneyu(sub[g].values[np.asarray(sub.index.get_level_values('region')=='cerebellum')],sub[g].values[np.asarray(sub.index.get_level_values('region')!='cerebellum')])[1]:.3f}" for g in [x for x in CBL_MK if x in sub.columns]))
