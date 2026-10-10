#!/usr/bin/env python3
"""③c：把"脑组织含量"回归掉，看 cerebrum vs cerebellum 还剩多少【肿瘤内在】区域差异。
脑含量分用固定神经元/脑标志基因集（独立于 DE），逐基因残差化后重测。"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import subprocess
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

B = ROOT
D = os.path.join(ROOT, "data", "GSE322943"); OUT = os.path.join(ROOT, "out")
H = os.path.join(ROOT, "resources", "hgnc_ensembl_symbol.tsv.gz")

cnt = pd.read_excel(f"{D}/counts.xlsx", index_col=0)
cnt.index = [i.split(".")[0] for i in cnt.index]; cnt = cnt.groupby(level=0).sum()
h = pd.read_csv(H, sep="\t", dtype=str, low_memory=False).dropna(subset=["ensembl_gene_id"])
e2s = dict(zip(h["ensembl_gene_id"].str.split(".").str[0], h["symbol"]))

raw = subprocess.run(["curl","-sL","-A","Mozilla/5.0","--max-time","90",
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE322943&targ=gsm&form=text&view=brief"],
    capture_output=True, text=True).stdout.splitlines()
cur=None; md=[]
for l in raw:
    if l.startswith("!Sample_title"): cur={"title": l.split("= ",1)[1].strip()}
    elif l.startswith("!Sample_characteristics_ch1") and cur is not None:
        k,v=l.split("= ",1)[1].strip().split(": ",1); cur[k]=v
    elif l.startswith("^SAMPLE") and cur: md.append(cur); cur=None
md=pd.DataFrame(md).set_index("title")

X = cnt.rename(columns={c:c.replace("Exp_","") for c in cnt.columns})
smp=[s for s in X.columns if s in md.index]; X=X[smp]; m=md.loc[smp]
X=X.loc[(X>=10).sum(1)>=20]
cpm=np.log2(X.div(X.sum(axis=0),axis=1)*1e6+1)
cpm.index=[e2s.get(i,i) for i in cpm.index]
cpm=cpm.groupby(level=0).mean()
loc=m["location"].values
print("矩阵:", cpm.shape, "| cerebrum:", (loc=="cerebrum").sum(), "cerebellum:", (loc=="cerebellum").sum())

# 固定脑/神经元标志集（独立于 DE）
NEURO = ["RBFOX3","SYT1","SNAP25","GRIN1","GRIN2B","GABRA1","GABRB2","MBP","PLP1","MAG",
         "GFAP","S100B","NEFL","NEFM","NEFH","MAP2","TUBB3","RELN","STMN2","UCHL1","SYN1",
         "GAD1","GAD2","SLC17A7","SLC17A6","NEUROD2","TBR1","SATB2","FOXG1","PAX6","OLIG1",
         "OLIG2","CNP","MOG","CLDN11","ENO2","SYP","CAMK2A","CAMK2B","DLG4","GRIA1","GABBR2"]
present=[g for g in NEURO if g in cpm.index]
print("脑标志集命中:", len(present), "/", len(NEURO))
Z = cpm.loc[present].apply(lambda r:(r-r.mean())/r.std(ddof=0), axis=1)
brain = Z.mean(0)          # 每样本脑含量分
print("\n脑含量分 按脑区:", "cerebrum %.2f" % brain[loc=='cerebrum'].mean(),
      "| cerebellum %.2f" % brain[loc=='cerebellum'].mean(),
      "| MWU p=%.3g" % mannwhitneyu(brain[loc=='cerebrum'], brain[loc=='cerebellum'])[1])

# 逐基因：残差化掉脑含量，再测 region。
# 注意：不 z 标准化基因 → 残差仍是 log2 CPM 尺度，log2FC_adj 与 03b 的 log2FC 同单位（可画对角线）。
Mc = cpm.values
zz = (brain - brain.mean())/brain.std(ddof=0)
rows=[]
for i in range(Mc.shape[0]):
    y = Mc[i]; mu = y.mean()
    beta = np.dot(zz, y-mu)/ (np.dot(zz,zz)+1e-9)      # y ~ zz
    resid = (y-mu) - beta*zz                            # log2 CPM 尺度残差
    a=resid[loc=="cerebrum"]; b=resid[loc=="cerebellum"]
    try: p=mannwhitneyu(a,b,alternative="two-sided")[1]
    except ValueError: p=1.0
    rows.append((cpm.index[i], beta, b.mean()-a.mean(), p))
r=pd.DataFrame(rows,columns=["gene","brain_load","log2FC_adj","p"]).set_index("gene")
r["q"]=multipletests(r.p,method="fdr_bh")[1]; r=r.sort_values("q")
print(f"\n== 回归掉脑含量后 cerebrum vs cerebellum ==")
print(f"  残余 DE(q<0.05): {(r.q<0.05).sum()} / {len(r)}")
sig=r[r.q<0.05]
print("  小脑高 top12:", ", ".join(sig[sig.log2FC_adj>0].head(12).index))
print("  大脑高 top12:", ", ".join(sig[sig.log2FC_adj<0].head(12).index))
print("\n  (对照) 脑负载最高的 8 个基因:", ", ".join(r.sort_values('brain_load',ascending=False).head(8).index))
r.to_csv(f"{OUT}/03c_region_adjusted.csv")
print("\n落盘:", f"{OUT}/03c_region_adjusted.csv")
