#!/usr/bin/env python3
"""③b 新问法：乳腺癌脑转移 大脑(cerebrum) vs 小脑(cerebellum) 的转录组差异。
GSE322943 n=100。先查 location×batch 混杂，再 DE + 阳性对照。"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import subprocess, collections
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

D = os.path.join(ROOT, "data", "GSE322943")
OUT = os.path.join(ROOT, "out")

cnt = pd.read_excel(f"{D}/counts.xlsx", index_col=0)
cnt.index = [i.split(".")[0] for i in cnt.index]      # 去 ENSG 版本号
cnt = cnt.groupby(level=0).sum()
print("counts:", cnt.shape)

# 元数据
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
print("元数据:", md.shape, list(md.columns))

# 列名 Exp_NCH710e -> NCH710e
col2t = {c: c.replace("Exp_","") for c in cnt.columns}
X = cnt.rename(columns=col2t)
samples = [s for s in X.columns if s in md.index]
X = X[samples]; m = md.loc[samples]
print("对齐样本:", len(samples))
print("\n⚠️ location × batch:\n", pd.crosstab(m["location"], m["batch"]).to_string())

keep = (X>=10).sum(1) >= 20
X = X.loc[keep]; cpm = np.log2(X.div(X.sum(axis=0),axis=1)*1e6+1)
genes = np.array(X.index); M = cpm.values
loc = m["location"].values
print("\n表达矩阵:", M.shape, "| cerebrum:", (loc=="cerebrum").sum(), "cerebellum:", (loc=="cerebellum").sum())

print("\n--- 阳性对照：脑区/细胞标志 ---")
S2S = {"CALB1":"小脑浦肯野","CALB2":"小脑","NEFH":"神经元","MBP":"髓鞘","GFAP":"星形胶质",
       "PTPRC":"免疫","EPCAM":"上皮(瘤)"}
# ENSG->symbol 需要映射；先尝试用已带符号的行
have_sym = any(not g.startswith("ENSG") for g in genes[:50])
print("索引是符号?", have_sym)
if not have_sym:
    print("  (索引为 ENSG，符号映射后补跑)")

rows=[]
for i in range(M.shape[0]):
    a=M[i][loc=="cerebrum"]; b=M[i][loc=="cerebellum"]
    try: p=mannwhitneyu(a,b,alternative="two-sided")[1]
    except ValueError: p=1.0
    rows.append((genes[i],a.mean()-b.mean(),p))
r=pd.DataFrame(rows,columns=["gene","log2FC","p"]).set_index("gene")
r["q"]=multipletests(r.p,method="fdr_bh")[1]
r=r.sort_values("q")
print(f"\n== cerebrum(66) vs cerebellum(34) ==")
print(f"  DE(q<0.05): {(r.q<0.05).sum()} / {len(r)}")
print("  top15:", ", ".join(r.head(15).index))
r.to_csv(f"{OUT}/03b_region_de.csv")
print("\n落盘:", f"{OUT}/03b_region_de.csv")

# ENSG -> symbol version (same rows, index replaced) for the downstream scripts
_h = pd.read_csv(os.path.join(ROOT, "resources", "hgnc_ensembl_symbol.tsv.gz"),
                 sep="\t", dtype=str).dropna(subset=["ensembl_gene_id"])
_e2s = dict(zip(_h["ensembl_gene_id"].str.split(".").str[0], _h["symbol"]))
rs = r.copy(); rs.index = [_e2s.get(g, g) for g in rs.index]
rs.to_csv(f"{OUT}/03b_region_de_symbols.csv")
print("落盘:", f"{OUT}/03b_region_de_symbols.csv")
