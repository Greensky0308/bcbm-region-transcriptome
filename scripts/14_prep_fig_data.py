#!/usr/bin/env python3
"""14 — 备好绘图数据（缓存）：
  A) bulk GSE322943: log2CPM 样本×基因 + region 标签 + 03b DE  → figdata/bulk_*.csv
  B) snRNA GSE324453: 每样本×区室 pseudobulk + bulk小脑↑集分数 → figdata/snrna_pb.csv
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import subprocess
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import mannwhitneyu

B=ROOT
OUT=os.path.join(ROOT, "out"); FIG=os.path.join(ROOT, "out", "figdata")
import os; os.makedirs(FIG, exist_ok=True)
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}
HOST=["Neurons","Astrocytes","ODG","Myeloid cells","T/NK cells","B cells","Plasma cells",
      "Endothelial cells","Pericytes","Stromal cells"]
TUM=["Tumor cells","Tumor cells (prol.)"]

# ---------- A) bulk ----------
cnt=pd.read_excel(os.path.join(ROOT, "data", "GSE322943", "counts.xlsx"),index_col=0)
cnt.index=[i.split(".")[0] for i in cnt.index]; cnt=cnt.groupby(level=0).sum()
raw=subprocess.run(["curl","-sL","-A","Mozilla/5.0","--max-time","90",
  "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE322943&targ=gsm&form=text&view=brief"],
  capture_output=True,text=True).stdout.splitlines()
cur=None; md=[]
for l in raw:
    if l.startswith("!Sample_title"): cur={"title":l.split("= ",1)[1].strip()}
    elif l.startswith("!Sample_characteristics_ch1") and cur is not None:
        k,v=l.split("= ",1)[1].strip().split(": ",1); cur[k]=v
    elif l.startswith("^SAMPLE") and cur: md.append(cur); cur=None
md=pd.DataFrame(md).set_index("title")
X=cnt.rename(columns={c:c.replace("Exp_","") for c in cnt.columns})
samples=[s for s in X.columns if s in md.index]; X=X[samples]; m=md.loc[samples]
keep=(X>=10).sum(1)>=20; X=X.loc[keep]
cpm=np.log2(X.div(X.sum(axis=0),axis=1)*1e6+1)
cpm.to_csv(f"{FIG}/bulk_cpm.csv")
pd.DataFrame({"sample":samples,"location":m["location"].values,"batch":m["batch"].values}).to_csv(f"{FIG}/bulk_meta.csv",index=False)
print("bulk cpm:",cpm.shape,"| region:",m["location"].value_counts().to_dict())

# ---------- B) snRNA ----------
IN=f"{OUT}/GSE324453"
meta=pd.read_csv(f"{IN}/meta.csv",index_col=0); feat=[l.strip() for l in open(f"{IN}/features.tsv")]
Xs=sp.csr_matrix(mmread(f"{IN}/counts.mtx")).T.tocsr()
lib=np.asarray(Xs.sum(1)).ravel(); lib[lib==0]=1
Xl=Xs.multiply(1e4/lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data); meta=meta.reset_index(drop=True)
meta["region"]=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")
meta["comp"]="Other"
for k,v in {"Tumor":TUM,"HostNeuron":["Neurons"],"HostAstro":["Astrocytes"],"HostODG":["ODG"],
            "Immune":["Myeloid cells","T/NK cells","B cells","Plasma cells"],
            "Vascular":["Endothelial cells","Pericytes"],"Stromal":["Stromal cells"]}.items():
    meta.loc[meta["common_cell_clusters"].isin(v),"comp"]=k
meta.loc[(meta["comp"]!="Other")&(meta["comp"]!="Tumor"),"comp2"]="Host"
meta["comp2"]=meta["comp2"].fillna(meta["comp"])   # Tumor / Host / 各宿主亚类
g2i={g:i for i,g in enumerate(feat)}
bk=pd.read_csv(f"{OUT}/03b_region_de_symbols.csv",index_col=0); bk=bk[bk.q<0.05]
SET=[g for g in bk.index[bk.log2FC>0] if g in g2i]; SETR=[g for g in bk.index[bk.log2FC<0] if g in g2i]
cols=[g2i[g] for g in SET]; colsr=[g2i[g] for g in SETR]
print("bulk小脑↑集 n=",len(SET),"| 大脑↑集 n=",len(SETR))

def pb(labels):
    rows=[]
    sub=meta[meta["comp"].isin(labels)]
    for (s,r),idx in sub.groupby(["sampleDemuxIDstrict","region"]).groups.items():
        ii=meta.index.get_indexer(idx); v=np.asarray(Xl[ii].mean(0)).ravel()
        rows.append((s,r,len(ii),v[cols].mean(),v[colsr].mean()))
    return pd.DataFrame(rows,columns=["sample","region","n_cells","sig_CBL","sig_CBR"])

COMPS={"Tumor":["Tumor"],"HostNeuron":["HostNeuron"],"HostAstro":["HostAstro"],"HostODG":["HostODG"],
       "Immune":["Immune"],"Vascular":["Vascular"],"Stromal":["Stromal"]}
dat=[]
for nm,cl in COMPS.items():
    d=pb(cl); d["comp"]=nm; dat.append(d)
allhost=pb(["HostNeuron","HostAstro","HostODG","Immune","Vascular","Stromal"]); allhost["comp"]="Host(all)"; dat.append(allhost)
pbdf=pd.concat(dat,ignore_index=True)
pbdf.to_csv(f"{FIG}/snrna_pb.csv",index=False)
print("snRNA pseudobulk 行:",pbdf.shape)
for nm in list(COMPS)+["Host(all)"]:
    d=pbdf[pbdf.comp==nm]; cb=d[d.region=="cerebellum"]; lbr=d[d.region=="cerebrum"]
    if len(cb)>=2 and len(lbr)>=2:
        p=mannwhitneyu(cb.sig_CBL,lbr.sig_CBL)[1]
        print(f"  {nm:12s} n_cb={len(cb)} n_cbr={len(lbr)} cb={cb.sig_CBL.mean():.3f} cbr={lbr.sig_CBL.mean():.3f} p={p:.4f}")
print("\n落盘:",FIG)
