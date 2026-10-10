#!/usr/bin/env python3
"""③2 单核处理：GSE325196 snRNA → QC → 聚类 → 注释 肿瘤/正常脑/免疫。
只为『脑区信号在肿瘤细胞内还是微环境』做铺垫。"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os, gzip, glob, subprocess, tarfile, shutil
import numpy as np, pandas as pd

B  = ROOT
RAW= os.path.join(ROOT, "data", "GSE325196")
EX = f"{RAW}/extracted"
OUT= os.path.join(ROOT, "out")

# 1) 解包
if not os.path.isdir(EX) or not glob.glob(f"{EX}/*barcodes*"):
    os.makedirs(EX, exist_ok=True)
    with tarfile.open(f"{RAW}/RAW.tar") as t:
        t.extractall(EX)
    print("解包完成:", len(os.listdir(EX)), "文件")

# 2) GSM → 元数据（脑区/原发类型/是否单核）
lines = subprocess.run(["curl","-sL","-A","Mozilla/5.0","--max-time","90",
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325196&targ=gsm&form=text&view=brief"],
    capture_output=True, text=True).stdout.splitlines()
cur=None; meta=[]
for l in lines:
    if l.startswith("^SAMPLE"):
        if cur: meta.append(cur)
        cur={"gsm": l.split("= ",1)[1].strip()}
    elif cur is not None:
        if l.startswith("!Sample_title"): cur["title"]=l.split("= ",1)[1].strip()
        elif l.startswith("!Sample_source_name_ch1"): cur["source"]=l.split("= ",1)[1].strip()
        elif l.startswith("!Sample_characteristics_ch1"):
            kv=l.split("= ",1)[1].strip()
            if ": " in kv: k,v=kv.split(": ",1); cur[k]=v
if cur: meta.append(cur)
meta=pd.DataFrame(meta)
print("GSM 元数据:", meta.shape, list(meta.columns))

# 3) 只取单核样本
sn = meta[meta["title"].str.startswith("Single nuclei")].copy()
print("单核样本:", len(sn))

# 4) 逐个载入（10x）
import scanpy as sc, tempfile, shutil
ads=[]
for _,row in sn.iterrows():
    gsm=row["gsm"]
    files=glob.glob(f"{EX}/{gsm}_*")
    b=[f for f in files if "barcodes" in f]; mtx=[f for f in files if "matrix" in f]; feat=[f for f in files if "features" in f]
    if not (b and mtx and feat):
        print("  跳过(缺文件):", gsm); continue
    tmp=tempfile.mkdtemp()
    import gzip as _gz
    def is_gz(p):
        with open(p,"rb") as fh: return fh.read(2)==b"\x1f\x8b"
    def to_plain(src, dst):
        if is_gz(src):
            with _gz.open(src,"rb") as fi, open(dst,"wb") as fo: shutil.copyfileobj(fi,fo)
        else:
            shutil.copy(src,dst)
    to_plain(b[0],    f"{tmp}/barcodes.tsv")
    to_plain(feat[0], f"{tmp}/features.tsv")
    to_plain(mtx[0],  f"{tmp}/matrix.mtx")
    bc = pd.read_csv(f"{tmp}/barcodes.tsv", header=None, sep="\t")[0].astype(str).values
    ft = pd.read_csv(f"{tmp}/features.tsv", header=None, sep="\t")
    sym = (ft[1] if ft.shape[1] >= 2 else ft[0]).astype(str).values
    from scipy.io import mmread
    Xm = mmread(f"{tmp}/matrix.mtx").T.tocsr().astype("float32")
    import anndata as ad
    d = ad.AnnData(Xm, obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=sym))
    d.var_names_make_unique()
    shutil.rmtree(tmp, ignore_errors=True)
    d.obs["gsm"]=gsm; d.obs["region"]=row.get("source",row.get("tissue")); d.obs["primary"]=row.get("cell type")
    ads.append(d)
    print("  载入", gsm, row.get("source"), row.get("cell type"), d.shape)

adata = ad.concat(ads, join="outer", label="batch", fill_value=0)
print("合并:", adata.shape)

# 5) QC
adata.var["mt"] = adata.var_names.str.startswith(("MT-","MT."))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
n0=adata.n_obs
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
print(f"QC: {n0} → {adata.n_obs} 细胞")

# 6) 归一 + 聚类
adata.layers["counts"]=adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata.raw=adata
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30, svd_solver="arpack", random_state=0)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=0)
sc.tl.leiden(adata, resolution=1.0, random_state=0)
print("聚类数:", adata.obs["leiden"].nunique())

# 7) 标志基因打分（肿瘤/神经元/胶质/免疫）
sc.tl.score_genes(adata, ["EPCAM","KRT8","KRT18","KRT19","ELF3","KRT7"], score_name="tumor")
sc.tl.score_genes(adata, ["RBFOX3","SYT1","SNAP25","NEFL","NEFM","NEFH","MAP2","SYP","SLC17A7"], score_name="neuron")
sc.tl.score_genes(adata, ["GFAP","AQP4","S100B","MBP","PLP1","MAG","MOBP","OLIG1","OLIG2"], score_name="glia")
sc.tl.score_genes(adata, ["PTPRC","CD3D","CD3E","CD14","CD68","MS4A1","NKG7"], score_name="immune")

cl = adata.obs.groupby("leiden")[["tumor","neuron","glia","immune"]].mean()
print("\n各 cluster 标志均分:")
print(cl.round(2).to_string())
adata.write(f"{OUT}/03d_sc_adata.h5ad")
cl.to_csv(f"{OUT}/03d_cluster_markers.csv")
print("\n落盘:", f"{OUT}/03d_sc_adata.h5ad")
