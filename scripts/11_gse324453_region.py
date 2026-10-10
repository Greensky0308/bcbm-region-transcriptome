#!/usr/bin/env python3
"""11 — GSE324453（BC 脑转移 snRNA，与 bulk GSE322943 同队列）脑区检验。

目标（③2 的单细胞佐证）：脑区(cerebellum vs cerebrum)差异在【肿瘤细胞】还是【宿主脑细胞】？
本数据集 BC-only、region 与原发不混杂（全 TNBC 队列），是唯一干净的资源。

输入：分析/产物/GSE324453/{counts.mtx,features.tsv,barcodes.tsv,meta.csv}
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

B=ROOT
IN=os.path.join(ROOT, "out", "GSE324453"); OUT=os.path.join(ROOT, "out")
CEREBELLUM={"NCH6239","NCH5119","NCH1367","NCH1098"}   # 来自 GEO 样本元数据 bcbm location

t0=time.time()
meta=pd.read_csv(f"{IN}/meta.csv",index_col=0)
feat=[l.strip() for l in open(f"{IN}/features.tsv")]
barc=[l.strip() for l in open(f"{IN}/barcodes.tsv")]
print("meta:",meta.shape,"| features:",len(feat),"barcodes:",len(barc),f"| {time.time()-t0:.0f}s",flush=True)

X=sp.csr_matrix(mmread(f"{IN}/counts.mtx"))     # genes × cells
X=X.T.tocsr()                                    # cells × genes
print("counts:",X.shape,"nnz",X.nnz,f"| {time.time()-t0:.0f}s",flush=True)

region=np.where(meta["sampleDemuxIDstrict"].isin(CEREBELLUM),"cerebellum","cerebrum")
meta["region"]=region
ct=meta["common_cell_clusters"]
COMP={
 "Tumor":      ["Tumor cells","Tumor cells (prol.)"],
 "HostNeuron": ["Neurons"], "HostAstro":["Astrocytes"], "HostODG":["ODG"],
 "Immune":     ["Myeloid cells","T/NK cells","B cells","Plasma cells"],
 "Vascular":   ["Endothelial cells","Pericytes"], "Stromal":["Stromal cells"],
}
meta["comp"]="Other"
for k,v in COMP.items(): meta.loc[ct.isin(v),"comp"]=k
print("\nregion:",pd.Series(region).value_counts().to_dict(),flush=True)
print("comp:",meta["comp"].value_counts().to_dict(),flush=True)
print("\n样本级 组成% × region(前几类):",flush=True)
print(pd.crosstab(meta["sampleDemuxIDstrict"],meta["comp"],normalize="index").mul(100).round(1).to_string(),flush=True)

# lognorm
sc_lib=np.asarray(X.sum(1)).ravel(); sc_lib[sc_lib==0]=1
Xl=X.multiply(1e4/sc_lib[:,None]).tocsr(); Xl.data=np.log1p(Xl.data)

# marker 验证
g2i={g:i for i,g in enumerate(feat)}
def mean_expr(gene,mask):
    j=g2i.get(gene);
    return float(np.asarray(Xl[mask,j].mean())) if j is not None else np.nan
print("\n== 标志验证(各组均值) ==",flush=True)
for comp in ["Tumor","HostNeuron","HostAstro","HostODG","Immune"]:
    is_m=(meta.comp==comp).values
    print(f" [{comp:10s}] EPCAM={mean_expr('EPCAM',is_m):.2f} KRT8={mean_expr('KRT8',is_m):.2f} RBFOX3={mean_expr('RBFOX3',is_m):.2f} MBP={mean_expr('MBP',is_m):.2f} PTPRC={mean_expr('PTPRC',is_m):.2f}",flush=True)

# pseudobulk: (sample × comp)
meta=meta.reset_index(drop=True)
rows=[]; mat=[]; key=[]
for (s,cp),idx in meta.groupby(["sampleDemuxIDstrict","comp"]).groups.items():
    if cp=="Other": continue
    ii=meta.index.get_indexer(idx)
    mat.append(np.asarray(Xl[ii].mean(0)).ravel()); key.append((s,cp,meta.loc[ii,"region"].iloc[0]))
pb=pd.DataFrame(np.vstack(mat),index=pd.MultiIndex.from_tuples(key,names=["sample","comp","region"]),columns=feat)
print("\npseudobulk:",pb.shape,flush=True)

def region_de(comps,label):
    sub=pb[pb.index.get_level_values("comp").isin(comps)]
    print(f"\n== {label} (cerebellum {sub[sub.index.get_level_values('region')=='cerebellum'].shape[0]} vs cerebrum {sub[sub.index.get_level_values('region')=='cerebrum'].shape[0]} 样本) ==",flush=True)
    g=sub.columns.tolist()
    V=sub.values; m=np.asarray(sub.index.get_level_values("region")=="cerebellum")
    if m.sum()<2 or (~m).sum()<2: print("  组太小"); return None
    p=[mannwhitneyu(V[m,i],V[~m,i])[1] if (V[m,i].std()+V[~m,i].std())>0 else 1.0 for i in range(len(g))]
    d=V[m].mean(0)-V[~m].mean(0)
    r=pd.DataFrame({"gene":g,"lfc":d,"p":p}).set_index("gene")
    r["q"]=multipletests(r.p,method="fdr_bh")[1]; r=r.sort_values("q")
    print(f"  DE(q<.05)={(r.q<.05).sum()} | cerebellum↑: {', '.join(r[(r.q<0.05)&(r.lfc>0)].head(10).index)}",flush=True)
    print(f"  cerebrum↑: {', '.join(r[(r.q<0.05)&(r.lfc<0)].head(10).index)}",flush=True)
    return r

for comps,label in [(["Tumor"],"肿瘤细胞"),(["HostNeuron"],"宿主神经元"),
                    (["HostAstro"],"宿主星形"),(["HostODG"],"宿主少突"),
                    (["Immune"],"免疫"),(["Vascular"],"血管"),(["Stromal"],"基质")]:
    if comps[0] in pb.index.get_level_values("comp"):
        rr=region_de(comps,label)
        if rr is not None: rr.to_csv(f"{OUT}/11_GSE324453_region_{comps[0]}.csv")
print(f"\n总 {time.time()-t0:.0f}s",flush=True)
