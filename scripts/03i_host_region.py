#!/usr/bin/env python3
"""③i ③2 的定论：脑区信号在「宿主脑细胞」，且必须剔除 SCLC 混杂。

背景/坑：
 - GSE325196 的 region 与原发完全混杂：小脑仅 3 样本(SCLC/BC/NSCLC)，Occipital=1样本(TNBC)，
   Brain=仅 SCLC。对肿瘤细胞比脑区 = 比原发，无意义。
 - ⚠️ 03e/03f 把 SCLC 肿瘤细胞当成了「宿主神经元」：SCLC 是神经内分泌瘤，表达
   ASCL1/INSM1/CHGA/NEUROD1 + 神经元基因，与宿主神经元分不开。小脑「神经元」77% 来自
   SCLC 样本 GSM9597702 → 早先"信号在神经元"是 region↔原发混杂的假象。
 - 剔除 SCLC 后，宿主神经元仍带小脑标志 → 真信号在宿主（微环境），与 bulk GSE322943 一致。

瘦身加载（同 03g/03h）。
"""
import os
ROOT = os.environ.get("BCBM_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp, h5py
import scanpy as sc, anndata as ad
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

B=ROOT
OUT=os.path.join(ROOT, "out")

NEURON_CL=[0,11,13,21,22,23,28,29]          # 神经元样簇（含 SCLC 瘤）
MARKERS=["RELN","BARHL2","GABRA6","RBFOX3","CBLN3","CALB2","NEUROD1","GRIN2B"]
SCLC_NET=["ASCL1","INSM1","CHGA","SYP"]     # 用于标注 SCLC 瘤

def cat(h,k):
    o=h["obs/"+k]; c=[x.decode() if isinstance(x,bytes) else x for x in o["categories"][:]]
    return pd.Categorical.from_codes(o["codes"][:],c)

with h5py.File(f"{OUT}/03d_sc_adata.h5ad","r") as h:
    cg=h["layers/counts"]; X=sp.csr_matrix((cg["data"][:],cg["indices"][:],cg["indptr"][:]),shape=tuple(cg.attrs["shape"]))
    obs=pd.DataFrame({k:cat(h,k) for k in ["leiden","gsm","primary","region"]})
    var=np.array([x.decode() if isinstance(x,bytes) else x for x in h["var/_index"][:]])
a=ad.AnnData(X=X,obs=obs,var=pd.DataFrame(index=var))
sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
g2i={g:i for i,g in enumerate(a.var_names)}

sel=a.obs.leiden.astype(str).astype(int).isin(NEURON_CL).values
sub=a[sel].copy()
sub.obs["region2"]=np.where(sub.obs["region"]=="Cerebellum","cerebellum","lobe")

# 逐样本 SCLC 神经内分泌分，用于识别「其实是 SCLC 瘤」的样本
net=[g for g in SCLC_NET if g in g2i]
Mnet=np.asarray(sub.X[:,[g2i[g] for g in net]].todense()).mean(1)
d=pd.DataFrame({"net":Mnet,"gsm":sub.obs.gsm.values,"primary":sub.obs.primary.values})
smp=d.groupby(["gsm","primary"],observed=True)["net"].mean().reset_index()
print("== 各样本 神经内分泌分（ASCL1/INSM1/CHGA/SYP 均值）==")
print(smp.sort_values("net",ascending=False).to_string(index=False))

def test(df_obs, Mask, label):
    mkm=[g for g in MARKERS if g in g2i]
    M=np.asarray(df_obs.X[:,[g2i[g] for g in mkm]].todense())
    cb=Mask; lb=~Mask
    rows=[]
    for j,g in enumerate(mkm):
        v=M[:,j]
        pc=mannwhitneyu(v[cb],v[lb])[1]
        t=pd.DataFrame({"v":v,"gsm":df_obs.obs.gsm.values,"r":np.where(cb,"cb","lb")})
        sm=t.groupby(["gsm","r"],observed=True)["v"].mean()
        cb2=sm[sm.index.get_level_values(1)=="cb"].values; lb2=sm[sm.index.get_level_values(1)=="lb"].values
        ps=mannwhitneyu(cb2,lb2)[1] if len(cb2)>1 and len(lb2)>1 else np.nan
        rows.append((g,v[cb].mean(),v[lb].mean(),pc,cb2.mean(),lb2.mean(),ps))
    r=pd.DataFrame(rows,columns=["gene","cb_cell","lb_cell","p_cell","cb_smp","lb_smp","p_smp"])
    print(f"\n== {label} (小脑样本 {int(sum(1 for x in set(df_obs.obs.gsm[cb])))} vs 脑叶) ==")
    print(r.round(4).to_string(index=False))
    return r

# A) 全神经元样簇（含 SCLC 瘤）—— 早先错误做法
test(sub, (sub.obs.region2=="cerebellum").values, "A. 全神经元样簇(含SCLC瘤) 小脑vs脑叶")

# B) 剔除 SCLC —— 正确做法（宿主神经元）
keep=(sub.obs.primary.values!="SCLC")
subB=sub[keep].copy()
rB=test(subB, (subB.obs.region2=="cerebellum").values, "B. 剔除SCLC后的宿主神经元 小脑vs脑叶")
rB.to_csv(f"{OUT}/03i_host_neuron_region.csv",index=False)
print("\n落盘: 03i_host_neuron_region.csv")
