#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-audit the global detrital-zircon compilation against the training domain."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

FEATURES=["Ce","Nd","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","U","Hf","Ti"]
CN={"Ce":"Ce（铈）","Nd":"Nd（钕）","Eu":"Eu（铕）","Gd":"Gd（钆）","Tb":"Tb（铽）","Dy":"Dy（镝）","Ho":"Ho（钬）","Er":"Er（铒）","Tm":"Tm（铥）","Yb":"Yb（镱）","Lu":"Lu（镥）","U":"U（铀）","Hf":"Hf（铪）","Ti":"Ti（钛）"}

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def source_id(d):
 ref=d["Reference"].fillna("REFERENCE_MISSING").astype(str).str.strip()
 region=d["Region"].fillna("REGION_MISSING").astype(str).str.strip()
 sub=d["Sub_region"].fillna("SUBREGION_MISSING").astype(str).str.strip()
 # The compiled Sample field is inconsistent: in some papers it denotes a
 # sediment sample, in others an individual grain/spot. Use publication ×
 # subregion as the conservative bootstrap/source unit and retain Sample raw.
 return ref+" || "+region+" || "+sub

def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(exist_ok=True)
 g=pd.read_excel(a.global_xlsx,header=2)
 needed=["Selected_age","P","La","Th/U"]+FEATURES
 for c in needed:g[c]=pd.to_numeric(g[c],errors="coerce")
 g=g[g.Selected_age.between(0,4000)].copy();g[FEATURES]=g[FEATURES].where(g[FEATURES]>0)
 g["GLOBAL_RECORD_ID"]=[f"WANG2025_{i+1:06d}" for i in range(len(g))]
 g["SOURCE_ID"]=source_id(g);g["N_FEATURES_MEASURED"]=g[FEATURES].notna().sum(axis=1)
 g["MISSING_FEATURES"]=g[FEATURES].isna().apply(lambda r:";".join(r.index[r]),axis=1)
 g["QC_P_GT750"]=(g.P>750).fillna(False);g["QC_LA_GT1"]=(g.La>1).fillna(False);g["QC_TH_U_LT0_1"]=(g["Th/U"]<.1).fillna(False)
 g["ZIRCON_QC_PASS"]=~(g.QC_P_GT750|g.QC_LA_GT1|g.QC_TH_U_LT0_1)
 g["COMPOSITION_DUPLICATE"]=g.duplicated(subset=["Reference","Sample","Selected_age"]+FEATURES,keep=False)

 t=pd.read_csv(a.training_csv,encoding="utf-8-sig")
 x=t[[CN[f] for f in FEATURES]].apply(pd.to_numeric,errors="coerce");x.columns=FEATURES;x=x.where(x>0)
 lx=np.log10(x);imp=SimpleImputer(strategy="median").fit(lx);xi=imp.transform(lx)
 scaler=RobustScaler(quantile_range=(10,90)).fit(xi);z=scaler.transform(xi);groups=t[a.training_group].astype(str).to_numpy()
 # Calibrate distance using nearest sample from another training location.
 train_dist=[]
 for i in range(len(z)):
  other=groups!=groups[i]
  train_dist.append(np.sqrt(((z[other]-z[i])**2).sum(axis=1)).min())
 train_dist=np.asarray(train_dist);threshold=float(np.quantile(train_dist,a.domain_quantile))
 candidate=g[(g.N_FEATURES_MEASURED>=a.min_features)&g.ZIRCON_QC_PASS].copy()
 zg=scaler.transform(imp.transform(np.log10(candidate[FEATURES])))
 candidate["DOMAIN_NN_DISTANCE"]=NearestNeighbors(n_neighbors=1).fit(z).kneighbors(zg,return_distance=True)[0][:,0]
 candidate["DOMAIN_THRESHOLD"]=threshold;candidate["DOMAIN_PASS"]=candidate.DOMAIN_NN_DISTANCE<=threshold
 candidate["DOMAIN_PERCENTILE_APPROX"]=np.searchsorted(np.sort(train_dist),candidate.DOMAIN_NN_DISTANCE,side="right")/len(train_dist)*100
 candidate["DATA_GRADE"]=np.select([
  candidate.DOMAIN_PASS&(candidate.N_FEATURES_MEASURED==14)&~candidate.COMPOSITION_DUPLICATE,
  candidate.DOMAIN_PASS&(candidate.N_FEATURES_MEASURED>=12),
 ],["A_complete_in_domain","B_1to2_missing_in_domain"],default="C_out_of_domain_or_duplicate")
 in_domain=candidate[candidate.DOMAIN_PASS].copy();core=in_domain[~in_domain.COMPOSITION_DUPLICATE].copy();complete=core[core.N_FEATURES_MEASURED==14].copy();warn=candidate[~candidate.DOMAIN_PASS].copy()
 # Duplicate flags are retained for audit; exact duplicates get a sensitivity exclusion file.
 dup=candidate[candidate.COMPOSITION_DUPLICATE].copy()
 candidate.to_csv(out/"01_扩展候选_10844.csv",index=False,encoding="utf-8-sig")
 core.to_csv(out/"02_主分析核心数据.csv",index=False,encoding="utf-8-sig")
 complete.to_csv(out/"03_完整14元素敏感性数据.csv",index=False,encoding="utf-8-sig")
 warn.to_csv(out/"04_训练域外警示数据.csv",index=False,encoding="utf-8-sig")
 dup.to_csv(out/"05_潜在重复记录审计.csv",index=False,encoding="utf-8-sig")
 # Source and 100-Myr diagnostics.
 src=core.groupby(["Reference","Region"],dropna=False).agg(锆石数=("GLOBAL_RECORD_ID","size"),来源单元数=("SOURCE_ID","nunique"),原始Sample字段唯一值数=("Sample","nunique"),年龄最小_Ma=("Selected_age","min"),年龄最大_Ma=("Selected_age","max"),缺失元素记录数=("N_FEATURES_MEASURED",lambda s:int((s<14).sum()))).reset_index().sort_values("锆石数",ascending=False)
 src["颗粒占比"]=src.锆石数/len(core);src.to_csv(out/"06_来源贡献统计.csv",index=False,encoding="utf-8-sig")
 bins=np.arange(0,4100,100);candidate["AGE_BIN_100MA"]=pd.cut(candidate.Selected_age,bins,right=False,include_lowest=True);core["AGE_BIN_100MA"]=pd.cut(core.Selected_age,bins,right=False,include_lowest=True)
 rows=[]
 for b,all_d in candidate.groupby("AGE_BIN_100MA",observed=False):
  d=core[core.AGE_BIN_100MA==b]
  rows.append({"年龄窗":str(b),"年龄中心_Ma":float(b.left+50),"扩展候选数":len(all_d),"核心锆石数":len(d),"域外警示数":int((~all_d.DOMAIN_PASS).sum()),"域外比例":float((~all_d.DOMAIN_PASS).mean()) if len(all_d) else np.nan,"独立来源数":d.SOURCE_ID.nunique(),"独立文献数":d.Reference.nunique(),"14元素完整数":int((d.N_FEATURES_MEASURED==14).sum()),"缺1至2元素数":int((d.N_FEATURES_MEASURED<14).sum()),"缺失比例":float((d.N_FEATURES_MEASURED<14).mean()) if len(d) else np.nan,"域距离中位数":d.DOMAIN_NN_DISTANCE.median(),"最大来源占比":float(d.SOURCE_ID.value_counts(normalize=True).max()) if len(d) else np.nan})
 age=pd.DataFrame(rows);age.to_csv(out/"07_每100Ma数据质量.csv",index=False,encoding="utf-8-sig")
 miss=pd.DataFrame({"元素":FEATURES,"核心数据实测数":[core[f].notna().sum() for f in FEATURES]});miss["核心数据完整率"]=miss["核心数据实测数"]/len(core);miss.to_csv(out/"08_元素完整率.csv",index=False,encoding="utf-8-sig")
 stat={"global_input_rows":len(g),"extended_candidate_rows":len(candidate),"in_domain_before_dedup_rows":len(in_domain),"core_in_domain_deduplicated_rows":len(core),"complete_14_feature_rows":len(complete),"out_of_domain_warning_rows":len(warn),"potential_duplicate_rows":len(dup),"references_in_core":int(core.Reference.nunique()),"source_ids_in_core":int(core.SOURCE_ID.nunique()),"domain_quantile":a.domain_quantile,"domain_threshold":threshold,"training_rows":len(t),"training_locations":int(t[a.training_group].nunique()),"global_sha256":sha(a.global_xlsx),"training_sha256":sha(a.training_csv)}
 (out/"00_审计摘要.json").write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding="utf-8")
 # Figures
 plt.rcParams.update({"font.family":"Noto Sans CJK JP","font.sans-serif":["Noto Sans CJK JP","Microsoft YaHei","SimHei"],"axes.unicode_minus":False,"figure.dpi":150})
 fig,ax=plt.subplots(figsize=(8,4.5));ax.hist(candidate.DOMAIN_NN_DISTANCE,bins=60,color="#4677a8");ax.axvline(threshold,color="#d1495b",ls="--",label=f"跨地点训练距离{a.domain_quantile:.0%}阈值");ax.set(xlabel="训练域最近邻距离",ylabel="锆石数",title="全球数据与训练适用域");ax.legend();fig.tight_layout();fig.savefig(out/"figures/01_训练域距离.png");plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(age.年龄中心_Ma,age.核心锆石数,label="核心锆石数");ax2=ax.twinx();ax2.plot(age.年龄中心_Ma,age.缺失比例*100,color="#d95f02",label="缺失1–2元素比例");ax.set(xlabel="年龄（Ma）",ylabel="锆石数",title="各年龄窗数量与缺失比例");ax2.set_ylabel("缺失比例（%）");fig.tight_layout();fig.savefig(out/"figures/02_年龄窗质量.png");plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(miss.元素,miss.核心数据完整率*100,color="#4d9b73");ax.set(ylim=(90,100),ylabel="完整率（%）",title="核心数据14元素完整率");fig.tight_layout();fig.savefig(out/"figures/03_核心元素完整率.png");plt.close(fig)
 print(json.dumps(stat,ensure_ascii=False,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--global-xlsx",required=True);p.add_argument("--training-csv",required=True);p.add_argument("--output",default="全球数据重新审计结果");p.add_argument("--training-group",default="site");p.add_argument("--min-features",type=int,default=12);p.add_argument("--domain-quantile",type=float,default=.95);main(p.parse_args())
