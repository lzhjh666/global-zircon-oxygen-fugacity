#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan all GEOROC zircon rows for additional detrital, in-domain candidates."""
from __future__ import annotations
import argparse,csv,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

F=["Ce","Nd","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","U","Hf","Ti"]
CN={"Ce":"Ce（铈）","Nd":"Nd（钕）","Eu":"Eu（铕）","Gd":"Gd（钆）","Tb":"Tb（铽）","Dy":"Dy（镝）","Ho":"Ho（钬）","Er":"Er（铒）","Tm":"Tm（铥）","Yb":"Yb（镱）","Lu":"Lu（镥）","U":"U（铀）","Hf":"Hf（铪）","Ti":"Ti（钛）"}
AGES=["AGE(MA)","AGE(KA)","AGE_PB207_PB206(MA)","AGE_PB207_PB206(KA)","AGE_PB207_U235(MA)","AGE_PB207_U235(KA)","AGE_PB206_U238(MA)","AGE_PB206_U238(KA)","AGE_PB208_TH232(MA)","AGE_PB208_TH232(KA)"]
META=["CITATION","SAMPLE NAME","ROCK NAME","ROCK TEXTURE","LOCATION","LOCATION COMMENT","TECTONIC SETTING","MINERAL","SPOT","CRYSTAL","PRIMARY/SECONDARY","LATITUDE (MIN.)","LATITUDE (MAX.)","LONGITUDE (MIN.)","LONGITUDE (MAX.)"]
DET=re.compile(r"detrit|sediment|sandstone|quartzite|conglomerate|greywacke|graywacke|metasediment|shale|mudstone|siltstone|beach|river|stream|alluvial|placer",re.I)

def num(s):
 s=s.astype("string").str.strip().str.replace(",","",regex=False);below=s.str.match(r"^<\s*[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?$",na=False);v=pd.to_numeric(s.str.replace(r"^[<>~=≤≥]\s*","",regex=True),errors="coerce");v.loc[below]=v.loc[below]/2;return v.mask(~np.isfinite(v)|(v<=0))
def elem(d,e):
 o=pd.Series(np.nan,index=d.index,dtype=float)
 for c,f in [(e.upper()+"(PPM)",1),(e.upper()+"(WT%)",1e4),(e.upper()+"(PPB)",1e-3)]:
  if c in d:
   v=num(d[c])*f;m=o.isna()&v.notna();o.loc[m]=v.loc[m]
 return o
def age(d):
 vals={c:num(d[c])/(1000 if c.endswith("(KA)") else 1) for c in AGES if c in d}
 o=pd.Series(np.nan,index=d.index,dtype=float)
 for c in ["AGE(MA)","AGE(KA)"]:
  if c in vals:o=o.combine_first(vals[c])
 u=vals.get("AGE_PB206_U238(MA)",vals.get("AGE_PB206_U238(KA)"));pb=vals.get("AGE_PB207_PB206(MA)",vals.get("AGE_PB207_PB206(KA)"))
 if u is not None:o=o.mask(o.isna()&(u<=1000),u)
 if pb is not None:o=o.combine_first(pb)
 if u is not None:o=o.combine_first(u)
 for c in ["AGE_PB207_U235(MA)","AGE_PB207_U235(KA)","AGE_PB208_TH232(MA)","AGE_PB208_TH232(KA)"]:
  if c in vals:o=o.combine_first(vals[c])
 return o
def cite_key(s):
 s=re.sub(r"^\s*\[\d+\]\s*","",str(s).lower());year=re.search(r"(?:19|20)\d{2}",s);name=re.search(r"[a-z]+",s);return (name.group(0) if name else "unknown")+"_"+(year.group(0) if year else "noyear")
def append(d,p,first):d.to_csv(p,index=False,mode="w" if first else "a",header=first,encoding="utf-8-sig",quoting=csv.QUOTE_MINIMAL)

def main(a):
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 train=pd.read_csv(a.training_csv,encoding="utf-8-sig");x=train[[CN[f] for f in F]].apply(pd.to_numeric,errors="coerce");x.columns=F;x=x.where(x>0);imp=SimpleImputer(strategy="median").fit(np.log10(x));xi=imp.transform(np.log10(x));sc=RobustScaler(quantile_range=(10,90)).fit(xi);z=sc.transform(xi);grp=train[a.training_group].astype(str).to_numpy();td=np.asarray([np.sqrt(((z[grp!=grp[i]]-z[i])**2).sum(1)).min() for i in range(len(z))]);threshold=float(np.quantile(td,a.domain_quantile));nn=NearestNeighbors(n_neighbors=1).fit(z)
 old=pd.read_csv(a.existing_core,low_memory=False);old_keys={cite_key(s) for s in old.Reference.dropna().unique()}
 # Strict fingerprints are used only to mark, never as the sole publication-level decision.
 def fp_rows(d,agecol="AGE_MA_CLEAN"):
  q=pd.DataFrame({"age":pd.to_numeric(d[agecol],errors="coerce").round(1)})
  for f in F:q[f]=np.log10(pd.to_numeric(d[f],errors="coerce")).round(3)
  return q.astype(str).agg("|".join,axis=1)
 old_fp=set(fp_rows(old,"Selected_age"))
 paths={k:out/v for k,v in {"a":"01_建议新增_非重叠域内.csv","b":"02_同文献可能重叠_待人工合并.csv","c":"03_域外或身份较弱_仅审计.csv"}.items()};first={k:True for k in paths};[p.unlink(missing_ok=True) for p in paths.values()]
 stats={"raw_rows":0,"zircon_rows":0,"age_0_4000":0,"detrital_keyword":0,"at_least_12_features":0,"zircon_qc_pass":0,"in_domain":0,"recommended_new":0,"same_publication_review":0,"warning":0,"strict_fingerprint_overlap":0}
 with zipfile.ZipFile(a.georoc_zip) as zz:
  names=[n for n in zz.namelist() if n.lower().endswith(".csv")];assert len(names)==1
  with zz.open(names[0]) as fh:
   header=pd.read_csv(fh,nrows=0,encoding="latin1").columns.tolist()
  use=[c for c in META+AGES if c in header]
  for e in F+["P","La","Th"]:
   use += [c for c in [e.upper()+"(PPM)",e.upper()+"(WT%)",e.upper()+"(PPB)"] if c in header]
  use=list(dict.fromkeys(use))
  with zz.open(names[0]) as fh:
   for block,d in enumerate(pd.read_csv(fh,usecols=use,dtype=str,chunksize=a.chunksize,encoding="latin1",low_memory=False,on_bad_lines="warn"),1):
    stats["raw_rows"]+=len(d);mineral=d.get("MINERAL",pd.Series("",index=d.index)).fillna("").str.upper();d=d[mineral.str.contains("ZIRCON",na=False)].copy();stats["zircon_rows"]+=len(d)
    if d.empty:continue
    r=pd.DataFrame(index=d.index);r["AGE_MA_CLEAN"]=age(d);ok=r.AGE_MA_CLEAN.between(0,4000);d=d.loc[ok];r=r.loc[ok];stats["age_0_4000"]+=len(r)
    if r.empty:continue
    for c in META:
     if c in d:r[c]=d[c]
    rock=d[[c for c in ["ROCK NAME","ROCK TEXTURE"] if c in d]].fillna("").agg(" | ".join,axis=1)
    context=d[[c for c in ["LOCATION","LOCATION COMMENT","SAMPLE NAME"] if c in d]].fillna("").agg(" | ".join,axis=1)
    r["DETRITAL_ROCK_MATCH"]=rock.str.contains(DET,na=False);r["DETRITAL_CONTEXT_MATCH"]=context.str.contains(DET,na=False);r["DETRITAL_CONFIDENCE"]=np.where(r.DETRITAL_ROCK_MATCH,"high",np.where(r.DETRITAL_CONTEXT_MATCH,"medium","none"));det=r.DETRITAL_CONFIDENCE.ne("none");stats["detrital_keyword"]+=int(det.sum());d=d.loc[det];r=r.loc[det]
    if r.empty:continue
    for f in F:r[f]=elem(d,f)
    r["P"]=elem(d,"P");r["La"]=elem(d,"La");r["Th"]=elem(d,"Th");r["N_FEATURES_MEASURED"]=r[F].notna().sum(1);r["MISSING_FEATURES"]=r[F].isna().apply(lambda x:";".join(x.index[x]),axis=1);enough=r.N_FEATURES_MEASURED>=a.min_features;stats["at_least_12_features"]+=int(enough.sum());r=r[enough]
    if r.empty:continue
    r["TH_U"]=r.Th/r.U;r["ZIRCON_QC_PASS"]=~((r.P>750).fillna(False)|(r.La>1).fillna(False)|(r.TH_U<.1).fillna(False));stats["zircon_qc_pass"]+=int(r.ZIRCON_QC_PASS.sum());r=r[r.ZIRCON_QC_PASS]
    if r.empty:continue
    zzg=sc.transform(imp.transform(np.log10(r[F])));r["DOMAIN_NN_DISTANCE"]=nn.kneighbors(zzg,return_distance=True)[0][:,0];r["DOMAIN_THRESHOLD"]=threshold;r["DOMAIN_PASS"]=r.DOMAIN_NN_DISTANCE<=threshold;stats["in_domain"]+=int(r.DOMAIN_PASS.sum());r["CITATION_KEY"]=r.CITATION.map(cite_key);r["PUBLICATION_OVERLAP_WITH_WANG"]=r.CITATION_KEY.isin(old_keys);r["STRICT_FINGERPRINT"]=fp_rows(r);r["STRICT_FINGERPRINT_OVERLAP"]=r.STRICT_FINGERPRINT.isin(old_fp);stats["strict_fingerprint_overlap"]+=int(r.STRICT_FINGERPRINT_OVERLAP.sum())
    good=r.DOMAIN_PASS & ~r.STRICT_FINGERPRINT_OVERLAP
    aa=r[good & ~r.PUBLICATION_OVERLAP_WITH_WANG].copy();bb=r[good & r.PUBLICATION_OVERLAP_WITH_WANG].copy();cc=r[~good].copy()
    stats["recommended_new"]+=len(aa);stats["same_publication_review"]+=len(bb);stats["warning"]+=len(cc)
    for k,q in [("a",aa),("b",bb),("c",cc)]:
     if len(q):append(q,paths[k],first[k]);first[k]=False
    print(f"block {block}: scanned {stats['raw_rows']:,}, recommended {stats['recommended_new']:,}",flush=True)
 (out/"00_扫描统计.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--georoc-zip",required=True);p.add_argument("--training-csv",required=True);p.add_argument("--existing-core",required=True);p.add_argument("--output",default="GEOROC新增候选扫描");p.add_argument("--training-group",default="site");p.add_argument("--min-features",type=int,default=12);p.add_argument("--domain-quantile",type=float,default=.95);p.add_argument("--chunksize",type=int,default=30000);main(p.parse_args())
