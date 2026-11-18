from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "completed"
OUT.mkdir(exist_ok=True)
GLOBAL = Path("gangdese_input/06_碎屑锆石双模型预测.csv")
df = pd.read_csv(GLOBAL, low_memory=False)
z = ZipFile(next(ROOT.glob("*.zip")))
new = pd.read_csv(BytesIO(z.read("02_新增627条逐ID.csv")), low_memory=False)

def continent(row):
    region = str(row.Region).lower()
    if row.DATA_ORIGIN == "Wang2025_compilation":
        if region in ("asia", "aisa"): return "Asia"
        if region in ("australia", "zealandia"): return "Oceania"
        if region == "iceland" or region == "europe": return "Europe"
        if region in ("north america", "south america", "africa"): return region.title()
        return "Unassigned"
    if any(k in region for k in ("china", "tibet", "mongolia", "cambodia", "yunnan")): return "Asia"
    if any(k in region for k in ("alps", "sardinia", "corsica", "baltic")): return "Europe"
    if "andean" in region or "chile" in region: return "South America"
    return "Unassigned"

df["CONTINENT_CLASS"] = df.apply(continent, axis=1)
df["AGE_BIN_100"] = (np.floor(df.AGE_MA_CLEAN / 100) * 100).astype(int).clip(upper=3900)
df["GRAIN_ROW"] = np.arange(len(df))
counts = df.groupby(["AGE_BIN_100","CONTINENT_CLASS"]).size().unstack(fill_value=0)
counts = counts.reindex(columns=["Africa","Asia","Europe","North America","South America","Oceania","Unassigned"],fill_value=0)
counts["TOTAL_GRAINS"] = counts.sum(axis=1)
counts["INDEPENDENT_SOURCES"] = df.groupby("AGE_BIN_100").SOURCE_ID.nunique()
counts["MAX_SOURCE_SHARE"] = df.groupby("AGE_BIN_100").SOURCE_ID.apply(lambda s:s.value_counts(normalize=True).iloc[0])
counts.to_csv(OUT/"01_每100Ma大陆构成与来源.csv",encoding="utf-8-sig")

stage_defs = {"early_target":(3000,3300),"early_reference":(3500,4001),
              "late_archean":(2500,3000),"mesoproterozoic":(800,1800)}
stages=[]
for label,(a,b) in stage_defs.items():
    q=df[(df.AGE_MA_CLEAN>=a)&(df.AGE_MA_CLEAN<b)]
    for name,n in q.CONTINENT_CLASS.value_counts().items():
        stages.append({"STAGE":label,"AGE_LOWER_MA":a,"AGE_UPPER_MA":b,
                       "CONTINENT_CLASS":name,"GRAINS":n,"TOTAL_GRAINS":len(q),
                       "SOURCE_COUNT":q.SOURCE_ID.nunique(),
                       "SOURCE_COUNT_IN_CLASS":q[q.CONTINENT_CLASS==name].SOURCE_ID.nunique(),
                       "MAX_SOURCE_SHARE":q.SOURCE_ID.value_counts(normalize=True).iloc[0]})
pd.DataFrame(stages).to_csv(OUT/"02_核心时段大陆构成.csv",index=False,encoding="utf-8-sig")
loo=[]
for removed in ["NONE","Africa","Asia","Europe","North America","Oceania"]:
    q=df if removed=="NONE" else df[df.CONTINENT_CLASS!=removed]
    def source_median(lo,hi):
        s=q[(q.AGE_MA_CLEAN>=lo)&(q.AGE_MA_CLEAN<hi)]
        return s.groupby("SOURCE_ID").DELTA_FMQ_ENSEMBLE.median().median(),s.SOURCE_ID.nunique()
    target,nt=source_median(3000,3300)
    baseline,nb=source_median(3500,4001)
    loo.append({"OMITTED_CONTINENT":removed,"EARLY_RISE_DELTA_FMQ":target-baseline,
                "TARGET_SOURCES":nt,"REFERENCE_SOURCES":nb})
pd.DataFrame(loo).to_csv(OUT/"02b_早期上升逐大陆剔除.csv",index=False,encoding="utf-8-sig")
df[["GRAIN_ROW","GLOBAL_RECORD_ID","SOURCE_ID","AGE_MA_CLEAN","CONTINENT_CLASS",
    "Region","Sub_region","Reference","DATA_ORIGIN"]].to_csv(
    OUT/"03_逐颗地区分类对账.csv",index=False,encoding="utf-8-sig")

qc = new[["FROZEN_ROW_INDEX","GLOBAL_RECORD_ID","SOURCE_ID","P","La","Th","U","TH_U",
          "ROCK NAME","DETRITAL_ROCK_MATCH","DETRITAL_CONFIDENCE",
          "QC_P_GT750","QC_LA_GT1","QC_TH_U_LT0_1","ZIRCON_QC_PASS"]].copy()
qc["QC_P_GT750_REBUILT"]=pd.to_numeric(qc.P,errors="coerce").gt(750)
qc["QC_LA_GT1_REBUILT"]=pd.to_numeric(qc.La,errors="coerce").gt(1)
ratio=pd.to_numeric(qc.TH_U,errors="coerce").combine_first(
      pd.to_numeric(qc.Th,errors="coerce")/pd.to_numeric(qc.U,errors="coerce"))
qc["QC_TH_U_LT0_1_REBUILT"]=ratio.lt(.1)
qc["TH_U_USED"]=ratio
qc["QC_REBUILT_PASS"]=~(qc.QC_P_GT750_REBUILT|qc.QC_LA_GT1_REBUILT|qc.QC_TH_U_LT0_1_REBUILT)
qc["QC_RATIO_MEASURED"]=ratio.notna()
qc["IDENTITY_STATUS"]="GEOROC sedimentary lithology; grain provenance unverified"
qc.to_csv(OUT/"04_新增627条逐项QC与身份.csv",index=False,encoding="utf-8-sig")
assert len(qc)==627 and qc.GLOBAL_RECORD_ID.notna().all()
old=df[df.DATA_ORIGIN=="Wang2025_compilation"]
status=pd.DataFrame([
    {"SUBSET":"GEOROC additional","GRAINS":len(qc),
     "IDENTITY":"Sedimentary lithology recorded; original detrital context requires publication review",
     "GRAIN_LEVEL_ID":int(qc.GLOBAL_RECORD_ID.notna().sum())},
    {"SUBSET":"Wang compilation retained","GRAINS":len(old),
     "IDENTITY":"No per-grain detrital/magmatic identity field in frozen table",
     "GRAIN_LEVEL_ID":int(old.GLOBAL_RECORD_ID.notna().sum())}])
status.to_csv(OUT/"05_身份可核查程度.csv",index=False,encoding="utf-8-sig")

fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
bottom=np.zeros(len(counts))
for key,color in zip(["Africa","Asia","Europe","North America","South America","Oceania","Unassigned"],
                     ["#bf8d57","#6f91ad","#788c68","#b66968","#c69b51","#9482ae","#aaaaaa"]):
    v=counts[key].to_numpy()
    ax.bar(counts.index+50,v,bottom=bottom,width=90,label=key,color=color)
    bottom+=v
ax.set(xlabel="Zircon age (Ma; older to right)",ylabel="Grains per 100 Myr",
       title="Continental composition of the frozen 10,879-grain compilation")
ax.legend(ncol=4,fontsize=8,frameon=False)
fig.savefig(OUT/"FigureS_100Ma大陆构成.png",dpi=220);plt.close(fig)
print("total",len(df),"unknown",sum(df.CONTINENT_CLASS=="Unassigned"))
print("QC",qc[["QC_P_GT750_REBUILT","QC_LA_GT1_REBUILT","QC_TH_U_LT0_1_REBUILT","QC_REBUILT_PASS"]].sum().to_dict())
print(pd.DataFrame(stages).to_string(index=False))
