from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from io import BytesIO
from docx import Document
import pandas as pd

ROOT=Path(__file__).parent
OUT=ROOT/"provenance_final"
OUT.mkdir(exist_ok=True)
df=pd.read_csv("gangdese_input/06_碎屑锆石双模型预测.csv",low_memory=False)
zipfile=ZipFile(next(ROOT.glob("*.zip")))
new=pd.read_csv(BytesIO(zipfile.read("02_新增627条逐ID.csv")),low_memory=False)
mapping=new.set_index("STRICT_FINGERPRINT")["GLOBAL_RECORD_ID"]
mask=df.DATA_ORIGIN.eq("GEOROC_V10_additional")
df.loc[mask,"GLOBAL_RECORD_ID"]=df.loc[mask,"STRICT_FINGERPRINT"].map(mapping)
assert len(df)==10879 and df.GLOBAL_RECORD_ID.notna().all()
assert df.GLOBAL_RECORD_ID.nunique()==10879

df["ORIGINAL_REFERENCE_LABEL"]=df.Reference.fillna("").astype(str).str.strip()
df["SOURCE_DATABASE"]=df.DATA_ORIGIN.map({
    "Wang2025_compilation":"Wang et al., DOI:10.1130/B38079.1, supplementary compilation",
    "GEOROC_V10_additional":"GEOROC V10 database"})
assert df.SOURCE_DATABASE.notna().all()
df["PUBLICATION_OR_DATABASE"]=df.ORIGINAL_REFERENCE_LABEL.where(
    df.ORIGINAL_REFERENCE_LABEL.ne(""),df.SOURCE_DATABASE)
df["PUBLICATION_STATUS"]=df.ORIGINAL_REFERENCE_LABEL.ne("").map(
    {True:"source publication label supplied in parent compilation",
     False:"publication label unavailable; database attribution"})
df["DETRITAL_IDENTITY_STATUS"]="not explicitly assigned per grain in Wang compilation"
df.loc[mask,"DETRITAL_IDENTITY_STATUS"]="sedimentary host lithology documented in GEOROC; detrital context"
df["HOST_ROCK_EVIDENCE"]=df["ROCK NAME"].fillna("")
df["DATABASE_RECORD_IDENTIFIER"]=df["ID"].fillna("").astype(str)
cols=["GLOBAL_RECORD_ID","SOURCE_DATABASE","ORIGINAL_REFERENCE_LABEL",
      "PUBLICATION_OR_DATABASE","PUBLICATION_STATUS","SOURCE_ID",
      "DETRITAL_IDENTITY_STATUS","HOST_ROCK_EVIDENCE","DATABASE_RECORD_IDENTIFIER",
      "Sample","Analysis","SPOT","Region","Sub_region","LOCATION","AGE_MA_CLEAN",
      "DATA_ORIGIN","STRICT_FINGERPRINT"]
df[cols].to_csv(OUT/"逐颗锆石来源与身份状态_10879条.csv",index=False,encoding="utf-8-sig")
refs=df.groupby(["SOURCE_DATABASE","ORIGINAL_REFERENCE_LABEL","PUBLICATION_STATUS"],dropna=False).agg(
    GRAINS=("GLOBAL_RECORD_ID","size"),SOURCE_IDS=("SOURCE_ID","nunique")).reset_index()
refs.to_csv(OUT/"原始来源标签登记表.csv",index=False,encoding="utf-8-sig")
summary=pd.DataFrame([{"dataset":name,"grains":len(g),"reference_labels":g.ORIGINAL_REFERENCE_LABEL.replace("",pd.NA).nunique(),
                       "empty_reference_labels":int(g.ORIGINAL_REFERENCE_LABEL.eq("").sum()),
                       "source_ids":g.SOURCE_ID.nunique(),"unique_record_ids":g.GLOBAL_RECORD_ID.nunique()}
                      for name,g in df.groupby("SOURCE_DATABASE")])
summary.to_csv(OUT/"溯源完整性汇总.csv",index=False,encoding="utf-8-sig")
for lang,original in [("cn","全球锆石氧逸度演化_中文修订稿.docx"),
                      ("en","Global_Zircon_Redox_Revised_Manuscript_EN.docx")]:
    d=Document(BytesIO(zipfile.read(original)))
    # Preserve the four-task text if available in the more recent revised document.
    candidates=list((ROOT/"completed").glob("*四项完善稿.docx" if lang=="cn" else "*Four_Tasks_Revised.docx"))
    if candidates: d=Document(candidates[0])
    if lang=="cn":
        d.paragraphs[16].text=("数据筛选与逐颗溯源。Wang汇编母集20,347条、旧版候选10,844条，"
          "移出588条组成域外记录和4条潜在重复后保留10,252条，另增627条GEOROC记录，合计10,879条。"
          "每颗均有唯一GLOBAL_RECORD_ID、SOURCE_ID与原库给出的Reference标签：Wang子集39种标签、"
          "GEOROC新增9种标签，合计48种；来源单元55个。全部10,879条都有原始文献标签，"
          "因此仅在标签真正缺失时才回退标记数据库来源。数据库/文献归属与碎屑身份分列："
          "GEOROC新增有沉积岩性字段，旧汇编缺统一逐颗碎屑/原位身份字段。P、La、Th/U的"
          "逐项标记按测得值重建；未测变量不等于实测通过。逐颗溯源、身份状态及字段可用性在补充表公开。")
        d.paragraphs[65 if len(d.paragraphs)==66 else 67].text=("数据与代码声明。冻结逐颗数据、"
          "原始来源标签和数据库归属、逐项QC审计、来源与地区统计及分析脚本随稿整理。"
          "正式公开仓库的永久链接与作者信息由研究团队在投稿前确定。")
        target=OUT/"全球锆石氧逸度演化_中文_溯源修订稿.docx"
    else:
        d.paragraphs[16].text=("Selection and grain-level provenance. Of 20,347 source records, "
          "10,844 were older candidates; 588 out-of-domain records and four potential duplicates were "
          "removed, retaining 10,252. GEOROC adds 627 for 10,879 total. Every grain has a unique "
          "GLOBAL_RECORD_ID, SOURCE_ID and a Reference label supplied by the parent compilation: "
          "39 labels in the Wang subset and nine in GEOROC, 48 in total, with 55 source units. "
          "No grain lacks a source publication label; database attribution is the fallback only when "
          "that label is absent. Provenance is distinct from detrital identity: sedimentary lithology "
          "is recorded for the GEOROC additions, whereas the older compilation has no uniform "
          "grain-level detrital versus in-situ field. Rebuilt P, La and Th/U QC flags and field "
          "availability accompany the per-grain provenance table.")
        d.paragraphs[65 if len(d.paragraphs)==66 else 67].text=("Data and code availability. "
          "The frozen grain table, parent publication labels, database attribution, QC audit, "
          "source and geographic summaries and analysis scripts are assembled. A permanent public "
          "repository and author metadata will be finalized by the research team before submission.")
        target=OUT/"Global_Zircon_Redox_Provenance_Revised.docx"
    d.save(target)
print(summary.to_string(index=False))
