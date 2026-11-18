#!/usr/bin/env python3
"""建立最终逐颗锆石ID、来源/文献注册表并冻结数据。不会训练模型。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs/最终10879条_旧冻结双模型预测.csv"
OUT = ROOT / "frozen"
FEATURES = ["Ce", "Nd", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "U", "Hf", "Ti"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def stable_row_hash(row: pd.Series) -> str:
    fields = [
        "DATA_ORIGIN", "SOURCE_ID", "Reference", "AGE_MA_CLEAN", "Sample", "Analysis",
        "CITATION", "SAMPLE NAME", "SPOT", "LOCATION", "Region"
    ] + FEATURES
    text = "|".join(clean_text(row.get(c, "")) for c in fields)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(INPUT, low_memory=False)

    # 统一年龄字段。
    clean_age = pd.to_numeric(d.get("AGE_MA_CLEAN"), errors="coerce")
    selected_age = pd.to_numeric(d.get("Selected_age"), errors="coerce")
    d["AGE_MA_CLEAN"] = clean_age.combine_first(selected_age)

    required = ["SOURCE_ID", "Reference", "DATA_ORIGIN", "AGE_MA_CLEAN",
                "DELTA_FMQ_MLP", "DELTA_FMQ_EXTRATREES", "DELTA_FMQ_ENSEMBLE"] + FEATURES
    missing_columns = [c for c in required if c not in d.columns]
    if missing_columns:
        raise RuntimeError(f"缺少必要字段：{missing_columns}")
    if len(d) != 10879:
        raise RuntimeError(f"预期10879条，实际{len(d)}条")
    for c in ["SOURCE_ID", "Reference", "DATA_ORIGIN", "AGE_MA_CLEAN"]:
        if d[c].isna().any():
            raise RuntimeError(f"字段{c}仍有空值")

    # 建立稳定的文献ID。REFERENCE_ID只依赖Reference标准化文本排序。
    refs = sorted(d["Reference"].map(clean_text).unique())
    ref_map = {r: f"REF_{i:03d}" for i, r in enumerate(refs, 1)}
    d["REFERENCE_ID"] = d["Reference"].map(clean_text).map(ref_map)

    # 新增GEOROC颗粒缺少GLOBAL_RECORD_ID：用内容哈希建立可复现ID。
    row_hash = d.apply(stable_row_hash, axis=1)
    d["ROW_CONTENT_SHA256"] = row_hash
    old_id = d.get("GLOBAL_RECORD_ID", pd.Series(index=d.index, dtype="object")).astype("string")
    missing_id = old_id.isna() | old_id.str.strip().eq("")
    # 极少数完全相同行使用确定性的出现序号避免ID碰撞。
    occurrence = row_hash.groupby(row_hash).cumcount().add(1)
    new_id = "GEOROCV10_" + row_hash.str[:16].str.upper() + "_" + occurrence.astype(str).str.zfill(2)
    d["GLOBAL_RECORD_ID"] = old_id.mask(missing_id, new_id)

    if d["GLOBAL_RECORD_ID"].isna().any() or d["GLOBAL_RECORD_ID"].duplicated().any():
        dup = d.loc[d["GLOBAL_RECORD_ID"].duplicated(False), "GLOBAL_RECORD_ID"].head().tolist()
        raise RuntimeError(f"GLOBAL_RECORD_ID未能保持唯一：{dup}")

    d = d.sort_values(["AGE_MA_CLEAN", "GLOBAL_RECORD_ID"], kind="stable").reset_index(drop=True)
    d.insert(0, "FROZEN_ROW_INDEX", np.arange(1, len(d) + 1))

    frozen_data = OUT / "01_论文最终逐颗锆石数据_10879条_FROZEN.csv"
    d.to_csv(frozen_data, index=False, encoding="utf-8-sig")

    provenance_cols = [
        "FROZEN_ROW_INDEX", "GLOBAL_RECORD_ID", "ROW_CONTENT_SHA256", "SOURCE_ID",
        "REFERENCE_ID", "Reference", "DATA_ORIGIN", "AGE_MA_CLEAN", "Region", "Sub_region",
        "Sample", "Analysis", "CITATION", "CITATION_KEY", "SAMPLE NAME", "SPOT", "LOCATION",
        "DATA_GRADE", "N_FEATURES_MEASURED", "MISSING_FEATURES", "DOMAIN_NN_DISTANCE",
        "DOMAIN_THRESHOLD", "DOMAIN_PASS", "STRICT_FINGERPRINT"
    ]
    provenance_cols = [c for c in provenance_cols if c in d.columns]
    provenance = OUT / "02_逐颗锆石溯源表_10879条.csv"
    d[provenance_cols].to_csv(provenance, index=False, encoding="utf-8-sig")

    source_registry = (
        d.groupby(["SOURCE_ID", "REFERENCE_ID", "Reference", "DATA_ORIGIN"], dropna=False)
        .agg(**{
            "锆石数": ("GLOBAL_RECORD_ID", "size"),
            "年龄最小_Ma": ("AGE_MA_CLEAN", "min"),
            "年龄最大_Ma": ("AGE_MA_CLEAN", "max"),
            "14元素完整数": ("N_FEATURES_MEASURED", lambda x: int((x == 14).sum())),
            "区域": ("Region", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
        }).reset_index().sort_values("SOURCE_ID")
    )
    source_path = OUT / "03_来源单元注册表_55个.csv"
    source_registry.to_csv(source_path, index=False, encoding="utf-8-sig")

    parent = np.where(
        d["DATA_ORIGIN"].eq("Wang2025_compilation"),
        "Wang et al. (2025) Supplementary Table S2, DOI:10.1130/B38079.1",
        "GEOROC V10 zircon compilation; original citation retained in Reference/CITATION",
    )
    temp = d.assign(PARENT_DATASET_CITATION=parent)
    ref_registry = (
        temp.groupby(["REFERENCE_ID", "Reference"], dropna=False)
        .agg(**{
            "锆石数": ("GLOBAL_RECORD_ID", "size"),
            "来源单元数": ("SOURCE_ID", "nunique"),
            "年龄最小_Ma": ("AGE_MA_CLEAN", "min"),
            "年龄最大_Ma": ("AGE_MA_CLEAN", "max"),
            "数据入口": ("DATA_ORIGIN", lambda x: " | ".join(sorted(set(x.astype(str))))),
            "上级数据集引用": ("PARENT_DATASET_CITATION", lambda x: " | ".join(sorted(set(x.astype(str))))),
        }).reset_index().sort_values("REFERENCE_ID")
    )
    ref_path = OUT / "04_文献注册表_48篇.csv"
    ref_registry.to_csv(ref_path, index=False, encoding="utf-8-sig")

    report = OUT / "06_冻结数据核验报告.md"
    report.write_text(
        f"""# 最终论文数据冻结核验报告

- 冻结版本：`ZIRCON_FMQ_PAPER_V1`
- 总记录数：{len(d):,}
- 唯一颗粒记录ID：{d['GLOBAL_RECORD_ID'].nunique():,}
- 新增GEOROC记录ID：{int(d['GLOBAL_RECORD_ID'].str.startswith('GEOROCV10_').sum()):,}
- 来源单元：{d['SOURCE_ID'].nunique():,}
- 文献ID：{d['REFERENCE_ID'].nunique():,}
- 年龄缺失：{int(d['AGE_MA_CLEAN'].isna().sum())}
- MLP预测缺失：{int(d['DELTA_FMQ_MLP'].isna().sum())}
- ExtraTrees预测缺失：{int(d['DELTA_FMQ_EXTRATREES'].isna().sum())}
- 双模型预测缺失：{int(d['DELTA_FMQ_ENSEMBLE'].isna().sum())}
- 14元素完整记录：{int((d['N_FEATURES_MEASURED'] == 14).sum()):,}

`GLOBAL_RECORD_ID`、`SOURCE_ID`和`REFERENCE_ID`均已逐颗建立。后续所有分析应只读取冻结文件，不再修改原始行或ID。
""",
        encoding="utf-8",
    )

    tracked = [frozen_data, provenance, source_path, ref_path, report]
    manifest = {
        "freeze_id": "ZIRCON_FMQ_PAPER_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_status": "OLD_MODELS_FROZEN",
        "training_performed": False,
        "n_rows": int(len(d)),
        "n_sources": int(d["SOURCE_ID"].nunique()),
        "n_references": int(d["REFERENCE_ID"].nunique()),
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in tracked
        ],
    }
    manifest_path = OUT / "05_冻结清单_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    print(f"冻结完成：{OUT}")


if __name__ == "__main__":
    main()
