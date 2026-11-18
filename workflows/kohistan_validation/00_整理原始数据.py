# -*- coding: utf-8 -*-
"""从论文补充表整理Kohistan 2026外部验证数据；不加载、不运行任何模型。"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "数据来源" / "Kohistan_2026_Supplementary_Data_2.xlsx"
OUT = ROOT / "外部验证数据"

FEATURE_MAP = {
    "Ce": "Ce（铈）", "Nd": "Nd（钕）", "Sm": "Sm（钐）", "Eu": "Eu（铕）",
    "Gd": "Gd（钆）", "Tb": "Tb（铽）", "Dy": "Dy（镝）", "Ho": "Ho（钬）",
    "Er": "Er（铒）", "Tm": "Tm（铥）", "Yb": "Yb（镱）", "Lu": "Lu（镥）",
    "Th": "Th（钍）", "U": "U（铀）", "Hf": "Hf（铪）", "Ti": "Ti（钛）",
    "Y": "Y（钇）",
}

# 论文Table 2的岩体平均值：Hbl logfO2由Ridolfi (2021)角闪石方法独立获得。
# T采用Table 2的平均Amp T（Ridolfi 2021），P采用平均Amp P（Schmidt 1992）。
GROUP_LABELS = {
    "Matiltan Granitoid": (-16.0, 783.0, 3.6),
    "Gabral Quartz Diorite": (-16.0, 776.5, 3.0),
    "Deshai Quartz Diorite": (-14.0, 793.7, 4.5),
    "Diwangar Granitoid": (-13.0, 787.3, 3.7),
    "Shahibagh Granitoid": (-13.0, 786.2, 4.1),
}


def fmq_frost_1991(temperature_c: float, pressure_kbar: float) -> float:
    """Frost (1991) beta-quartz FMQ，T为K、P为bar。"""
    temperature_k = temperature_c + 273.15
    pressure_bar = pressure_kbar * 1000.0
    return -25096.3 / temperature_k + 8.735 + 0.11 * (pressure_bar - 1.0) / temperature_k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(RAW, sheet_name="S1.4", header=1)
    raw = raw.loc[raw["Rock type"].isin(GROUP_LABELS)].copy()
    raw.insert(0, "外部记录编号", [f"KOH-{i:03d}" for i in range(1, len(raw) + 1)])
    raw = raw.rename(columns=FEATURE_MAP)

    feature_cols = list(FEATURE_MAP.values())
    for column in feature_cols:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[feature_cols].isna().any().any():
        missing = raw[feature_cols].isna().sum()
        raise ValueError("17元素存在缺失，停止整理：\n" + missing[missing > 0].to_string())

    blind = raw[["外部记录编号", "Samples", "Rock type", "Spot", "Age (Ma)", *feature_cols]].rename(
        columns={"Samples": "样品编号", "Rock type": "岩体", "Spot": "分析点", "Age (Ma)": "年龄_Ma"}
    )
    blind.to_csv(OUT / "Kohistan_盲测特征_113颗_不含标签.csv", index=False, encoding="utf-8-sig")

    rows = []
    for rock, (logfo2, temp_c, pressure_kbar) in GROUP_LABELS.items():
        fmq = fmq_frost_1991(temp_c, pressure_kbar)
        rows.append({
            "岩体": rock,
            "锆石数": int((blind["岩体"] == rock).sum()),
            "独立角闪石约束_logfO2": logfo2,
            "FMQ换算温度_C": temp_c,
            "FMQ换算压力_kbar": pressure_kbar,
            "FMQ缓冲_logfO2": fmq,
            "独立约束_ΔFMQ": logfo2 - fmq,
            "标签来源": "Ahmad et al. (2026) Table 2；Ridolfi (2021)角闪石氧逸度",
            "标签注意": "论文组均值logfO2按整数报告；按Frost (1991)在组平均T-P换算ΔFMQ",
        })
    labels = pd.DataFrame(rows)
    labels.to_csv(OUT / "Kohistan_封存组级标签_评分时读取.csv", index=False, encoding="utf-8-sig")

    audit = pd.DataFrame({
        "检查项": ["可评分锆石数", "独立岩体数", "17元素缺失单元格", "排除岩体"],
        "结果": [len(blind), len(labels), int(blind[feature_cols].isna().sum().sum()),
                 "Bhankhwarh、Jut Banda（论文Table 2没有角闪石fO2）"],
    })
    audit.to_csv(OUT / "数据整理审计.csv", index=False, encoding="utf-8-sig")
    print(audit.to_string(index=False))
    print(labels[["岩体", "锆石数", "独立约束_ΔFMQ"]].to_string(index=False))


if __name__ == "__main__":
    main()
