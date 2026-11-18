#!/usr/bin/env python3
"""Tang et al. (2021) four-river zircon data; frozen-model inference only.

Run: python 冈底斯_冻结模型预测.py
No model fitting, retraining, cross-validation, or external-data tuning.
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "G47745_SuppData.xlsx"  # Download the publisher supplement; see docs/EXTERNAL_DATA.md
MODELS = ROOT / "models"
OUT = HERE / "results"
FEATURES = ["Ce", "Nd", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu", "U", "Hf", "Ti"]
# Fixed before any predictions; age is zircon crystallization age (Ma).
STAGES = {"Paleocene_Eocene": (40., 65.), "Miocene": (10., 25.)}


def as_positive(value):
    """Keep genuine numeric observations; 'below detection' is missing."""
    v = pd.to_numeric(value, errors="coerce")
    return float(v) if pd.notna(v) and np.isfinite(v) and v > 0 else np.nan


def load_and_filter():
    book = openpyxl.load_workbook(DATA, read_only=True, data_only=True)
    records = []
    for sheet in book:
        if sheet.title not in {"River 1", "River 2", "River 3", "River 4"}:
            continue
        rows = sheet.iter_rows(values_only=True)
        next(rows)  # group header
        headers = next(rows)
        ix = {str(v): i for i, v in enumerate(headers) if v is not None}
        missing_headers = set(FEATURES + ["P", "La", "Th"]) - set(ix)
        if missing_headers:
            raise ValueError(f"{sheet.title} missing columns: {missing_headers}")
        for row in rows:
            if not row or row[0] is None:
                continue
            grain = {"RIVER": sheet.title, "SPOT_ID": str(row[0]),
                     "AGE_MA": pd.to_numeric(row[11], errors="coerce"),
                     "P": as_positive(row[ix["P"]]),
                     "La": as_positive(row[ix["La"]]),
                     "Th": as_positive(row[ix["Th"]])}
            for feature in FEATURES:
                grain[feature] = as_positive(row[ix[feature]])
            records.append(grain)
    df = pd.DataFrame(records)
    df["N_FEATURES_MEASURED"] = df[FEATURES].notna().sum(axis=1)
    df["TH_U"] = df["Th"] / df["U"]
    df["REJECT_REASON"] = ""
    def reason(row):
        why = []
        if not pd.notna(row.AGE_MA) or not 0 <= row.AGE_MA < 4000:
            why.append("age_outside_0_4000")
        if row.N_FEATURES_MEASURED < 12:
            why.append("less_than_12_elements")
        if pd.notna(row.P) and row.P > 750:
            why.append("P_gt_750")
        if pd.notna(row.La) and row.La > 1:
            why.append("La_gt_1")
        if pd.notna(row.TH_U) and row.TH_U < 0.1:
            why.append("Th_U_lt_0.1")
        return ";".join(why)
    df["REJECT_REASON"] = df.apply(reason, axis=1)
    df["QC_PASS"] = df["REJECT_REASON"].eq("")
    return df


def predict(df):
    if sklearn.__version__ != "1.3.2":
        raise RuntimeError(
            "Frozen joblib models require scikit-learn==1.3.2; "
            f"current version: {sklearn.__version__}. Install requirements.txt "
            "in a separate environment and run this script again."
        )
    mlp_path = next(MODELS.glob("*MLP*.joblib"))
    tree_path = next(MODELS.glob("*ExtraTrees*.joblib"))
    mlp = joblib.load(mlp_path)["bundle"]
    trees = joblib.load(tree_path)["bundle"]
    for name, bundle in [("MLP", mlp), ("ExtraTrees", trees)]:
        names = [str(v).split("（")[0].split("(")[0] for v in bundle["features"]]
        if names != FEATURES:
            raise RuntimeError(f"{name} feature-order mismatch: {names}")
    x = np.log10(df[FEATURES].astype(float))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X has feature names, but SimpleImputer")
        xm = mlp["imputer"].transform(x)
        xe = trees["imputer"].transform(x)
    df = df.copy()
    df["PRED_MLP"] = np.mean([m.predict(mlp["scaler"].transform(xm))
                              for m in mlp["models"]], axis=0)
    df["PRED_EXTRATREES"] = trees["model"].predict(xe)
    df["PRED_ENSEMBLE"] = (df["PRED_MLP"] + df["PRED_EXTRATREES"]) / 2
    return df


def summarize(df):
    stage_rows = []
    for label, (a, b) in STAGES.items():
        sub = df[(df.AGE_MA >= a) & (df.AGE_MA < b)]
        for river, group in sub.groupby("RIVER"):
            stage_rows.append({"STAGE": label, "RIVER": river,
                               "N": len(group), "PRED_MEDIAN": group.PRED_ENSEMBLE.median(),
                               "MLP_MEDIAN": group.PRED_MLP.median(),
                               "TREES_MEDIAN": group.PRED_EXTRATREES.median()})
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(OUT / "03_按河流分阶段.csv", index=False, encoding="utf-8-sig")
    tab = stages.pivot(index="RIVER", columns="STAGE", values="PRED_MEDIAN")
    paired = tab.dropna(subset=list(STAGES)).copy()
    paired["MIOCENE_MINUS_PALEOCENE_EOCENE"] = paired["Miocene"] - paired["Paleocene_Eocene"]
    paired.to_csv(OUT / "04_同河流配对差值.csv", encoding="utf-8-sig")
    # Four catchments are the independent sampling units. Do not treat grains as independent regions.
    report = {"total_valid": len(df), "source_rivers": int(df.RIVER.nunique()),
              "stages": {s: {"grains": int(sum((df.AGE_MA >= a) & (df.AGE_MA < b))),
                             "rivers": int(stages[stages.STAGE == s].RIVER.nunique())}
                         for s, (a, b) in STAGES.items()},
              "paired_rivers": int(len(paired)),
              "paired_river_differences": paired["MIOCENE_MINUS_PALEOCENE_EOCENE"].round(5).to_dict(),
              "paired_difference_median": (float(paired["MIOCENE_MINUS_PALEOCENE_EOCENE"].median())
                                           if len(paired) else None),
              "reference_magnetite_ilmenite_mean_delta_FMQ": {
                  "Paleocene_Eocene": -0.2, "Miocene": 1.8},
              "note": "Reference values are from Wang et al. 2014, not training labels. "
                      "Agreement is assessed by direction; comparison across different minerals "
                      "does not establish a grain-level RMSE."}
    (OUT / "05_地区对照摘要.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return stages, paired


def figures(df, stage, paired):
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160})
    recent = df[df.AGE_MA.between(0, 120)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for river, sub in recent.groupby("RIVER"):
        x = np.arange(0, 121, 5)
        med = sub.groupby(pd.cut(sub.AGE_MA, x, right=False), observed=True).PRED_ENSEMBLE.median()
        middle = np.array([v.mid for v in med.index])
        ax.plot(middle, med.values, marker="o", ms=3, label=river)
    ax.set(xlabel="Zircon crystallization age (Ma; older to right)",
           ylabel="Frozen-model prediction (Delta FMQ)",
           title="Tang et al. Gangdese river-sand detrital zircons")
    ax.legend();fig.tight_layout();fig.savefig(OUT / "Figure1_四条河流预测.png");plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 5))
    for river, sub in stage.groupby("RIVER"):
        a = sub.set_index("STAGE").PRED_MEDIAN
        if set(STAGES) <= set(a.index):
            ax.plot([0, 1], [a["Paleocene_Eocene"], a["Miocene"]],
                    marker="o", label=river)
    ax.set_xticks([0, 1], ["Paleocene–Eocene\n40–65 Ma", "Miocene\n10–25 Ma"])
    ax.set_ylabel("Median predicted Delta FMQ per river")
    ax.set_title("Paired within-river stage comparison")
    ax.legend();fig.tight_layout();fig.savefig(OUT / "Figure2_河流配对阶段差.png");plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    raw = load_and_filter()
    raw[["RIVER", "SPOT_ID", "AGE_MA", "N_FEATURES_MEASURED", "QC_PASS",
         "REJECT_REASON"]].to_csv(OUT / "01_逐颗筛选对账.csv", index=False, encoding="utf-8-sig")
    valid = raw[raw.QC_PASS].copy()
    if valid.empty:
        raise RuntimeError("No eligible grains after predefined screening.")
    pred = predict(valid)
    pred.to_csv(OUT / "02_逐颗冻结模型预测.csv", index=False, encoding="utf-8-sig")
    stages, paired = summarize(pred)
    figures(pred, stages, paired)
    print("完成：", OUT)
    print("筛选颗粒：", len(pred), "；成对河流：", len(paired))
    print("运行过程仅调用冻结模型 predict，没有 fit、训练或调参。")


if __name__ == "__main__":
    main()
