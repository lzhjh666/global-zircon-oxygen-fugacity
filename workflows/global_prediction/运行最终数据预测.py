#!/usr/bin/env python3
"""使用旧版正式冻结模型预测最终10879条全球碎屑锆石；只推理，不训练。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from run_georoc_pipeline import FEATURES, load_models, predict_file, binned, setup_plot, savefig

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data/prediction_input/07_合并后全球主数据_10879条.csv"
MODELS = ROOT.parents[1] / "models"
OUT = ROOT / "最终预测结果"


def main():
    OUT.mkdir(exist_ok=True)
    # 最终表由 Wang 汇编与 GEOROC 新增记录拼接而成；两批数据原始年龄列名不同。
    # 统一优先使用 AGE_MA_CLEAN，空值再由 Selected_age 补齐，防止绘图只统计新增批次。
    source = pd.read_csv(DATA, low_memory=False)
    clean_age = pd.to_numeric(source.get("AGE_MA_CLEAN"), errors="coerce")
    selected_age = pd.to_numeric(source.get("Selected_age"), errors="coerce")
    source["AGE_MA_CLEAN"] = clean_age.combine_first(selected_age)
    if source["AGE_MA_CLEAN"].isna().any():
        raise RuntimeError("最终数据仍存在无法解析的年龄，已停止预测。")
    normalized = OUT / "01_统一年龄字段后的10879条输入.csv"
    source.to_csv(normalized, index=False, encoding="utf-8-sig")
    models = load_models(MODELS)
    pred = predict_file(normalized, OUT, models, chunksize=30000)
    trend, all_pred = binned(pred, 100)
    trend.to_csv(OUT / "02_每100Ma氧逸度演化.csv", index=False, encoding="utf-8-sig")

    setup_plot()
    fig_dir = OUT / "图件"
    fig_dir.mkdir(exist_ok=True)
    x = trend["年龄中心_Ma"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, trend["MLP_中位数"], label="MLP", alpha=.75)
    ax.plot(x, trend["ExtraTrees_中位数"], label="ExtraTrees", alpha=.75)
    ax.plot(x, trend["双模型均值_三窗平滑"], color="black", lw=2,
            label="双模型均值（3窗平滑）")
    ax.fill_between(x, trend["双模型均值_Q25"], trend["双模型均值_Q75"],
                    color="gray", alpha=.18, label="四分位范围")
    ax.axhline(0, color="gray", lw=.8)
    ax.set(xlabel="年龄（Ma，向右更古老）", ylabel="预测 ΔFMQ",
           title="最终10879条全球碎屑锆石氧逸度演化")
    ax.legend()
    savefig(fig, fig_dir / "01_全球氧逸度演化.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, trend["样本数"], color="#d55e00")
    ax.set(xlabel="年龄（Ma）", ylabel="样本数", title="每100 Ma样本量")
    savefig(fig, fig_dir / "02_每100Ma样本量.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, trend["模型差异中位数"], color="#8c564b")
    ax.set(xlabel="年龄（Ma）", ylabel="|MLP−ExtraTrees|", title="两模型预测分歧")
    savefig(fig, fig_dir / "03_模型差异.png")

    manifest = {
        "model_status": "OLD_MODELS_FROZEN",
        "training_performed": False,
        "input_file": DATA.name,
        "input_rows": int(len(all_pred)),
        "age_missing_after_merge": int(source["AGE_MA_CLEAN"].isna().sum()),
        "age_merge_rule": "AGE_MA_CLEAN first; fill missing with Selected_age",
        "features": FEATURES,
        "models": [p.name for p in sorted(MODELS.glob("*.joblib"))],
        "note": "本脚本仅调用冻结模型predict，不执行fit或调参。"
    }
    (OUT / "00_运行说明.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{OUT}")


if __name__ == "__main__":
    main()
