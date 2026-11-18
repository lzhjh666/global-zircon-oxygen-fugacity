# -*- coding: utf-8 -*-
"""只用原训练数据选择ExtraTrees正则化参数；不读取任何外部验证数据。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from validation_core import GROUP, SITE, TARGET, load_locked_bundle, log_x, regression_metrics


SEED = 81001
EXCLUDED_GROUPS = {
    "Mt St Helens, Yn tephra, 3510 yr BP | ΔFMQ=2.75",
    "Kimberlites, Kaapvaal, Southern Africa | ΔFMQ=-2.810",
}
CANDIDATES = [
    {"min_samples_leaf": 3, "max_depth": None, "max_features": 0.75, "bootstrap": False},
    {"min_samples_leaf": 5, "max_depth": None, "max_features": 0.80, "bootstrap": False},
    {"min_samples_leaf": 5, "max_depth": 9, "max_features": 0.80, "bootstrap": False},
    {"min_samples_leaf": 8, "max_depth": 12, "max_features": 0.80, "bootstrap": False},
    {"min_samples_leaf": 8, "max_depth": None, "max_features": 0.50, "bootstrap": False},
    {"min_samples_leaf": 10, "max_depth": 12, "max_features": 0.65, "bootstrap": False},
    {"min_samples_leaf": 12, "max_depth": 9, "max_features": 0.65, "bootstrap": False},
    {"min_samples_leaf": 12, "max_depth": 12, "max_features": 0.80, "bootstrap": False},
    {"min_samples_leaf": 16, "max_depth": 9, "max_features": 0.65, "bootstrap": False},
]


def target_band(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-np.inf, 0.5, 2.0, np.inf], labels=["低", "中", "高"], right=False)


def site_folds(data: pd.DataFrame, n_splits: int, seed: int):
    site_target = data.groupby(SITE)[TARGET].median().sort_index()
    bands = target_band(site_target)
    held = [set() for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for band in ["低", "中", "高"]:
        names = np.asarray(sorted(site_target.index[bands == band]), dtype=object)
        rng.shuffle(names)
        for index, name in enumerate(names):
            held[index % n_splits].add(str(name))
    for fold, test_sites in enumerate(held, 1):
        train = data.loc[~data[SITE].isin(test_sites)].copy().reset_index(drop=True)
        test = data.loc[data[SITE].isin(test_sites)].copy().reset_index(drop=True)
        yield fold, sorted(test_sites), train, test


def aggregate_groups(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    work = frame[[GROUP, TARGET]].copy()
    work["预测值"] = np.asarray(prediction, float)
    return work.groupby(GROUP, as_index=False).agg(
        真实值=(TARGET, "first"), 预测值=("预测值", "median"))


def fit_model(train: pd.DataFrame, features: list[str], params: dict, seed: int, trees: int):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    x = imputer.fit_transform(log_x(train, features))
    y = train[TARGET].to_numpy(float)
    weights = 1.0 / train.groupby(GROUP)[GROUP].transform("size").to_numpy(float)
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=params["min_samples_leaf"],
        max_depth=params["max_depth"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y, sample_weight=weights)
    return imputer, model


def predict(imputer, model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(imputer.transform(log_x(frame, features)))


def evaluate_candidate(data, features, params, repeats=5, trees=500):
    rows = []
    for repeat in range(1, repeats + 1):
        for fold, held, train, test in site_folds(data, 5, SEED + repeat * 10007):
            imputer, model = fit_model(
                train, features, params, SEED + repeat * 100003 + fold * 1009, trees)
            train_groups = aggregate_groups(train, predict(imputer, model, train, features))
            test_groups = aggregate_groups(test, predict(imputer, model, test, features))
            train_metrics = regression_metrics(train_groups["真实值"], train_groups["预测值"])
            test_metrics = regression_metrics(test_groups["真实值"], test_groups["预测值"])
            rows.append({
                "轮次": repeat,
                "折": fold,
                "测试地点": "；".join(held),
                "训练MSE": train_metrics["MSE"],
                "测试MSE": test_metrics["MSE"],
                "测试R2": test_metrics["R2"],
            })
    detail = pd.DataFrame(rows)
    summary = {
        **params,
        "完成折数": int(len(detail)),
        "训练MSE": float(detail["训练MSE"].mean()),
        "测试MSE": float(detail["测试MSE"].mean()),
        "测试MSE标准差": float(detail["测试MSE"].std(ddof=1)),
        "测试R2": float(detail["测试R2"].mean()),
        "测试训练误差比": float(detail["测试MSE"].mean() / max(detail["训练MSE"].mean(), 1e-12)),
        "MSE低于0.4折占比": float((detail["测试MSE"] < 0.4).mean()),
    }
    return summary, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="训练数据/入模数据_宽松筛选.csv")
    parser.add_argument("--model-dir", default="模型文件放这里")
    parser.add_argument("--output", default="ExtraTrees正则化内部结果")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_path = (root / args.data).resolve()
    model_dir = (root / args.model_dir).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    mlp_bundle = load_locked_bundle(model_dir / "MLP64x32_锁定模型.joblib")
    features = list(mlp_bundle["features"])
    raw = pd.read_csv(data_path, encoding="utf-8-sig")
    required = features + [TARGET, GROUP, SITE]
    missing = [column for column in required if column not in raw]
    if missing:
        raise ValueError("训练数据缺少列：" + "、".join(missing))
    data = raw.loc[~raw[GROUP].isin(EXCLUDED_GROUPS)].copy().reset_index(drop=True)
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data[features] = data[features].mask(data[features] <= 0)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET, GROUP, SITE]).reset_index(drop=True)

    summaries, details = [], []
    for candidate_id, params in enumerate(CANDIDATES, 1):
        print(f"正在复核ExtraTrees候选 {candidate_id}/{len(CANDIDATES)}：{params}", flush=True)
        summary, detail = evaluate_candidate(data, features, params)
        summary["候选编号"] = candidate_id
        detail.insert(0, "候选编号", candidate_id)
        summaries.append(summary)
        details.append(detail)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["测试MSE", "测试MSE标准差", "测试训练误差比"]).reset_index(drop=True)
    detail_df = pd.concat(details, ignore_index=True)
    summary_df.to_csv(output / "ExtraTrees正则化候选汇总.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(output / "ExtraTrees正则化逐折结果.csv", index=False, encoding="utf-8-sig")

    lowest = summary_df.iloc[0]
    one_se = float(lowest["测试MSE标准差"] / math.sqrt(lowest["完成折数"]))
    threshold = float(lowest["测试MSE"] + one_se)
    eligible = summary_df[summary_df["测试MSE"] <= threshold]
    selected = eligible.sort_values(["测试训练误差比", "测试MSE"]).iloc[0]
    params = {
        "min_samples_leaf": int(selected["min_samples_leaf"]),
        "max_depth": None if pd.isna(selected["max_depth"]) else int(selected["max_depth"]),
        "max_features": float(selected["max_features"]),
        "bootstrap": bool(selected["bootstrap"]),
    }

    imputer, model = fit_model(data, features, params, SEED + 990001, trees=1200)
    bundle = {
        "kind": "ExtraTrees",
        "features": features,
        "imputer": imputer,
        "model": model,
        "n_estimators": 1200,
        "regularized_params": params,
    }
    joblib.dump({
        "bundle": bundle,
        "metadata": {
            "参数选择数据": "只使用原训练数据",
            "外部标签是否参与": False,
            "选择规则": "一标准误差范围内选择测试/训练MSE比最低者",
            "候选汇总": selected.to_dict(),
        },
    }, model_dir / "ExtraTrees_正则化锁定模型.joblib")

    clean_selected = {key: (None if pd.isna(value) else value)
                      for key, value in selected.to_dict().items()}
    (output / "最终正则化参数.json").write_text(json.dumps({
        "选择阈值MSE": threshold,
        "最终参数": params,
        "内部验证结果": clean_selected,
        "说明": "外部验证数据未被读取，不得根据后续外部结果反复修改这些参数。",
    }, ensure_ascii=False, indent=2,
       default=lambda value: value.item() if hasattr(value, "item") else str(value)),
       encoding="utf-8")
    print("\n完成。正则化模型已保存到：", model_dir / "ExtraTrees_正则化锁定模型.joblib")
    print("内部参数结果位于：", output)


if __name__ == "__main__":
    main()
