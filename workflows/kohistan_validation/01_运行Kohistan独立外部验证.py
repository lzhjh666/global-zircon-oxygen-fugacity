# -*- coding: utf-8 -*-
"""用冻结模型验证Kohistan 2026数据；不训练、不调参、不更换元素。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from validation_core import load_locked_bundle, predict_bundle, regression_metrics, sha256


def set_chinese_font():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_results(metrics, groups, output):
    set_chinese_font()
    colors = ["#2878B5", "#F28E2B", "#59A14F"]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(metrics["模型"], metrics["MSE"], color=colors[:len(metrics)])
    ax.axhline(0.4, color="#B22222", linestyle="--", linewidth=1.6, label="导师目标 MSE=0.4")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set(ylabel="五岩体等权MSE", title="Kohistan 2026 全新独立外部验证")
    ax.grid(axis="y", alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / "01_外部验证MSE.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    low = min(groups["真实ΔFMQ"].min(), groups["预测ΔFMQ"].min()) - 0.25
    high = max(groups["真实ΔFMQ"].max(), groups["预测ΔFMQ"].max()) + 0.25
    for i, (name, part) in enumerate(groups.groupby("模型", sort=False)):
        ax.scatter(part["真实ΔFMQ"], part["预测ΔFMQ"], s=70, label=name,
                   color=colors[i % len(colors)], alpha=0.85)
    ax.plot([low, high], [low, high], "--", color="#B22222", label="理想预测线")
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="角闪石独立约束 ΔFMQ",
           ylabel="冻结模型预测 ΔFMQ", title="5个全新岩体：真实值与整岩体预测")
    ax.grid(alpha=0.2); ax.legend(); fig.tight_layout()
    fig.savefig(output / "02_真实值与整岩体预测.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    names = list(groups["模型"].drop_duplicates())
    rocks = list(groups["岩体"].drop_duplicates())
    x = np.arange(len(rocks)); width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    truth = groups.drop_duplicates("岩体").set_index("岩体").reindex(rocks)["真实ΔFMQ"]
    ax.plot(x, truth, "o-", color="#B22222", linewidth=2, label="独立约束")
    for i, name in enumerate(names):
        part = groups[groups["模型"] == name].set_index("岩体").reindex(rocks)
        ax.bar(x + (i - (len(names)-1)/2)*width, part["预测ΔFMQ"], width=width,
               color=colors[i % len(colors)], alpha=0.78, label=name)
    ax.set_xticks(x, [r.replace(" Granitoid", "").replace(" Quartz Diorite", "") for r in rocks], rotation=15)
    ax.set(ylabel="ΔFMQ", title="各岩体独立约束与模型预测")
    ax.axhline(0, color="#777777", linewidth=0.8); ax.grid(axis="y", alpha=0.2); ax.legend()
    fig.tight_layout(); fig.savefig(output / "03_各岩体预测对比.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[2] / "models"))
    parser.add_argument("--external-dir", default="外部验证数据")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model_dir = root / args.model_dir
    ext_dir = root / args.external_dir

    candidates = [
        ("MLP_64x32", model_dir / "MLP64x32_锁定模型.joblib"),
        ("ExtraTrees_原版", model_dir / "ExtraTrees_锁定模型.joblib"),
        ("ExtraTrees_正则化", model_dir / "ExtraTrees_正则化锁定模型.joblib"),
    ]
    found = [(name, path) for name, path in candidates if path.exists()]
    if not found:
        raise FileNotFoundError("模型文件夹中没有找到任何锁定模型。请先阅读‘先看这里.md’。")
    models = {name: load_locked_bundle(path) for name, path in found}

    feature_path = ext_dir / "Kohistan_盲测特征_113颗_不含标签.csv"
    label_path = ext_dir / "Kohistan_封存组级标签_评分时读取.csv"
    features = pd.read_csv(feature_path, encoding="utf-8-sig")

    # 第一阶段：只读特征，先写出冻结预测；尚未读取标签文件。
    frozen = features[["外部记录编号", "样品编号", "岩体", "分析点", "年龄_Ma"]].copy()
    for name, bundle in models.items():
        frozen[f"{name}_预测ΔFMQ"] = predict_bundle(bundle, features)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / f"Kohistan独立外部验证结果_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    frozen_path = output / "Kohistan_冻结预测_标签解封前.csv"
    frozen.to_csv(frozen_path, index=False, encoding="utf-8-sig")
    record = {
        "数据是否参与模型训练、元素筛选或调参": False,
        "预测是否先于标签读取写盘": True,
        "冻结预测SHA256": sha256(frozen_path),
        "模型SHA256": {name: sha256(path) for name, path in found},
        "评分单位": "5个岩体等权；不是113颗伪独立样品",
    }
    (output / "冻结预测声明.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # 第二阶段：预测冻结后才读取组级标签。
    labels = pd.read_csv(label_path, encoding="utf-8-sig")
    group_rows, metric_rows = [], []
    for name in models:
        column = f"{name}_预测ΔFMQ"
        pred = frozen.groupby("岩体", as_index=False).agg(
            预测ΔFMQ=(column, "median"), 锆石数=(column, "size"))
        scored = labels[["岩体", "独立约束_ΔFMQ"]].merge(pred, on="岩体", validate="one_to_one")
        scored = scored.rename(columns={"独立约束_ΔFMQ": "真实ΔFMQ"})
        scored["模型"] = name
        scored["残差_预测减真实"] = scored["预测ΔFMQ"] - scored["真实ΔFMQ"]
        group_rows.append(scored[["模型", "岩体", "锆石数", "真实ΔFMQ", "预测ΔFMQ", "残差_预测减真实"]])
        metric_rows.append({"模型": name, **regression_metrics(scored["真实ΔFMQ"], scored["预测ΔFMQ"])})

    groups = pd.concat(group_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "外部验证核心指标_五岩体等权.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output / "五岩体真实值与预测值.csv", index=False, encoding="utf-8-sig")
    frozen.to_csv(output / "113颗锆石冻结预测.csv", index=False, encoding="utf-8-sig")
    plot_results(metrics, groups, output)

    notes = [
        "# Kohistan 2026 全新独立外部验证", "",
        "- 本程序只调用冻结模型，不训练、不调参、不重新筛选元素。",
        "- 评分按5个岩体等权计算，113颗锆石只用于形成各岩体预测中位数。",
        "- 氧逸度真值来自角闪石Ridolfi (2021)方法，独立于锆石17元素模型。",
        "- 论文Table 2把组平均logfO2四舍五入为整数，因此本验证的标签精度有限。",
        "- ΔFMQ由论文组平均logfO2、温压和Frost (1991) FMQ缓冲式换算。",
        "", "## 核心指标", "", "```text", metrics.to_string(index=False), "```",
        "", "## 五岩体结果", "", "```text", groups.to_string(index=False), "```",
    ]
    (output / "先看这里_验证结论.md").write_text("\n".join(notes), encoding="utf-8")
    print("验证完成：", output)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
