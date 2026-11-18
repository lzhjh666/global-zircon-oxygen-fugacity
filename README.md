# Global Zircon Magmatic Oxygen Fugacity Reconstruction

**Research code and frozen inference models for reconstructing long-term magmatic oxygen fugacity (ΔFMQ) trends from zircon geochemistry.**

本仓库整理全球锆石氧逸度研究所使用的冻结模型、数据处理与分析工作流、区域验证程序、统计结果及图件。研究主线是全球 0–4000 Ma 岩浆氧逸度变化；冈底斯和 Kohistan 数据仅用于独立区域验证，不替代全球研究目标。

> **Public-release data note.** The complete 10,879-row grain-level global compilation is not redistributed in this public repository. Of those records, 10,252 originate from the Wang et al. (2025) GSA supplementary compilation, for which redistribution permission has not been established; the remaining 627 GEOROC-derived records are provided separately under GEOROC's CC BY-SA 4.0 terms. Training tables assembled from multiple publications are also not bundled. See [`docs/DATA_AVAILABILITY_AND_LICENSES.md`](docs/DATA_AVAILABILITY_AND_LICENSES.md).

## Research scope

- Predict zircon-hosted magmatic oxygen fugacity relative to the fayalite–magnetite–quartz buffer (ΔFMQ).
- Apply the frozen MLP 64→32 and ExtraTrees model bundles.
- Evaluate model behaviour using grouped validation, source-level sensitivity analysis, age-domain diagnostics, and regional external validation.
- Reconstruct global age-binned trends and report data support and uncertainty diagnostics.

The repository preserves the paper-use frozen models. The less successful V2 experiment is not used as the production model. No model is retrained by the global prediction or regional validation scripts.

## Repository structure

```text
.
├── models/                 # Frozen MLP and ExtraTrees model bundles
├── data/
│   ├── audits/             # Aggregate audit records and the 627-row GEOROC addition
│   ├── frozen/             # Source and publication registries; freeze audit report
│   └── provenance/         # Aggregate provenance summaries and source-label registry
├── workflows/
│   ├── global_prediction/  # GEOROC / publisher-data processing and prediction pipeline
│   ├── model_development/  # Training and internal grouped-validation code
│   ├── bootstrap/          # Source-level bootstrap analysis
│   ├── sensitivity/        # Time-window, source-removal, and model sensitivity analyses
│   ├── age_domain/         # Age-dependent applicability-domain analysis
│   ├── gangdese/           # Gangdese regional validation workflow
│   ├── kohistan_validation/# Independent Kohistan validation workflow
│   └── figures/            # Release-figure plotting code
├── results/                # Aggregate metrics, summaries, and validation outputs
├── figures/                # Research figures
├── docs/                   # Data access, licensing, and reproducibility notes
├── requirements.txt
├── CITATION.cff
└── SHA256_MANIFEST.json
```

## Environment

Recommended: Python 3.10 or 3.11.

```bash
python -m pip install -r requirements.txt
```

The frozen model bundles were saved with scikit-learn 1.3.2. Keep this version when loading the supplied `joblib` files. Only load model files from a trusted source.

## Main workflows

### 1. Global prediction

`workflows/global_prediction/run_georoc_pipeline.py` supports the publisher-compilation mode and the GEOROC mode. The corresponding source inputs are not all included in this public release. See [`docs/EXTERNAL_DATA.md`](docs/EXTERNAL_DATA.md) for the official source, required local filename, and command examples.

### 2. Model development

The training scripts are retained for transparency. The original multi-publication calibration table is not redistributed because its source records have differing publication and reuse terms. If you have lawful access to the compiled training input, supply it explicitly:

```bash
python workflows/model_development/train.py --input /path/to/authorized_training_input.csv
```

The regularized ExtraTrees experiment is under `workflows/model_development/extra_trees_regularization/`. Its required training input is likewise not bundled.

### 3. Regional validation

- **Gangdese:** `workflows/gangdese/冈底斯_冻结模型预测.py` uses the frozen models in `models/`. The Tang et al. (2021) supplementary workbook must be downloaded from the publisher and placed at `workflows/gangdese/G47745_SuppData.xlsx`.
- **Kohistan:** `workflows/kohistan_validation/01_运行Kohistan独立外部验证.py` uses the CC BY 4.0 supplementary data included under `workflows/kohistan_validation/数据来源/`.

### 4. Figures and aggregate analyses

`workflows/figures/plot_release_figures.py` regenerates the release figures from the aggregate tables shipped in `results/`. Bootstrap, sensitivity, and age-domain scripts are retained; workflows that require the full grain-level Wang et al. compilation need that source input to be obtained separately.

## Data interpretation and limitations

- A database source label does not prove that every record is a detrital zircon; grain-level identity and host-rock context should be interpreted using the provenance status fields and reports.
- The model predicts zircon crystallization / source-magma redox information (ΔFMQ). It is not a direct measurement of the average oxidation state of the entire continental crust or atmospheric oxygen.
- The full grain-level global input is intentionally not included in this public release. Aggregate results and figures are provided for inspection; do not infer that every analysis can be rerun from this repository alone.
- The original manuscript figures are supplied as research outputs. The plotting script reproduces the corresponding figure themes from the available aggregate tables; it is not claimed to reproduce every manuscript figure pixel-for-pixel.

## Citation

Please cite the relevant data sources and methods listed in [`docs/DATA_AVAILABILITY_AND_LICENSES.md`](docs/DATA_AVAILABILITY_AND_LICENSES.md). If you use this repository in research, cite the repository version and the original publications from which the data were compiled.

## Language

The code comments and workflow notes are primarily in Chinese. The repository overview is bilingual so that both Chinese-speaking collaborators and international researchers can navigate the project.
