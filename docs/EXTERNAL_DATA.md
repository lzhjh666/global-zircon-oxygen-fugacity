# External data inputs and reproducibility

This repository intentionally does not rehost every third-party source table. Obtain source files from the official publisher/database and follow the license or terms shown at the source.

## 1. Wang et al. (2025), global detrital-zircon compilation

- Article: *Evolution of magma redox state throughout Earth's history revealed by zircon geochemistry*.
- DOI: https://doi.org/10.1130/B38079.1
- Supplementary material DOI: https://doi.org/10.1130/GSAB.S.30104938
- The public release does **not** include the publisher's 20,489-row supplementary workbook or the 10,252 row-level records derived from that compilation.
- For local reproduction of `run_georoc_pipeline.py --data-mode paper`, obtain the supplementary workbook from the publisher and supply it using `--paper-data` (default local path: `data/B38079_SuppTab2_全球碎屑锆石微量元素.xlsx`).

## 2. Tang et al. (2021), Gangdese river detrital zircons

- Article: *Reconstructing crustal thickness evolution from europium anomalies in detrital zircons*.
- DOI: https://doi.org/10.1130/G47745.1
- The publisher supplementary workbook is not redistributed in this repository.
- For local reproduction of the Gangdese script, obtain the official supplementary workbook and place it at `workflows/gangdese/G47745_SuppData.xlsx`.

## 3. GEOROC

- Website and citation guidance: https://georoc.eu/georoc/cite.asp
- The 627-row GEOROC-derived addition included in `data/audits/新增GEOROC_627条.csv` is shared under CC BY-SA 4.0.
- Cite GEOROC and the original publications represented in the data. Any further redistributed GEOROC-derived compilation must retain the same license and attribution requirements.
- The complete raw GEOROC download is not bundled. Use the current official GEOROC download and preserve its version and query metadata.

## 4. Kohistan

- Ahmad et al. (2026), *Tectono-magmatic evolution of the central Kohistan Arc, NW Pakistan: Granitoid emplacement revealed by mineral chemistry*.
- DOI: https://doi.org/10.1016/j.jseaes.2026.107049
- The supplementary workbook retained under `workflows/kohistan_validation/数据来源/` is associated with the article's CC BY 4.0 open-access publication. Retain the citation and license notice when redistributing it.

## 5. Model calibration tables

The multi-publication training/calibration tables are not included in this public release. Their source papers and data-entry information are documented in the project records, but the reuse terms differ among publications. Obtain each source dataset under its applicable terms and reconstruct the authorized training table before running the training scripts.
