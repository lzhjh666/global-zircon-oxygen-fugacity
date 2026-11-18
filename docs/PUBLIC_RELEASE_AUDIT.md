# Public release audit

This public-release package was prepared from the user-supplied release archive.

## Preserved
- Frozen production MLP and ExtraTrees models.
- Research scripts and configuration files.
- Aggregate model-validation, bootstrap, sensitivity, age-domain, and regional summaries.
- Figures and documentation.
- The 627-row GEOROC-derived addition under GEOROC CC BY-SA 4.0 terms.
- The Kohistan 2026 supplementary workbook under the article's CC BY 4.0 terms.

## Excluded from this public package
- Wang et al. (2025) publisher supplementary workbook and row-level records derived from its global compilation.
- Tang et al. (2021) publisher supplementary workbook and per-grain Gangdese input/output tables.
- Multi-publication row-level model training/calibration tables whose individual redistribution terms have not been verified.
- Redundant copies of the frozen models and source workbooks.

## Technical changes
- The Gangdese script now loads the single frozen model copy from the repository-root `models/` directory.
- `.gitignore`, README, external-data instructions, data-license notes, and `CITATION.cff` were added or updated.
- `SHA256_MANIFEST.json` was regenerated for the files in this release.

This package is suitable as a public code-and-aggregate-results release. It is not a complete standalone reproduction archive because some publisher-controlled source inputs are intentionally not redistributed.
