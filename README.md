[README.md](https://github.com/user-attachments/files/34087073/README.md)
# BNT v2.2 — Forecasting Non-Tariff Trade Measures with Machine Learning

**An empirical forecasting framework for WTO SPS/TBT notifications and Specific Trade Concerns (STCs).**

BNT v2.2 builds a monthly panel of **WTO member × Harmonized System (HS) product section** observations to estimate the risk of new non-tariff regulatory activity. It combines historical notifications, trade flows, macroeconomic indicators, and regulatory text in a reproducible, time-aware machine learning pipeline.

> **Research status:** This repository contains an experimental forecasting pipeline, not a validated real-time warning service. Model performance must be assessed from the output of an actual run; no benchmark scores are asserted in this README.

## Research questions

- Can historical regulatory activity predict future SPS/TBT notifications?
- Do trade, macroeconomic, and text features add predictive value beyond simple historical baselines?
- Can the framework identify new regulatory activity in previously inactive product sections?
- Can forecasts help prioritize trade-policy monitoring under limited resources?

## Prediction tasks

| Target | Definition | Forecast horizons |
|---|---|---|
| `any` | A new SPS/TBT notification in the relevant member–product group | 3 and 6 months |
| `onset` | A new notification following **12 months without notifications** in that group | 3 and 6 months |
| `stc` | A WTO Specific Trade Concern raised against the member in the relevant product group | 3 and 6 months |

The `stc` task is skipped if STC data cannot be loaded. The default product resolution is **HS section** (21 sections), with an option to work at HS chapter level.

## Data sources

| Source | Role |
|---|---|
| [WTO ePing](https://eping.wto.org/) | SPS/TBT notifications, dates, descriptions, and product references |
| [WTO Trade Concerns](https://tradeconcerns.wto.org/en/stcs) | Specific Trade Concerns (STCs) |
| [CEPII BACI](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) | International trade flows |
| [World Bank API](https://api.worldbank.org/) | Macroeconomic indicators |
| [WTO Timeseries API](https://api.wto.org/) | Optional WTO indicators, when configured |
| [GDELT](https://www.gdeltproject.org/) | News context and monitoring alerts |

Data availability, revisions, publication lags, and source coverage can affect results. The code includes explicit release-lag assumptions for selected annual indicators and STCs.

## Modeling approach

The notebook evaluates several model families rather than assuming that a complex algorithm will outperform a simple baseline:

- Frequency, persistence, and smoothed-frequency baselines
- Logistic regression, including historical-feature and fixed-effect variants
- **LightGBM** with historical, text, macroeconomic, trade, and contextual feature combinations
- A fallback to scikit-learn's histogram gradient boosting if LightGBM is unavailable

Text representations use **TF–IDF** and **Truncated SVD**. The framework also tests feature-family contributions and performs falsification and robustness checks.

### Temporal evaluation protocol

1. Construct labels at the configured 3- and 6-month horizons.
2. Allocate the **first 80% of labeled months** to development and the **last 20%** to a held-out test period.
3. Use development-period folds for model selection and tuning.
4. Reserve the final **12 development months** for calibration, subject to the implemented protocol.
5. Register model decisions before evaluating on the held-out test period.
6. Evaluate discrimination, calibration, robustness, and monitoring utility.

The implementation includes isotonic calibration, grouped bootstrap procedures, sensitivity checks, and tests against baseline models. This design is intended to reduce look-ahead bias; users should still audit feature timestamps and source revisions for their own replication.

### Evaluation metrics

- ROC-AUC and PR-AUC
- Brier score and calibration diagnostics
- Precision and lift among the top 5% of predicted risks
- Paired comparisons and grouped bootstrap uncertainty
- Sensitivity to member selection, feature groups, and stress periods
- Scenario-based monitoring utility and cost assumptions

## Getting started

### Option A: Kaggle (recommended)

1. Upload `bnt-model (1)(1).ipynb` to a Kaggle Notebook (consider renaming it to `bnt_v2_2.ipynb`).
2. Enable **Internet** under notebook settings so the data loaders can access public sources.
3. Optionally add `WTO_API_KEY` under **Add-ons → Secrets**.
4. Run the notebook cells in order.
5. Inspect the generated CSV reports and JSON manifests in `/kaggle/working/bnt_v2/`.

### Option B: Local Jupyter environment

Use Python 3 with Jupyter and the notebook's imported dependencies. A practical starting point is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install notebook numpy pandas requests matplotlib scikit-learn lightgbm
jupyter notebook
```

The notebook imports additional Python standard-library modules and may require source-specific connectivity or locally provided input files. On a local machine, the default output directory is `bnt_v2_out/`, the input directory is `inputs/`, and the cache is stored under `/tmp/bnt_v2_cache` (adjust paths as appropriate for your operating system).

## Configuration

The main settings are defined in the notebook's `Config` class:

```python
START = '2000-01'
END = None
LEVEL = 'section'          # or 'chapter'
HORIZONS = [3, 6]
TARGETS = ['any', 'onset', 'stc']
ONSET_GAP = 12
DEV_FRAC = 0.80
CAL_MONTHS = 12
FOCUS_EXPORTER = 'BRA'
SEED = 42
FAST = False
```

`END = None` uses the latest eligible completed period inferred from the notification source. `FAST = True` is intended for debugging rather than final evaluation. Other settings control member eligibility, data-source versions, publication lags, text dimensionality, bootstrap replications, and monitoring-cost scenarios.

## Outputs and reproducibility

Depending on data availability and the tasks successfully executed, the pipeline writes files such as:

```text
bnt_v2/                     # Kaggle; bnt_v2_out/ locally
├── execucao.log
├── auditoria_rotulos_casos.csv
├── auditoria_rotulos_por_ano.csv
├── auditoria_disponibilidade.csv
├── selecao_amostra_membros.csv
├── hipoteses_teste_final.csv
├── hipoteses_desenvolvimento_exploratorio.csv
├── desenvolvimento_resumo_<target>_h<horizon>.csv
├── teste_metricas_<target>_h<horizon>.csv
├── teste_vs_baselines_<target>_h<horizon>.csv
├── teste_calibracao_decis_<target>_h<horizon>.csv
├── robustez_grupos_<target>_h<horizon>.csv
├── falsificacao_<target>_h<horizon>.csv
├── utilidade_<target>_h<horizon>.csv
├── previsoes_<target>_h<horizon>.csv
├── noticias_ultima_execucao.csv
└── manifesto_reprodutibilidade.json
```

Additional diagnostics and summary artifacts are generated by the notebook. The project supports input snapshots (`SNAPSHOT_INPUTS = True`) and replay from an earlier snapshot (`REPLAY_FROM`) to help reproduce experiments using the same data inputs.

## Important limitations

- **Notifications are proxies for regulatory activity**, not direct proof of trade restrictiveness or economic harm.
- **STCs are a distinct WTO process** and should not be interpreted as interchangeable with notifications.
- Source coverage, publication delays, retrospective revisions, and incomplete product mappings can affect observed labels and features.
- A high ROC-AUC does not establish causal validity, calibrated probabilities, or practical usefulness.
- Model selection and reporting should remain separate from the untouched test period.
- GDELT monitoring signals and forecasts should be treated as decision-support inputs, not verified future events.
- Estimated economic utility depends on explicitly stated assumptions about avoided losses and monitoring costs.

## Project structure

The current repository artifact is a **single end-to-end Jupyter notebook**. It includes data ingestion, panel construction, feature engineering, model comparison, evaluation, forecasting, reporting, and a fallback procedure for retrieving WTO STC records through a browser when API access fails.

## Suggested citation

If you use this code in research, cite the repository URL and the specific commit or release used. Add author, institution, license, and publication details once they have been finalized.

## License

**Not specified.** No open-source license is implied by publication on GitHub. Add a `LICENSE` file if you intend to grant reuse rights.
