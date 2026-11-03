# pagrus-age

Predicts the **scale-read age of red porgy (*Pagrus pagrus*)** from cheap measurements. Reading a
scale is slow and sometimes impossible; 24% of the fish in this dataset have no age.

**Data:** Haimovici, Kikuchi, Cardoso & Morales (2021), *Biological dataset for Pagrus pagrus fished
along Southern Brazil*, SEANOE, <https://doi.org/10.17882/75444> (also published through
[OBIS](https://obis.org/dataset/93c80b5b-3bc3-4e59-a283-b4edebaa63ce)). The project reads the original
SEANOE CSV rather than the OBIS copy. Please also cite Haimovici, Kikuchi & Cardoso (2020),
*Aquatic Living Resources*, <https://doi.org/10.1051/alr/2020010>. SEANOE lists the licence as CC BY
4.0 and marineinfo.org as CC BY-SA 4.0; check before redistributing the file.

## Result

The **simplest adequate model won**: an age-length key, the standard fisheries method. No model using
gear, depth, month, weight or even dissection data clearly beat it.

| On 949 fish from 19 landings never used for selection | Value |
|---|---|
| Mean absolute error | 0.99 years (95% CI 0.90 to 1.17) |
| Predicted age within one year | 78.6% |
| Median-age baseline error | 1.86 years |
| 90% interval coverage | 93.0% (95% CI 88.8 to 94.9) |
| Error on later years (fit up to 1981, test from 1982) | 1.27 years |

Do not rely on it for old fish: at ages 13+ it under-predicts by 3 years on average (only 7 test
fish). Details are in [`reports/model_card.md`](reports/model_card.md).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
make setup        # install pinned dependencies
make all          # lint, types, tests, train, predict
make serve        # API on http://127.0.0.1:8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"fish":[{"total_length_mm":340,"month":8,"gear":"otter_trawl"}]}'
```

`make train` writes `reports/` and `models/`. `make predict` writes ages for unaged fish to
`data/predictions/unaged_fish.csv`.

## Design decisions

| Risk | What the project does |
|---|---|
| Fish from one landing are correlated | Splits, cross-validation and confidence intervals all work at landing level |
| Target leakage | `scale_border` and `scale_readability` exist only because a scale was read. They are blocked in code, and a test enforces it |
| Unaged fish differ from aged fish | Models using features with a shifted missing rate cannot be selected; the card reports the gap |
| Tuning to the test set | Test landings are fixed by hash. Selection uses CV only, and `train --no-test` stops before them |
| Overfitted complexity | The one-standard-error rule picks the simplest model close to the best |
| Bad records | A strict data contract stops on schema changes. Fixed QC rules set implausible values to missing |
| Silent data change | The raw file's SHA-256 is pinned |
| Training and serving diverging | Feature steps live in the pipeline. Batch inference reuses training QC |
| Bad or altered models in use | A promotion gate decides what is served, and bundle hashes are checked on load |
| Overconfident predictions | Conformal 90% intervals, wider at older predicted ages |
| Unreproducible runs | Pinned versions, seed and a manifest with data, config and code identity |
| Data shifting after deployment | `pagrus-age drift --input batch.csv` returns exit code 1 on an alert |

## Layout

```text
configs/default.yaml   every modelling decision
src/pagrus_age/
  data.py              download, contract, QC
  splits.py            landing-level splits
  models.py            baselines, candidates, leakage guard
  metrics.py           metrics, landing bootstrap, conformal, PSI
  train.py             selection, evaluation, gate, artefacts
  reporting.py         figures, generated model card
  predict.py           batch inference, drift
  serve.py             FastAPI
  cli.py               entry point
tests/                 run on synthetic data only
```

## Limits

- The target is the age read from scales, not true age. The authors found 53% exact and 40% one-year
  agreement between scales and otoliths on 89 fish.
- Accuracy cannot be measured on fish whose scales could not be read. Those are the main target.
- The data cover 1976-1985 and one port. Do not use this for current stock assessment.
- Only one test landing used handline, so the per-gear figure for handline is weak.
