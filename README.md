# battery-degradation-ml

ML pipeline predicting Li-ion battery **State-of-Health (SOH)** and **Remaining-Useful-Life (RUL)** from raw MATR cycling data — curve-feature engineering (dQ/dV, ΔQ(V)) and a 15-model bouquet with leakage-free, grouped cross-validation.

Dataset: the MATR fast-charging dataset (Severson et al., *Nature Energy* 2019) — 180 commercial LFP/graphite cells cycled to failure, ~10 GB of raw `.mat` files.

## Headline results

Curve features (mined from the raw voltage/capacity curves) improve every task well beyond the fold-to-fold noise. Scores are R² mean ± std under grouped-by-cell 5-fold CV (SOH/RUL) and 5-fold CV (early-EOL):

| Task | scalar features | + curve features |
|---|---|---|
| SOH soft-sensor (Extra Trees) | 0.750 ± 0.035 | **0.982 ± 0.012** |
| RUL (XGBoost) | 0.654 ± 0.071 | **0.838 ± 0.054** |
| Early-cycle EOL (Random Forest) | 0.572 ± 0.078 | **0.791 ± 0.091** |

The single Severson feature `log10(var(ΔQ(V)))` correlates **−0.84** with `log10(cycle_life)` on its own.

## Repository layout

| File | What it does |
|---|---|
| `download_matr.sh` | Downloads the 4 raw MATR `.mat` files (~10 GB) from data.matr.io |
| `preprocess_matr_summary.py` | Parses the `.mat` files into a tidy per-cycle summary table |
| `battery_curve_features.py` | Mines per-cycle curve-shape features (Qdlin/dQ-dV) and per-cell Severson ΔQ(V) features |
| `battery_bouquet_augmented.py` | **Main pipeline** — scalar vs scalar+curve comparison with leakage-free grouped CV and error bars |
| `battery_model_bouquet.py` | Original single-split 15-model bouquet (incl. LSTM/GRU/CNN-LSTM) |
| `battery_insights.py` | Non-objective insights: early-cycle EOL screening, charge-time/temperature effects, knee detection |
| `MATR_cell_dQ_features.csv` | Per-cell ΔQ(V) feature table (180 cells) — enough to run the early-EOL analysis without the raw data |
| `battery_augmented_results.csv`, `battery_model_results.csv`, `battery_insights_summary.csv` | Result tables |
| `battery_insights.png` | Multi-panel insights figure |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download the raw MATR data (~10 GB) — gitignored
bash download_matr.sh

# 2. Preprocess -> tidy per-cycle table
python preprocess_matr_summary.py

# 3. Extract curve features (per-cycle + per-cell ΔQ(V))
python battery_curve_features.py

# 4. Main results: scalar vs scalar+curve, leakage-free grouped CV
python battery_bouquet_augmented.py

# (optional) original bouquet and insights
python battery_model_bouquet.py
python battery_insights.py
```

## Notes

- The raw `.mat` files and the large regenerable intermediates (`MATR_augmented.csv`, `MATR_battery_summary.csv`) are **gitignored** — regenerate them with the steps above.
- SOH is modelled as a **soft-sensor**: capacity-level features are excluded so the model infers health from the *shape* of the discharge/dQ-dV curve, not from the capacity it is trying to predict.
- Evaluation uses GroupKFold **by cell** so no cell appears in both train and test, with preprocessing fit inside each CV fold (no leakage).

## Data license / citation

MATR data is released under CC BY 4.0. If you use it, cite:
Severson, K. A., et al. "Data-driven prediction of battery cycle life before capacity degradation." *Nature Energy* 4, 383–391 (2019).
