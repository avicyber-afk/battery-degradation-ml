# Extra analysis beyond SOH/RUL - stuff the dataset supports that's handy for
# a manufacturing / digital-twin angle:
#   1. early-cycle EOL prediction (Severson: cycle life from the first 100
#      cycles). SVR/GPR run here since n = ~178 cells, not 117k rows.
#   2. charge time vs cycle life
#   3. temperature vs degradation rate
#   4. IR growth as an ageing signal
#   5. cell-to-cell variability
#   6. knee-point detection
# Writes battery_insights_summary.csv + battery_insights.png.

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

RAW_DIR = Path(__file__).parent
DATA_CSV = RAW_DIR / 'MATR_battery_summary.csv'
OUT_CSV = RAW_DIR / 'battery_insights_summary.csv'
OUT_PNG = RAW_DIR / 'battery_insights.png'
NOMINAL = 1.1
EOL_SOH = 0.8          # end-of-life threshold (80% of nominal capacity)
EARLY_WINDOW = 100     # cycles of early-life data used for screening


def load():
    df = pd.read_csv(DATA_CSV)
    df = df[df['Capacity'] > 0].copy()
    df['SOH'] = df['Capacity'] / NOMINAL
    return df


# ---------------------------------------------------------------------------
# Per-cell aggregate features (one row per cell) for the screening tasks
# ---------------------------------------------------------------------------
def build_cell_table(df):
    rows = []
    for bid, g in df.groupby('battery_id'):
        g = g.sort_values('test_id')
        cl = g['cycle_life'].dropna()
        eol_reported = float(cl.iloc[0]) if len(cl) else np.nan

        # Observed EOL: first cycle at/below 80% SOH (fallback: last cycle seen)
        below = g[g['SOH'] <= EOL_SOH]['test_id']
        eol_obs = float(below.min()) if len(below) else float(g['test_id'].max())

        early = g[g['test_id'] <= EARLY_WINDOW]
        if len(early) < 20:      # need enough early cycles to characterise
            continue

        # Early-life descriptors (all computed from first 100 cycles only)
        cap_early = early['Capacity'].values
        soh_early = early['SOH'].values
        ir_early = early['Re'].replace(0, np.nan).dropna().values

        cap_fade_early = cap_early[0] - cap_early[-1]           # Ah lost in 100 cyc
        cap_slope = np.polyfit(early['test_id'], cap_early, 1)[0]  # Ah/cycle
        cap_var = np.var(soh_early)                             # variance of SOH
        cap_min = cap_early.min()
        cap_mean = cap_early.mean()
        ir_mean = ir_early.mean() if len(ir_early) else np.nan
        ir_slope = (np.polyfit(range(len(ir_early)), ir_early, 1)[0]
                    if len(ir_early) > 5 else np.nan)
        temp_mean = early['ambient_temperature'].replace(0, np.nan).mean()
        chargetime_mean = early['chargetime'].replace(0, np.nan).mean()

        rows.append(dict(
            battery_id=bid, batch=g['batch'].iloc[0],
            eol_reported=eol_reported, eol_observed=eol_obs,
            cap_fade_early=cap_fade_early, cap_slope=cap_slope,
            cap_var_early=cap_var, cap_min_early=cap_min, cap_mean_early=cap_mean,
            ir_mean_early=ir_mean, ir_slope_early=ir_slope,
            temp_mean=temp_mean, chargetime_mean=chargetime_mean,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Insight 1: early-cycle EOL prediction (n = cells)
# ---------------------------------------------------------------------------
def insight_early_eol(cells):
    feat_cols = ['cap_fade_early', 'cap_slope', 'cap_var_early', 'cap_min_early',
                 'cap_mean_early', 'ir_mean_early', 'ir_slope_early',
                 'temp_mean', 'chargetime_mean']
    d = cells.dropna(subset=feat_cols + ['eol_observed']).copy()
    X = d[feat_cols].values
    y = d['eol_observed'].values

    # log10(EOL) is the Severson target -- it linearises the variance model
    y_log = np.log10(y)

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
    models = {
        'ElasticNet (variance model)': make_pipeline(
            StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
        'SVR (RBF)': make_pipeline(StandardScaler(), SVR(C=10, gamma='scale')),
        'GaussianProcess': make_pipeline(
            StandardScaler(), GaussianProcessRegressor(kernel=kernel, alpha=1e-3,
                                                       normalize_y=True, random_state=42)),
        'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Guard against a linear model extrapolating a wild log10(EOL) on an
    # out-of-fold cell (10**large -> astronomical cycle count). Real cell
    # lives here are ~148-2238 cycles, so clip predictions to a generous
    # bounded window before exponentiating.
    log_lo, log_hi = np.log10(100), np.log10(4000)
    out = []
    print(f"\n[Insight 1] Early-cycle EOL prediction "
          f"(first {EARLY_WINDOW} cycles -> total cycle life), n={len(d)} cells")
    for name, m in models.items():
        pred_log = np.clip(cross_val_predict(m, X, y_log, cv=cv), log_lo, log_hi)
        pred = 10 ** pred_log
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y_log, pred_log)
        # % of cells whose predicted life is within 15% of true life
        within15 = np.mean(np.abs(pred - y) / y <= 0.15) * 100
        out.append(dict(Insight='early_eol', Model=name, MAE_cycles=mae,
                        R2_log=r2, pct_within_15pct=within15))
        print(f"  {name:28s} MAE={mae:6.1f} cyc   R2(log)={r2:.3f}   "
              f"within-15%={within15:4.1f}%")
    return pd.DataFrame(out), d, feat_cols


# ---------------------------------------------------------------------------
# Insight 2-5: descriptive correlations / effects
# ---------------------------------------------------------------------------
def insight_effects(cells):
    print("\n[Insight 2] Fast-charging protocol effect (early mean chargetime vs life)")
    c = cells.dropna(subset=['chargetime_mean', 'eol_observed'])
    r_ct = np.corrcoef(c['chargetime_mean'], c['eol_observed'])[0, 1]
    print(f"  Pearson r(chargetime, EOL) = {r_ct:+.3f}  "
          f"(shorter charge time -> {'longer' if r_ct>0 else 'shorter'} life)")

    print("\n[Insight 3] Operating-temperature effect on degradation rate")
    c = cells.dropna(subset=['temp_mean', 'cap_slope'])
    r_temp = np.corrcoef(c['temp_mean'], np.abs(c['cap_slope']))[0, 1]
    print(f"  Pearson r(temp, |capacity fade slope|) = {r_temp:+.3f}")

    print("\n[Insight 4] Internal-resistance growth as an ageing signal")
    c = cells.dropna(subset=['ir_slope_early', 'eol_observed'])
    r_ir = np.corrcoef(c['ir_slope_early'], c['eol_observed'])[0, 1]
    print(f"  Pearson r(early IR growth slope, EOL) = {r_ir:+.3f}")

    print("\n[Insight 5] Cell-to-cell manufacturing variability")
    eol = cells['eol_observed'].dropna()
    print(f"  EOL spread across {len(eol)} cells: "
          f"min={eol.min():.0f}  median={eol.median():.0f}  max={eol.max():.0f}  "
          f"cycles  (max/min = {eol.max()/eol.min():.1f}x)")
    print(f"  Coefficient of variation of EOL = {eol.std()/eol.mean()*100:.1f}%")
    return pd.DataFrame([
        dict(Insight='corr_chargetime_eol', value=r_ct),
        dict(Insight='corr_temp_fade', value=r_temp),
        dict(Insight='corr_ir_growth_eol', value=r_ir),
        dict(Insight='eol_cv_pct', value=eol.std()/eol.mean()*100),
        dict(Insight='eol_max_over_min', value=eol.max()/eol.min()),
    ])


# ---------------------------------------------------------------------------
# Insight 6: knee-point detection (Bacon-Watts style, simplified)
# ---------------------------------------------------------------------------
def insight_knee(df):
    print("\n[Insight 6] Capacity-fade knee-point (cycle where fade accelerates)")
    knees = []
    for bid, g in df.groupby('battery_id'):
        g = g.sort_values('test_id')
        soh = g['SOH'].values
        cyc = g['test_id'].values
        if len(g) < 100:
            continue
        # Knee via the "maximum distance to the chord" (Kneedle-style) method,
        # which is far more robust than a raw 2nd-derivative peak: smooth the
        # SOH curve, draw the straight line from first to last point, and take
        # the cycle of greatest vertical departure below that chord. Restricted
        # to 15-90% of life so early formation transients and the final EOL
        # cliff don't masquerade as the knee.
        s = pd.Series(soh).rolling(21, min_periods=1, center=True).mean().values
        n = len(s)
        chord = np.linspace(s[0], s[-1], n)
        # For a decreasing-convex fade curve (flat plateau then a late cliff,
        # typical of these LFP fast-charge cells) the knee is the "shoulder"
        # where the smoothed curve bulges most ABOVE the first->last chord.
        dist = s - chord
        lo, hi = int(0.15 * n), int(0.90 * n)
        if hi - lo < 5:
            continue
        knee_idx = int(np.argmax(dist[lo:hi]) + lo)
        eol = cyc[-1]
        knees.append(cyc[knee_idx] / eol if eol else np.nan)
    knees = np.array([k for k in knees if not np.isnan(k)])
    print(f"  Knee typically onsets at {np.median(knees)*100:.0f}% of total life "
          f"(IQR {np.percentile(knees,25)*100:.0f}-{np.percentile(knees,75)*100:.0f}%), "
          f"n={len(knees)} cells")
    return pd.DataFrame([dict(Insight='knee_frac_of_life_median', value=float(np.median(knees)))])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(cells, d, feat_cols):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].hist(cells['eol_observed'].dropna(), bins=30, color='steelblue', edgecolor='k')
    ax[0, 0].set(title='Cell-to-cell EOL variability', xlabel='Cycles to 80% SOH', ylabel='# cells')

    c = cells.dropna(subset=['chargetime_mean', 'eol_observed'])
    ax[0, 1].scatter(c['chargetime_mean'], c['eol_observed'], alpha=0.5, color='darkorange')
    ax[0, 1].set(title='Charge time vs life', xlabel='Mean early charge time (min)', ylabel='EOL (cycles)')

    c = cells.dropna(subset=['temp_mean', 'cap_slope'])
    ax[0, 2].scatter(c['temp_mean'], np.abs(c['cap_slope']), alpha=0.5, color='firebrick')
    ax[0, 2].set(title='Temperature vs fade rate', xlabel='Mean temp (C)', ylabel='|capacity fade| (Ah/cyc)')

    # Predicted vs actual EOL (GP, via CV) for the screening panel
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    X = d[feat_cols].values
    y = d['eol_observed'].values
    pred = cross_val_predict(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
                             X, np.log10(y), cv=KFold(5, shuffle=True, random_state=42))
    pred = 10 ** pred
    ax[1, 0].scatter(y, pred, alpha=0.6, color='green')
    lims = [y.min(), y.max()]
    ax[1, 0].plot(lims, lims, 'r--')
    ax[1, 0].set(title='Early-cycle EOL prediction (RF, 5-fold CV)',
                 xlabel='Actual EOL (cycles)', ylabel='Predicted EOL (cycles)')

    c = cells.dropna(subset=['ir_slope_early', 'eol_observed'])
    ax[1, 1].scatter(c['ir_slope_early'], c['eol_observed'], alpha=0.5, color='purple')
    ax[1, 1].set(title='Early IR growth vs life', xlabel='Early IR slope (ohm/cyc)', ylabel='EOL (cycles)')

    # A few representative degradation curves
    for bid in cells['battery_id'].dropna().iloc[:6]:
        pass
    ax[1, 2].axis('off')
    ax[1, 2].text(0.02, 0.98, 'MATR battery insights\n(non-objective analysis)\n\n'
                  'Panels: EOL spread, charge-time &\ntemperature effects, early-cycle\n'
                  'EOL screening, IR-growth signal.',
                  va='top', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nFigure written to {OUT_PNG}")


def main():
    df = load()
    cells = build_cell_table(df)
    print(f"Built per-cell table: {len(cells)} cells with >=20 early cycles.")

    r1, d, feat_cols = insight_early_eol(cells)
    r2 = insight_effects(cells)
    r6 = insight_knee(df)

    summary = pd.concat([r1, r2, r6], ignore_index=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nSummary written to {OUT_CSV}")

    make_figure(cells, d, feat_cols)


if __name__ == '__main__':
    main()
