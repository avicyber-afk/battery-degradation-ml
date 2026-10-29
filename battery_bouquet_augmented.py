# SOH / RUL / early-EOL, scalar features vs scalar + curve features.
# GroupKFold by cell + preprocessing inside the pipeline so nothing leaks.
# Results go to battery_augmented_results.csv (see README for the numbers).

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
RAW = Path(__file__).parent
PERCYCLE = RAW / 'MATR_augmented.csv'
PERCELL = RAW / 'MATR_cell_dQ_features.csv'
RESULTS = RAW / 'battery_augmented_results.csv'
NOMINAL = 1.1
SEED = 42

# feature blocks (per-cycle table)
SCALAR_COMMON = ['ambient_temperature', 'Re', 'chargetime', 'Tmax', 'Tmin', 'test_id']
CURVE_SHAPE = ['dQdV_min', 'dQdV_peak_V', 'dQdV_var', 'Qdlin_var', 'Qdlin_range']
CURVE_LEVEL = ['Qdlin_at_2.0V', 'Qdlin_at_2.5V', 'Qdlin_at_3.0V', 'Qdlin_at_3.3V',
               'Qdlin_min', 'Qdlin_max']


def per_cycle_models():
    return {
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=SEED),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1),
    }


def wrap(model, scale):
    steps = [SimpleImputer(strategy='median')]
    if scale:
        steps.append(StandardScaler())
    steps.append(model)
    return make_pipeline(*steps)


def cv_report(X, y, groups, models, feats_scalar, feats_curve, cv, task):
    """For each model, CV-score scalar-only and scalar+curve; return rows."""
    rows = []
    for name, base in models.items():
        scale = name in ('Ridge', 'SVR', 'GaussianProcess', 'ElasticNet')
        for label, cols in [('scalar', feats_scalar), ('scalar+curve', feats_curve)]:
            Xc = X[cols].values
            # n_jobs=1 here, let the models use the cores (nested jobs thrash)
            cvres = cross_validate(wrap(base, scale), Xc, y, cv=cv, groups=groups,
                                   scoring=['r2', 'neg_mean_absolute_error'], n_jobs=1)
            r2 = cvres['test_r2']
            mae = -cvres['test_neg_mean_absolute_error']
            rows.append({'Task': task, 'Model': name, 'Features': label,
                         'R2_mean': r2.mean(), 'R2_std': r2.std(),
                         'MAE_mean': mae.mean(), 'MAE_std': mae.std()})
        s = rows[-2]; c = rows[-1]
        print(f"  {name:22s} scalar R2={s['R2_mean']:.3f}+/-{s['R2_std']:.3f}  "
              f"| +curve R2={c['R2_mean']:.3f}+/-{c['R2_std']:.3f}  "
              f"| dR2={c['R2_mean']-s['R2_mean']:+.3f}")
    return rows


def engineer(df):
    df = df[df['Capacity'] > 0].copy()
    df['SOH'] = df['Capacity'] / NOMINAL
    df['Capacity_diff'] = df.groupby('battery_id')['Capacity'].diff().fillna(0)
    df['Cycle_diff'] = df.groupby('battery_id')['test_id'].diff().fillna(1)
    df['Degradation_rate'] = df['Capacity_diff'] / df['Cycle_diff'].replace(0, 1)
    return df


def build_rul(df, thr=0.8):
    parts = []
    for bid, g in df.groupby('battery_id'):
        g = g.sort_values('test_id')
        below = g[g['SOH'] <= thr]['test_id']
        eol = below.min() if len(below) else g['test_id'].max()
        gg = g.copy(); gg['RUL'] = (eol - gg['test_id']).clip(lower=0)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def main():
    df = engineer(pd.read_csv(PERCYCLE))
    print(f'Loaded augmented per-cycle table: {df.shape[0]} rows, {df.battery_id.nunique()} cells\n')
    gkf = GroupKFold(n_splits=5)
    all_rows = []

    # Task A: SOH. drop capacity-level features so it's an actual soft-sensor
    print('[Task A] SOH soft-sensor  (scalar vs scalar + IC-curve SHAPE)')
    soh_scalar = ['ambient_temperature', 'Re', 'chargetime', 'Tmax', 'Tmin', 'test_id']
    soh_curve = soh_scalar + CURVE_SHAPE
    all_rows += cv_report(df, df['SOH'].values, df['battery_id'].values,
                          per_cycle_models(), soh_scalar, soh_curve, gkf, 'SOH')

    # Task B: RUL
    print('\n[Task B] RUL  (scalar vs scalar + curve shape)')
    rul_df = build_rul(df)
    rul_scalar = ['ambient_temperature', 'Re', 'Degradation_rate', 'SOH', 'test_id']
    rul_curve = rul_scalar + CURVE_SHAPE
    all_rows += cv_report(rul_df, rul_df['RUL'].values, rul_df['battery_id'].values,
                          per_cycle_models(), rul_scalar, rul_curve, gkf, 'RUL')

    # Task C: early-cycle EOL, cell-level (Severson dQ(V))
    print('\n[Task C] Early-cycle EOL  (cell-level; scalar vs scalar + delta-Q(V))')
    cell = pd.read_csv(PERCELL)
    # per-cell scalars from the first 100 cycles
    early = df[df['test_id'] <= 100]
    agg = early.groupby('battery_id').agg(
        cap0=('Capacity', 'first'), cap_min=('Capacity', 'min'),
        cap_slope=('Capacity', lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) > 2 else 0),
        ir_mean=('Re', 'mean'), chargetime_mean=('chargetime', 'mean'),
        temp_mean=('ambient_temperature', 'mean')).reset_index()
    cell = cell.merge(agg, on='battery_id', how='inner')
    cell = cell.dropna(subset=['dQ_var_log', 'cycle_life'])
    y = np.log10(cell['cycle_life'].values)          # predict log10 cycle life
    cell_scalar = ['cap0', 'cap_min', 'cap_slope', 'ir_mean', 'chargetime_mean', 'temp_mean']
    dq_cols = ['dQ_var_log', 'dQ_min_log', 'dQ_mean_log', 'dQ_skew', 'dQ_kurt', 'dQ_at_2V_log']
    cell_curve = cell_scalar + dq_cols
    # isotropic RBF - works for both the 6 and 12 feature sets (anisotropic
    # would need a fixed dim)
    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel()
    cell_models = {
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=SEED),
        'SVR': SVR(C=10, gamma='scale'),
        'GaussianProcess': GaussianProcessRegressor(kernel=kernel, alpha=1e-3,
                                                    normalize_y=True, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, base in cell_models.items():
        scale = name in ('SVR', 'GaussianProcess', 'ElasticNet')
        for label, cols in [('scalar', cell_scalar), ('scalar+dQ', cell_curve)]:
            Xc = cell[cols].values
            cvres = cross_validate(wrap(base, scale), Xc, y, cv=kf,
                                   scoring=['r2', 'neg_mean_absolute_error'], n_jobs=1)
            r2 = cvres['test_r2']
            mae = -cvres['test_neg_mean_absolute_error']
            all_rows.append({'Task': 'EarlyEOL', 'Model': name, 'Features': label,
                             'R2_mean': r2.mean(), 'R2_std': r2.std(),
                             'MAE_mean': mae.mean(), 'MAE_std': mae.std()})
        s = all_rows[-2]; c = all_rows[-1]
        print(f"  {name:22s} scalar R2={s['R2_mean']:.3f}+/-{s['R2_std']:.3f}  "
              f"| +dQ R2={c['R2_mean']:.3f}+/-{c['R2_std']:.3f}  "
              f"| dR2={c['R2_mean']-s['R2_mean']:+.3f}")

    res = pd.DataFrame(all_rows)
    res.to_csv(RESULTS, index=False)
    print('\n=== FEATURE-ENGINEERING IMPACT (best model per task, curve vs scalar) ===')
    for task in ['SOH', 'RUL', 'EarlyEOL']:
        t = res[res.Task == task]
        best_curve = t[t.Features.str.contains('curve|dQ')].sort_values('R2_mean').iloc[-1]
        base = t[(t.Model == best_curve.Model) & (~t.Features.str.contains('curve|dQ'))].iloc[0]
        print(f"  {task:9s}: {best_curve.Model:20s} "
              f"scalar R2={base.R2_mean:.3f} -> +curve R2={best_curve.R2_mean:.3f} "
              f"(dR2={best_curve.R2_mean-base.R2_mean:+.3f})")
    print(f'\nFull results -> {RESULTS}')


if __name__ == '__main__':
    main()
