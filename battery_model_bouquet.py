# First pass at SOH/RUL on the MATR data - a big model bouquet on a single
# 80/20 split by cell. Kept for reference; battery_bouquet_augmented.py is the
# proper evaluation (grouped CV, no leakage) and the curve features.
# Note: MATR only gives one resistance value (IR) per cycle, so there's no Rct.
# LSTM/GRU/CNN-LSTM are in PyTorch. Results -> battery_model_results.csv.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (IsolationForest, RandomForestRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

RAW_DIR = Path(__file__).parent
DATA_CSV = RAW_DIR / 'MATR_battery_summary.csv'
RESULTS_CSV = RAW_DIR / 'battery_model_results.csv'
NOMINAL_CAPACITY = 1.1  # Ah -- same nominal capacity as the NASA/Kaggle set
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---- load, clean, feature engineering ----
def load_and_engineer():
    df = pd.read_csv(DATA_CSV)
    df = df.drop_duplicates()
    df['battery_id'] = df['battery_id'].astype('category')
    df['test_id'] = pd.to_numeric(df['test_id'], errors='coerce')

    # drop outliers with IsolationForest (no Rct column in MATR)
    numerical_features = ['ambient_temperature', 'Capacity', 'Re']
    df = df.dropna(subset=numerical_features)
    iso = IsolationForest(contamination=0.05, random_state=SEED)
    keep = iso.fit_predict(df[numerical_features]) == 1
    df = df[keep].copy()

    df['Capacity_diff'] = df.groupby('battery_id', observed=True)['Capacity'].diff().fillna(0)
    df['Cycle_diff'] = df.groupby('battery_id', observed=True)['test_id'].diff().fillna(1)
    df['Degradation_rate'] = df['Capacity_diff'] / df['Cycle_diff'].replace(0, 1)

    df['Capacity_rolling_mean_5'] = df.groupby('battery_id', observed=True)['Capacity'] \
        .rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Capacity_rolling_mean_10'] = df.groupby('battery_id', observed=True)['Capacity'] \
        .rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
    df['Capacity_rolling_std_5'] = df.groupby('battery_id', observed=True)['Capacity'] \
        .rolling(5, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    df['Capacity_rolling_std_10'] = df.groupby('battery_id', observed=True)['Capacity'] \
        .rolling(10, min_periods=1).std().fillna(0).reset_index(level=0, drop=True)

    df['Capacity_lag_1'] = df.groupby('battery_id', observed=True)['Capacity'].shift(1).bfill()
    df['Capacity_lag_2'] = df.groupby('battery_id', observed=True)['Capacity'].shift(2).bfill()
    df['Cumulative_cycle_count'] = df.groupby('battery_id', observed=True).cumcount() + 1
    df['Capacity_fade_5'] = df.groupby('battery_id', observed=True)['Capacity'].diff(periods=5).fillna(0)

    df['SOH'] = df['Capacity'] / NOMINAL_CAPACITY

    scaling_features = [
        'ambient_temperature', 'Re', 'Degradation_rate', 'Capacity_diff',
        'Capacity_rolling_mean_5', 'Capacity_rolling_mean_10',
        'Capacity_rolling_std_5', 'Capacity_rolling_std_10',
        'Capacity_lag_1', 'Capacity_lag_2', 'Cumulative_cycle_count', 'Capacity_fade_5'
    ]
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[scaling_features])
    scaled_df = pd.DataFrame(scaled, columns=scaling_features, index=df.index)
    scaled_df['Capacity'] = df['Capacity']
    scaled_df['SOH'] = df['SOH']
    scaled_df['battery_id'] = df['battery_id']
    scaled_df['test_id'] = df['test_id']
    return scaled_df


# ---- SOH bouquet ----
def run_soh_bouquet(df):
    features = ['ambient_temperature', 'Re', 'test_id']  # 'Rct' dropped (unavailable in MATR)
    target = 'SOH'

    battery_ids = df['battery_id'].unique()
    train_ids, test_ids = train_test_split(battery_ids, test_size=0.2, random_state=SEED)

    train_mask = df['battery_id'].isin(train_ids)
    X_train, y_train = df.loc[train_mask, features].dropna(), df.loc[train_mask, target]
    y_train = y_train.loc[X_train.index]
    X_test, y_test = df.loc[~train_mask, features].dropna(), df.loc[~train_mask, target]
    y_test = y_test.loc[X_test.index]

    # throw a bunch of regressors at it. SVR/GPR don't scale to ~117k rows so
    # they live in battery_insights.py instead (cell-level, n=178).
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.001, random_state=SEED, max_iter=5000),
        'PLS Regression': PLSRegression(n_components=3),
        'KNN': KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=SEED),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=300, random_state=SEED, verbose=0),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu',
                                       solver='adam', max_iter=500, random_state=SEED),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test)).ravel()
        results.append({
            'Task': 'SOH', 'Model': name,
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred),
        })
        print(f"[SOH] {name:22s} MAE={results[-1]['MAE']:.5f}  "
              f"RMSE={results[-1]['RMSE']:.5f}  R2={results[-1]['R2']:.4f}")
    return pd.DataFrame(results)


# ---- RUL target + sequence building ----
RUL_FEATURES = ['ambient_temperature', 'Re', 'Degradation_rate', 'SOH']  # 'Rct' dropped


def build_rul_table(df, threshold_soh=0.8):
    rows = []
    for bid in df['battery_id'].unique():
        bdata = df[df['battery_id'] == bid].sort_values('test_id')
        below = bdata[bdata['SOH'] <= threshold_soh]['test_id']
        end_cycle = below.min() if len(below) else bdata['test_id'].max()
        rul = (end_cycle - bdata['test_id']).clip(lower=0)
        chunk = bdata[['battery_id', 'test_id'] + RUL_FEATURES].copy()
        chunk['RUL'] = rul.values
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def create_sequences(data, seq_length=10):
    sequences, targets = [], []
    for bid in data['battery_id'].unique():
        bdata = data[data['battery_id'] == bid].sort_values('test_id')
        X = bdata[RUL_FEATURES].values.astype('float32')
        y = bdata['RUL'].values.astype('float32')
        if len(bdata) > seq_length:
            for i in range(len(bdata) - seq_length):
                sequences.append(X[i:i + seq_length])
                targets.append(y[i + seq_length])
    return np.array(sequences, dtype='float32'), np.array(targets, dtype='float32')


# ---- torch models (BiLSTM / BiGRU / CNN-LSTM) ----
class BiLSTMNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(self.bn1(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.drop2(self.bn2(x))
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class BiGRUNet(nn.Module):
    def __init__(self, n_features, units=64):
        super().__init__()
        self.gru1 = nn.GRU(n_features, units, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(units * 2)
        self.drop1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(units * 2, 16, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.drop1(self.bn1(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.bn2(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(64, 100, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(200, 50, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        return self.fc(x).squeeze(-1)


def train_torch_model(model, X_train, y_train, epochs=15, batch_size=256, lr=5e-4):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        print(f"    epoch {epoch + 1}/{epochs}  train_MSE={total_loss / len(ds):.4f}")
    return model


@torch.no_grad()
def torch_predict(model, X, batch_size=1024):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


# ---- RUL bouquet ----
def run_rul_bouquet(df, seq_length=10, epochs=15):
    rul_df = build_rul_table(df)
    battery_ids = rul_df['battery_id'].unique()
    train_ids, test_ids = train_test_split(battery_ids, test_size=0.2, random_state=SEED)
    train_df = rul_df[rul_df['battery_id'].isin(train_ids)]
    test_df = rul_df[rul_df['battery_id'].isin(test_ids)]

    # flat (non-sequence) regressors on single-cycle RUL features
    X_train_flat, y_train_flat = train_df[RUL_FEATURES], train_df['RUL']
    X_test_flat, y_test_flat = test_df[RUL_FEATURES], test_df['RUL']

    flat_models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        'LightGBM': LGBMRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=300, random_state=SEED, verbose=0),
        'XGBoost': XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            random_state=SEED, n_jobs=-1),
    }
    results = []
    xgb_model = None
    for name, model in flat_models.items():
        model.fit(X_train_flat, y_train_flat)
        pred = np.asarray(model.predict(X_test_flat)).ravel()
        results.append({
            'Task': 'RUL', 'Model': name,
            'MAE': mean_absolute_error(y_test_flat, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_flat, pred)),
            'R2': r2_score(y_test_flat, pred),
        })
        print(f"[RUL] {name:22s} MAE={results[-1]['MAE']:.3f}  "
              f"RMSE={results[-1]['RMSE']:.3f}  R2={results[-1]['R2']:.4f}")
        if name == 'XGBoost':
            xgb_model = model  # reused below for ensemble / stacking

    # --- Sequence models: LSTM, GRU, CNN-LSTM ---
    X_train_seq, y_train_seq = create_sequences(train_df, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_df, seq_length)
    n_features = len(RUL_FEATURES)

    seq_models = {
        'LSTM': BiLSTMNet(n_features),
        'GRU': BiGRUNet(n_features),
        'CNN-LSTM': CNNLSTMNet(n_features),
    }
    seq_preds = {}
    for name, net in seq_models.items():
        print(f"  Training {name} ...")
        net = train_torch_model(net, X_train_seq, y_train_seq, epochs=epochs)
        pred = torch_predict(net, X_test_seq)
        seq_preds[name] = pred
        results.append({
            'Task': 'RUL', 'Model': name,
            'MAE': mean_absolute_error(y_test_seq, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_seq, pred)),
            'R2': r2_score(y_test_seq, pred),
        })
        print(f"[RUL] {name:20s} MAE={results[-1]['MAE']:.3f}  "
              f"RMSE={results[-1]['RMSE']:.3f}  R2={results[-1]['R2']:.4f}")

    # weighted ensemble (GRU 0.4 / CNN-LSTM 0.3 / XGB 0.3). align XGB preds to
    # the sequence test set using the last row of each window
    pred_xgb_seq = xgb_model.predict(pd.DataFrame(X_test_seq[:, -1, :], columns=RUL_FEATURES))
    ensemble_pred = 0.4 * seq_preds['GRU'] + 0.3 * seq_preds['CNN-LSTM'] + 0.3 * pred_xgb_seq
    results.append({
        'Task': 'RUL', 'Model': 'Ensemble (weighted avg)',
        'MAE': mean_absolute_error(y_test_seq, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_seq, ensemble_pred)),
        'R2': r2_score(y_test_seq, ensemble_pred),
    })
    print(f"[RUL] {'Ensemble (weighted avg)':20s} MAE={results[-1]['MAE']:.3f}  "
          f"RMSE={results[-1]['RMSE']:.3f}  R2={results[-1]['R2']:.4f}")

    # stacking: LinearRegression meta-model on a validation slice of the
    # training sequences
    val_idx = int(len(X_train_seq) * 0.8)
    X_meta_train_seq = X_train_seq[val_idx:]
    y_meta_train_true = y_train_seq[val_idx:]
    X_meta_train_xgb = pd.DataFrame(X_meta_train_seq[:, -1, :], columns=RUL_FEATURES)

    meta_models = {name: seq_models[name] for name in ['GRU', 'CNN-LSTM']}
    pred_xgb_val = xgb_model.predict(X_meta_train_xgb)
    pred_gru_val = torch_predict(meta_models['GRU'], X_meta_train_seq)
    pred_cnn_val = torch_predict(meta_models['CNN-LSTM'], X_meta_train_seq)
    X_meta_train = np.column_stack([pred_xgb_val, pred_gru_val, pred_cnn_val])

    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_meta_train_true)

    X_meta_test = np.column_stack([pred_xgb_seq, seq_preds['GRU'], seq_preds['CNN-LSTM']])
    stacking_pred = meta_model.predict(X_meta_test)
    results.append({
        'Task': 'RUL', 'Model': 'Stacking (meta LinearRegression)',
        'MAE': mean_absolute_error(y_test_seq, stacking_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_seq, stacking_pred)),
        'R2': r2_score(y_test_seq, stacking_pred),
    })
    print(f"[RUL] {'Stacking (meta LinearRegression)':20s} "
          f"MAE={results[-1]['MAE']:.3f}  RMSE={results[-1]['RMSE']:.3f}  "
          f"R2={results[-1]['R2']:.4f}")

    return pd.DataFrame(results)


# ---- main ----
def main():
    print(f'Loading and engineering features from {DATA_CSV} ...')
    df = load_and_engineer()
    print(f'{len(df):,} cycles across {df.battery_id.nunique()} cells after cleaning.\n')

    print('=== SOH model bouquet ===')
    soh_results = run_soh_bouquet(df)

    print('\n=== RUL model bouquet ===')
    rul_results = run_rul_bouquet(df)

    all_results = pd.concat([soh_results, rul_results], ignore_index=True)
    all_results.to_csv(RESULTS_CSV, index=False)

    print('\n=== FINAL LEADERBOARD (by R2, higher is better) ===')
    for task in ['SOH', 'RUL']:
        sub = all_results[all_results['Task'] == task].sort_values('R2', ascending=False)
        print(f'\n-- {task} --')
        print(sub.to_string(index=False))
        best = sub.iloc[0]
        print(f'>>> Best {task} model: {best.Model}  (R2={best.R2:.4f}, '
              f'RMSE={best.RMSE:.4f}, MAE={best.MAE:.4f})')

    print(f'\nFull results written to {RESULTS_CSV}')


if __name__ == '__main__':
    main()
