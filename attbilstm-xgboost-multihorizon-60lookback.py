"""
=============================================================
Thesis Forecasting Pipeline
Att-BiLSTM + XGBoost hybrid model
3 models (A, B, C) x 4 horizons (1, 7, 14, 30) = 12 experiments
Look-back window: 60 days
=============================================================
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:29:59 2026

@author: ilsem
"""

=============================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.layers import (Input, Bidirectional, LSTM,
                                     Dense, Dropout, Attention)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ============================================================
# 1. CONFIGURATION
# ============================================================

HORIZONS      = [1, 7, 14, 30]
MODEL_TYPES   = ['a', 'b', 'c']
TARGET_COLUMN = 'sales_volume'
GROUP_COLUMN  = 'category_num'
TIME_STEPS    = 60

DATA_PATH     = r"C:\Users\ilsem\Documents\Thesis - memoire\dataset"
OUTPUT_FOLDER = r"C:\Users\ilsem\Documents\Thesis - memoire\forecast_outputs3"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Numeric feature columns to normalise with the shared x_scaler.
# day / month / year / holiday are left as-is (date components or binary).
# category_num is excluded from features entirely.
FEATURES_TO_SCALE = [
    'avg_price',
    'avg_sentiment_score',
    'avg_review_score',
    'review_volume',
    'avg_has_text_review'
]


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================

def calculate_wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error."""
    return (np.sum(np.abs(y_true - y_pred)) /
            (np.sum(np.abs(y_true)) + 1e-10)) * 100


def calculate_nse(y_true, y_pred):
    """Nash-Sutcliffe Efficiency (1 = perfect, <0 = worse than the mean)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
    return 1 - ss_res / ss_tot


def calculate_me(y_true, y_pred):
    """Mean Error / bias. Positive = model under-predicts on average."""
    return float(np.mean(y_true - y_pred))


# ============================================================
# 3. SEQUENCE CREATION — grouped by category
# ============================================================

def create_sequences_by_category(df_scaled, y_scaled, steps, horizon,
                                  categories, feature_cols):
    """
    Build sliding-window sequences independently for each product category,
    then concatenate the results.

    Building windows per category avoids mixing unrelated time series —
    e.g. the last day of 'electronics' should never be the first day
    of a window that continues into 'toys'.

    Parameters
    ----------
    df_scaled   : DataFrame (already scaled) with at least feature_cols
    y_scaled    : 1-D numpy array of scaled target values, same row order
    steps       : look-back window length (TIME_STEPS = 60)
    horizon     : forecast horizon in days
    categories  : 1-D array of category_num values, same row order
    feature_cols: list of column names used as model input features

    Returns
    -------
    X : ndarray of shape (n_samples, steps, n_features)
    y : ndarray of shape (n_samples,)
    """
    X_list, y_list = [], []

    for cat in np.unique(categories):
        mask  = categories == cat
        X_cat = df_scaled.loc[mask, feature_cols].values
        y_cat = y_scaled[mask]

        # Minimum rows needed: steps + horizon
        limit = len(X_cat) - steps - (horizon - 1)
        if limit <= 0:
            continue

        for i in range(limit):
            X_list.append(X_cat[i: i + steps])
            y_list.append(y_cat[i + steps + horizon - 1])

    if not X_list:
        return np.array([]).reshape(0, steps, 1), np.array([])

    return np.array(X_list), np.array(y_list)


def create_sequences_by_category_with_meta(df_scaled, y_scaled, steps,
                                            horizon, categories,
                                            feature_cols, dates):
    """
    Same logic as create_sequences_by_category but also returns the
    forecast target date and category_num for each sample.
    These are needed to populate predictions_all.csv correctly.

    Returns
    -------
    X          : ndarray (n_samples, steps, n_features)
    y          : ndarray (n_samples,)
    date_list  : list of Timestamps — the date being predicted
    cat_list   : list of ints — corresponding category_num
    """
    X_list, y_list, date_list, cat_list = [], [], [], []

    for cat in np.unique(categories):
        idx   = np.where(categories == cat)[0]
        X_cat = df_scaled.iloc[idx][feature_cols].values
        y_cat = y_scaled[idx]
        d_cat = dates.iloc[idx].values

        limit = len(X_cat) - steps - (horizon - 1)
        if limit <= 0:
            continue

        for i in range(limit):
            X_list.append(X_cat[i: i + steps])
            target_idx = i + steps + horizon - 1
            y_list.append(y_cat[target_idx])
            date_list.append(d_cat[target_idx])
            cat_list.append(cat)

    if not X_list:
        return np.array([]), np.array([]), [], []

    return np.array(X_list), np.array(y_list), date_list, cat_list


# ============================================================
# 4. MODEL ARCHITECTURE
# ============================================================

def build_att_bilstm_model(steps, n_features):
    """
    2-layer Bidirectional LSTM with self-attention and L2 regularization.

    Architecture
    ------------
    Input  (steps, n_features)
      -> BiLSTM(64, L2=1e-4) + Dropout(0.2)
      -> BiLSTM(64, L2=1e-4) + Dropout(0.2)
      -> Self-Attention  [query = value = output of 2nd BiLSTM]
      -> Last time step of attended sequence  -> Dense(1, linear)

    L2 regularization is applied to both kernel and recurrent weights
    to reduce overfitting, especially visible in Model A (more features).

    Two outputs are returned: [prediction, attention_scores]
    so attention weights can be inspected after training.
    """
    inp = Input(shape=(steps, n_features))

    # First BiLSTM layer with L2 regularization
    x = Bidirectional(LSTM(64,
                           return_sequences=True,
                           kernel_regularizer=l2(1e-4),
                           recurrent_regularizer=l2(1e-4)))(inp)
    x = Dropout(0.2)(x)

    # Second BiLSTM layer with L2 regularization
    x = Bidirectional(LSTM(64,
                           return_sequences=True,
                           kernel_regularizer=l2(1e-4),
                           recurrent_regularizer=l2(1e-4)))(x)
    x = Dropout(0.2)(x)

    # Self-attention: the sequence attends to itself
    attended, att_scores = Attention()([x, x], return_attention_scores=True)

    # Use only the last attended time step as the context vector
    context = attended[:, -1, :]

    prediction = Dense(1, activation='linear')(context)

    model = Model(inputs=inp, outputs=[prediction, att_scores])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=['mse', None]   # only optimise the prediction head
    )
    return model


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

dataset_registry = {}
for m in MODEL_TYPES:
    for split in ['train', 'val', 'test']:
        fn   = f"{split}_model_{m}.csv"
        path = os.path.join(DATA_PATH, fn)
        df   = pd.read_csv(path)
        dataset_registry[f"{split}_{m}"] = df
        print(f"  Loaded {fn:30s} -> {df.shape}")


# ============================================================
# 6. STORAGE FOR ALL OUTPUTS
# ============================================================

all_predictions  = []   # rows that will form predictions_all.csv
results_log      = []   # rows that will form metrics_report.csv
attention_global = {}   # exp_id -> mean attention vector of length TIME_STEPS


# ============================================================
# 7. MAIN TRAINING & EVALUATION LOOP
# ============================================================

total_exps = len(MODEL_TYPES) * len(HORIZONS)   # 12
pbar = tqdm(total=total_exps, desc="Overall progress")

for m in MODEL_TYPES:

    print(f"\n{'='*60}")
    print(f"  MODEL {m.upper()}")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # 7a. Load raw splits for this model
    # ----------------------------------------------------------
    train_raw = dataset_registry[f'train_{m}'].copy()
    val_raw   = dataset_registry[f'val_{m}'].copy()
    test_raw  = dataset_registry[f'test_{m}'].copy()

    # All input feature columns (everything except the group identifier)
    all_feature_cols = [c for c in train_raw.columns if c != GROUP_COLUMN]

    # Subset of features that actually need scaling
    cols_to_scale = [c for c in FEATURES_TO_SCALE if c in train_raw.columns]

    # ----------------------------------------------------------
    # 7b. Fit scalers ONCE per model — always on training data only
    # ----------------------------------------------------------

    x_scaler = MinMaxScaler()
    x_scaler.fit(train_raw[cols_to_scale])

    # Dedicated scaler for target — avoids leakage
    y_scaler = MinMaxScaler()
    y_scaler.fit(train_raw[[TARGET_COLUMN]])

    # Save scalers for future inference without retraining
    with open(os.path.join(OUTPUT_FOLDER, f"x_scaler_model_{m}.pkl"), 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(OUTPUT_FOLDER, f"y_scaler_model_{m}.pkl"), 'wb') as f:
        pickle.dump(y_scaler, f)

    # ----------------------------------------------------------
    # 7c. Apply scaling — transform only, never re-fit on val/test
    # ----------------------------------------------------------
    def scale_split(df_raw):
        df = df_raw.copy()
        df[cols_to_scale] = x_scaler.transform(df[cols_to_scale])
        return df

    train_scaled = scale_split(train_raw)
    val_scaled   = scale_split(val_raw)
    test_scaled  = scale_split(test_raw)

    # Scaled target vectors (1-D)
    y_train_sc = y_scaler.transform(train_raw[[TARGET_COLUMN]]).flatten()
    y_val_sc   = y_scaler.transform(val_raw[[TARGET_COLUMN]]).flatten()
    y_test_sc  = y_scaler.transform(test_raw[[TARGET_COLUMN]]).flatten()

    # Category arrays aligned with each split
    train_cats = train_raw[GROUP_COLUMN].values
    val_cats   = val_raw[GROUP_COLUMN].values
    test_cats  = test_raw[GROUP_COLUMN].values

    # Reconstruct proper dates for the test set
    test_dates = pd.to_datetime(
        test_raw[['year', 'month', 'day']].rename(
            columns={'year': 'year', 'month': 'month', 'day': 'day'}
        )
    ).reset_index(drop=True)

    n_features = len(all_feature_cols)

    # ----------------------------------------------------------
    # 7d. Loop over forecasting horizons
    # ----------------------------------------------------------
    for h in HORIZONS:

        exp_id = f"Model_{m.upper()}_H{h}"
        print(f"\n  >>> {exp_id} <<<")

        # ---- Build sequences per category ----
        X_tr, y_tr = create_sequences_by_category(
            train_scaled, y_train_sc, TIME_STEPS, h, train_cats, all_feature_cols)

        X_va, y_va = create_sequences_by_category(
            val_scaled, y_val_sc, TIME_STEPS, h, val_cats, all_feature_cols)

        X_te, y_te, te_dates, te_cats = create_sequences_by_category_with_meta(
            test_scaled, y_test_sc, TIME_STEPS, h,
            test_cats, all_feature_cols, test_dates)

        if len(X_tr) == 0 or len(X_te) == 0:
            print(f"    Skipping {exp_id}: not enough data.")
            pbar.update(1)
            continue

        has_val = len(y_va) > 0

        print(f"    Shapes -> X_tr:{X_tr.shape}  "
              f"X_va:{X_va.shape if has_val else 'empty'}  "
              f"X_te:{X_te.shape}")

        # Dummy attention targets (required by Keras for the second output head,
        # but never used in gradient computation since loss=None for that head)
        att_shape_tr = (len(X_tr), TIME_STEPS, TIME_STEPS)
        att_shape_va = (len(X_va), TIME_STEPS, TIME_STEPS) if has_val else None

        y_tr_targets = [y_tr, np.zeros(att_shape_tr)]
        y_va_targets = [y_va, np.zeros(att_shape_va)] if has_val else None

        # ---- Train BiLSTM ----
        nn_model   = build_att_bilstm_model(TIME_STEPS, n_features)
        monitor    = 'val_loss' if has_val else 'loss'
        early_stop = EarlyStopping(
            monitor=monitor, patience=20, restore_best_weights=True)

        history = nn_model.fit(
            X_tr,
            y_tr_targets,
            validation_data=(X_va, y_va_targets) if has_val else None,
            epochs=50,
            batch_size=128,
            callbacks=[early_stop],
            verbose=1
        )

        # ---- Save training history as CSV ----
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(
            os.path.join(OUTPUT_FOLDER, f"history_{exp_id}.csv"), index=False)

        # ---- Save training history plot ----
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history_df['loss'], label='Train loss', linewidth=1.5)
        if 'val_loss' in history_df.columns:
            ax.plot(history_df['val_loss'], label='Val loss',
                    linewidth=1.5, linestyle='--')
        ax.set_title(f'Training history — {exp_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Mark the epoch where early stopping restored weights
        best_epoch = np.argmin(history_df.get('val_loss',
                                               history_df['loss']))
        ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7,
                   label=f'Best epoch ({best_epoch+1})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"history_{exp_id}.png"),
                    dpi=100, bbox_inches='tight')
        plt.close()

        # ---- Train XGBoost to correct residual errors ----
        # XGBoost input: only the BiLSTM scalar prediction per sample.
        nn_pred_tr, _ = nn_model.predict(X_tr, verbose=0)
        nn_pred_tr    = nn_pred_tr.flatten()
        residuals_tr  = y_tr - nn_pred_tr

        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(nn_pred_tr.reshape(-1, 1), residuals_tr)

        # ---- Generate test set predictions ----
        nn_pred_te, att_matrix = nn_model.predict(X_te, verbose=0)
        nn_pred_te   = nn_pred_te.flatten()
        xgb_residual = xgb_model.predict(nn_pred_te.reshape(-1, 1))

        final_scaled = nn_pred_te + xgb_residual

        # Inverse transform to original sales_volume scale
        actual = y_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        pred   = y_scaler.inverse_transform(final_scaled.reshape(-1, 1)).flatten()

        # ---- Extract attention weights ----
        if att_matrix.ndim == 3:
            mean_att = att_matrix.mean(axis=(0, 1))
        elif att_matrix.ndim == 4:
            mean_att = att_matrix.mean(axis=(0, 1, 2))
        else:
            mean_att = att_matrix.mean(axis=0).flatten()

        attention_global[exp_id] = mean_att

        pd.DataFrame({
            'lag':       np.arange(len(mean_att)),
            'attention': mean_att
        }).to_csv(
            os.path.join(OUTPUT_FOLDER, f"attention_{exp_id}.csv"), index=False)

        # ---- Compute error metrics ----
        mae   = mean_absolute_error(actual, pred)
        rmse  = np.sqrt(mean_squared_error(actual, pred))
        wmape = calculate_wmape(actual, pred)
        nse   = calculate_nse(actual, pred)
        me    = calculate_me(actual, pred)

        results_log.append({
            'Experiment_ID': exp_id,
            'Model':   m.upper(),
            'Horizon': f'H{h}',
            'MAE':     round(mae,   4),
            'RMSE':    round(rmse,  4),
            'WMAPE':   round(wmape, 4),
            'NSE':     round(nse,   4),
            'ME':      round(me,    6)
        })

        print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"WMAPE={wmape:.2f}%  NSE={nse:.4f}  ME={me:.6f}")

        # ---- Store prediction rows for the output CSV ----
        for i, (date_val, cat_val) in enumerate(zip(te_dates, te_cats)):
            dt = pd.Timestamp(date_val)
            all_predictions.append({
                'model':          m.upper(),
                'horizon':        h,
                'category_num':   int(cat_val),
                'day':            dt.day,
                'month':          dt.month,
                'year':           dt.year,
                'actual':         round(float(actual[i]), 4),
                'predicted':      round(float(pred[i]),   4),
                'absolute_error': round(abs(float(actual[i]) - float(pred[i])), 4)
            })

        # ---- Persist model artefacts ----
        nn_model.save(os.path.join(OUTPUT_FOLDER, f"bilstm_{exp_id}.keras"))
        with open(os.path.join(OUTPUT_FOLDER, f"xgb_{exp_id}.pkl"), 'wb') as f:
            pickle.dump(xgb_model, f)

        pbar.update(1)

pbar.close()


# ============================================================
# 8. SAVE PREDICTIONS CSV
# ============================================================

print("\nSaving predictions_all.csv ...")
pred_df = pd.DataFrame(all_predictions)
pred_df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions_all.csv"), index=False)
print(f"  -> {len(pred_df):,} rows written.")


# ============================================================
# 9. SAVE METRICS TABLE (CSV + styled PNG)
# ============================================================

print("Saving metrics report ...")
metrics_df = pd.DataFrame(results_log)
metrics_df.to_csv(os.path.join(OUTPUT_FOLDER, "metrics_report.csv"), index=False)

# ---- Styled table image ----
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

col_labels = ['Experiment', 'Model', 'Horizon', 'MAE', 'RMSE', 'WMAPE', 'NSE', 'ME']
cell_data  = metrics_df[
    ['Experiment_ID', 'Model', 'Horizon', 'MAE', 'RMSE', 'WMAPE', 'NSE', 'ME']
].values.tolist()

tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.7)

# Style header row
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#4472C4')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Alternating row shading
for i in range(1, len(cell_data) + 1):
    bg = '#FFFFFF' if i % 2 == 1 else '#DCE6F1'
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(bg)

# Highlight the best MAE row per horizon in green
for horizon_label in metrics_df['Horizon'].unique():
    sub      = metrics_df[metrics_df['Horizon'] == horizon_label]
    best_pos = sub['MAE'].idxmin()
    row_pos  = metrics_df.index.get_loc(best_pos) + 1
    for j in range(len(col_labels)):
        tbl[row_pos, j].set_facecolor('#92D050')

plt.title('Error Metrics — All Models and Horizons',
          fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "metrics_report.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("  -> metrics_report.png saved.")


# ============================================================
# 10. ABSOLUTE ERROR TREND PLOTS — one per horizon, 3 models
#
# Each plot shows the daily absolute error (|actual - predicted|)
# aggregated across all categories, for each of the 3 models.
# This makes it easy to compare which model is closer to reality
# on each date, without cluttering the chart with the actual line.
# ============================================================

print("Generating absolute error trend plots ...")

MODEL_COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
MODEL_LABELS = {
    'A': 'Model A (full sentiment)',
    'B': 'Model B (review metrics)',
    'C': 'Model C (baseline)'
}

# Reconstruct date column once for the full predictions DataFrame
pred_df['date'] = pd.to_datetime(pred_df[['year', 'month', 'day']])

for h in HORIZONS:
    h_data = pred_df[pred_df['horizon'] == h].copy()

    if h_data.empty:
        print(f"  -> No data for H{h}, skipping plot.")
        continue

    fig, ax = plt.subplots(figsize=(14, 5))

    for model_label in ['A', 'B', 'C']:
        sub = h_data[h_data['model'] == model_label]
        if sub.empty:
            continue

        # Aggregate absolute error by date: sum across all categories
        daily = (sub.groupby('date')['absolute_error']
                    .sum()
                    .reset_index()
                    .sort_values('date'))

        ax.plot(daily['date'], daily['absolute_error'],
                color=MODEL_COLORS[model_label], linewidth=1.3,
                label=MODEL_LABELS[model_label])

    # Format x-axis as calendar dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')

    ax.set_title(f'Absolute Error by Model — Horizon H{h} days',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Absolute Error (all categories)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, f"plot_H{h}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> plot_H{h}.png saved.")


# ============================================================
# 11. ATTENTION HEATMAP (12 experiments x 60 lags)
# ============================================================

print("Generating attention heatmap ...")

att_df = pd.DataFrame(attention_global).T   # shape (12, 60)
att_df.columns = np.arange(TIME_STEPS)
att_df = att_df.sort_index()

fig, ax = plt.subplots(figsize=(16, 6))

sns.heatmap(
    att_df,
    cmap='viridis',
    ax=ax,
    cbar_kws={'label': 'Importance Score'}
)

tick_pos = np.arange(0, TIME_STEPS, 2)
ax.set_xticks(tick_pos + 0.5)
ax.set_xticklabels(tick_pos, fontsize=7, rotation=0)

ax.set_title('Temporal Attention Importance (Lags Analysis)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Days into the past (Lags)')
ax.set_ylabel('Experiment ID')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "attention_heatmap.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("  -> attention_heatmap.png saved.")


# ============================================================
# 12.  print summary of all generated files
# ============================================================

print("\n" + "=" * 60)
print(f"All outputs saved to:\n  {OUTPUT_FOLDER}")
print("=" * 60)
print("\nFiles generated:")
for fn in sorted(os.listdir(OUTPUT_FOLDER)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_FOLDER, fn)) / 1024
    print(f"  {fn:50s} {size_kb:8.1f} KB")

