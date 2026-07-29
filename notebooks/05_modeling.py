# =============================================================
# NOTEBOOK 05 — PREDICTIVE MODELING
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# NOTE: 15 observations (2010–2024). Results are indicative.
#       Focus is on feature direction, not prediction accuracy.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve,
                              ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

os.makedirs('resultsfigs', exist_ok=True)

df = pd.read_csv('dataclean/model_dataset.csv')
print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Years  : {df['year'].min()} – {df['year'].max()}")

# =============================================================
# SECTION 1 — Define stress flag
# fund_stress_flag already in model_dataset from notebook 02
# (claims_intensity > 0.01).
# For modeling we also add: recession flag (gdp_growth < 0)
# =============================================================

df['fund_stress_flag'] = (
    (df['claims_intensity'] > 0.01) |
    (df['gdp_growth'] < 0)
).astype(int)

print(f"\nStress flag distribution:")
print(df[['year','claims_intensity','gdp_growth','fund_stress_flag']].to_string(index=False))
print(f"\nStress years (flag=1): {df[df['fund_stress_flag']==1]['year'].tolist()}")
print(f"Calm   years (flag=0): {df[df['fund_stress_flag']==0]['year'].tolist()}")

# =============================================================
# SECTION 2 — Feature selection
# =============================================================

FEATURES = [
    'local_fund_adequacy_ratio',
    'claims_intensity',
    'coverage_ratio',
    'npl_ratio_pct',
    'car_pct',
    'roa_pct',
    'liquidity_ratio_pct',
    'gdp_growth_lag1',
    'inflation_lag1',
    'exchange_rate_lag1',
    'fund_growth_rate'
]

# Drop first row (NaN from lag) and keep only complete rows
df_model = df.dropna(subset=FEATURES + ['fund_stress_flag']).copy()
print(f"\nModeling rows after dropping NaN: {len(df_model)}")

X = df_model[FEATURES].values
y = df_model['fund_stress_flag'].values

# =============================================================
# SECTION 3 — Train / test split (out-of-time)
# =============================================================

train_mask = df_model['year'] <= 2019
test_mask  = df_model['year'] >  2019

X_train = df_model.loc[train_mask, FEATURES].values
y_train = df_model.loc[train_mask, 'fund_stress_flag'].values
X_test  = df_model.loc[test_mask,  FEATURES].values
y_test  = df_model.loc[test_mask,  'fund_stress_flag'].values

print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
print(f"Train stress rate: {y_train.mean():.2f} | Test stress rate: {y_test.mean():.2f}")

# Scale features
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# =============================================================
# SECTION 4 — Baseline: Logistic Regression
# =============================================================

lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
lr.fit(X_train_s, y_train)

# Predict
if len(np.unique(y_test)) > 1:
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])
else:
    lr_auc = float('nan')
    print("⚠️  Only one class in test set — AUC not defined. Check stress flag distribution.")

lr_pred = lr.predict(X_test_s)
lr_prec = precision_score(y_test, lr_pred, zero_division=0)
lr_rec  = recall_score(y_test,  lr_pred, zero_division=0)
lr_f1   = f1_score(y_test,     lr_pred, zero_division=0)

print(f"\n── Logistic Regression ──────────────────────────────")
print(f"  AUC-ROC  : {lr_auc:.4f}")
print(f"  Precision: {lr_prec:.4f}")
print(f"  Recall   : {lr_rec:.4f}")
print(f"  F1 Score : {lr_f1:.4f}")

# =============================================================
# SECTION 5 — Advanced: XGBoost
# =============================================================

xgb = XGBClassifier(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)

if len(np.unique(y_test)) > 1:
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
else:
    xgb_auc = float('nan')

xgb_pred = xgb.predict(X_test)
xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
xgb_rec  = recall_score(y_test,  xgb_pred, zero_division=0)
xgb_f1   = f1_score(y_test,     xgb_pred, zero_division=0)

print(f"\n── XGBoost ──────────────────────────────────────────")
print(f"  AUC-ROC  : {xgb_auc:.4f}")
print(f"  Precision: {xgb_prec:.4f}")
print(f"  Recall   : {xgb_rec:.4f}")
print(f"  F1 Score : {xgb_f1:.4f}")

# =============================================================
# SECTION 6 — Cross-validation on full dataset (leave-one-out)
# With 14 rows, use LOO CV to maximise each fold
# =============================================================

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

lr_cv_scores  = cross_val_score(
    LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    scaler.fit_transform(X), y, cv=loo, scoring='f1_macro'
)
xgb_cv_scores = cross_val_score(
    XGBClassifier(n_estimators=50, max_depth=2, verbosity=0, random_state=42),
    X, y, cv=loo, scoring='f1_macro'
)

print(f"\n── Leave-One-Out CV (F1 macro) ──────────────────────")
print(f"  Logistic Regression : mean={lr_cv_scores.mean():.4f}  std={lr_cv_scores.std():.4f}")
print(f"  XGBoost             : mean={xgb_cv_scores.mean():.4f}  std={xgb_cv_scores.std():.4f}")

# =============================================================
# SECTION 7 — Save model metrics CSV
# =============================================================

metrics_df = pd.DataFrame([
    {
        'model'         : 'Logistic Regression (baseline)',
        'auc_roc'       : round(lr_auc, 4)  if not np.isnan(lr_auc)  else 'N/A',
        'precision'     : round(lr_prec, 4),
        'recall'        : round(lr_rec, 4),
        'f1_score'      : round(lr_f1, 4),
        'loo_cv_f1_mean': round(lr_cv_scores.mean(), 4),
        'loo_cv_f1_std' : round(lr_cv_scores.std(), 4),
        'train_years'   : '2011–2019',
        'test_years'    : '2020–2024',
        'n_train'       : len(X_train),
        'n_test'        : len(X_test),
        'note'          : 'Small sample (n=14). Indicative only.'
    },
    {
        'model'         : 'XGBoost (advanced)',
        'auc_roc'       : round(xgb_auc, 4) if not np.isnan(xgb_auc) else 'N/A',
        'precision'     : round(xgb_prec, 4),
        'recall'        : round(xgb_rec, 4),
        'f1_score'      : round(xgb_f1, 4),
        'loo_cv_f1_mean': round(xgb_cv_scores.mean(), 4),
        'loo_cv_f1_std' : round(xgb_cv_scores.std(), 4),
        'train_years'   : '2010–2019',
        'test_years'    : '2020–2024',
        'n_train'       : len(X_train),
        'n_test'        : len(X_test),
        'note'          : 'Small sample (n=14). Indicative only.'
    }
])

metrics_df.to_csv('results/model_metrics.csv', index=False)
print("\n✅  model_metrics.csv saved to results/")
print(metrics_df.to_string(index=False))

# =============================================================
# SECTION 8 — ROC curve (best model = XGBoost)
# =============================================================

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

if len(np.unique(y_test)) > 1:
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr.predict_proba(X_test_s)[:, 1])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb.predict_proba(X_test)[:, 1])
    ax_roc.plot(fpr_lr,  tpr_lr,  'b--', lw=2,
                label=f'Logistic Regression (AUC={lr_auc:.3f})')
    ax_roc.plot(fpr_xgb, tpr_xgb, 'g-',  lw=2.5,
                label=f'XGBoost (AUC={xgb_auc:.3f})')
else:
    ax_roc.text(0.5, 0.5, 'AUC not computable\n(only one class in test set)',
                ha='center', va='center', fontsize=13, color='grey')

ax_roc.plot([0,1],[0,1],'k:', lw=1.5, label='Random baseline')
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curve — Fund Stress Prediction\n(NDIC Nigeria 2010–2024)',
                 fontsize=13, fontweight='bold')
ax_roc.legend(fontsize=11)
ax_roc.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('resultsfigs/08_roc_curve.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 8 saved: ROC curve")

# =============================================================
# SECTION 9 — Confusion matrix (XGBoost)
# =============================================================

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Calm (0)', 'Stress (1)'])
disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
ax_cm.set_title('Confusion Matrix — XGBoost\n(Test: 2020–2024)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('resultsfigs/09_confusion_matrix.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 9 saved: confusion matrix")

# Save best model
joblib.dump(xgb,    'results/best_model_xgb.pkl')
joblib.dump(scaler, 'results/scaler.pkl')
joblib.dump(FEATURES,'results/feature_names.pkl')
print("✅  Model artifacts saved to results/")
