# NOTEBOOK 06 — SHAP EXPLAINABILITY
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# ======================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('resultsfigs', exist_ok=True)

# Load model and data 
df       = pd.read_csv('dataclean/model_dataset.csv')
xgb      = joblib.load('results/best_model_xgb.pkl')
FEATURES = joblib.load('results/feature_names.pkl')

df['fund_stress_flag'] = (
    (df['claims_intensity'] > 0.01) | (df['gdp_growth'] < 0)
).astype(int)

df_model = df.dropna(subset=FEATURES + ['fund_stress_flag']).copy()
X        = df_model[FEATURES].values
feature_labels = [
    'Fund Adequacy Ratio',
    'Claims Intensity',
    'Coverage Ratio',
    'NPL Ratio (%)',
    'CAR (%)',
    'ROA (%)',
    'Liquidity Ratio (%)',
    'GDP Growth (lag 1yr)',
    'Inflation (lag 1yr)',
    'Exchange Rate (lag 1yr)',
    'Fund Growth Rate'
]

# SECTION 1 — Compute SHAP values

explainer  = shap.TreeExplainer(xgb)
shap_vals  = explainer.shap_values(X)

print("SHAP values computed.")
print(f"Shape: {shap_vals.shape}  (rows × features)")

# SECTION 2 — Summary plot (beeswarm)

shap.initjs()
fig_s = plt.figure(figsize=(11, 7))
shap.summary_plot(
    shap_vals, X,
    feature_names=feature_labels,
    show=False,
    plot_size=(11, 7)
)
plt.title('SHAP Summary Plot — XGBoost Fund Stress Predictor\n(Nigeria NDIC 2010–2024)',
          fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('resultsfigs/10_shap_summary.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 10 saved: SHAP summary plot")

# SECTION 3 — Feature importance bar chart

mean_shap = np.abs(shap_vals).mean(axis=0)
importance_df = pd.DataFrame({
    'feature'   : feature_labels,
    'mean_shap' : mean_shap
}).sort_values('mean_shap', ascending=True)

colors = ['#C0392B' if i >= len(importance_df) - 5 else '#7F8C8D'
          for i in range(len(importance_df))]

fig_imp, ax_imp = plt.subplots(figsize=(10, 7))
bars = ax_imp.barh(importance_df['feature'], importance_df['mean_shap'],
                   color=colors, edgecolor='white', height=0.6)
for bar in bars:
    w = bar.get_width()
    ax_imp.text(w + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', va='center', ha='left', fontsize=10)

ax_imp.set_xlabel('Mean |SHAP Value|  (average impact on model output)', fontsize=11)
ax_imp.set_title('Top Predictors of Fund Stress — XGBoost SHAP Importance\n(Nigeria NDIC 2011–2024)',
                 fontsize=13, fontweight='bold', pad=12)
ax_imp.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('resultsfigs/11_shap_feature_importance.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 11 saved: SHAP feature importance bar chart")

# SECTION 4 — Plain-language top 5 feature explanations

top5 = importance_df.nlargest(5, 'mean_shap')
print("\n── Top 5 Predictors (SHAP) — Plain-Language Explanations ──")
explanations = {
    'Exchange Rate (lag 1yr)' :
        "When the Naira depreciated sharply in the prior year, fund stress became more likely "
        "in the following year. Exchange rate shocks erode the real value of the insurance fund "
        "and raise the cost of insured deposits denominated in foreign currency.",
    'GDP Growth (lag 1yr)' :
        "Recessions in the prior year are a strong predictor of fund stress. "
        "When the Nigerian economy contracted (2016, 2020), bank NPLs rose and "
        "NDIC faced increased claims pressure.",
    'Inflation (lag 1yr)' :
        "High inflation in the prior year signals macroeconomic instability. "
        "It erodes depositor real returns, reduces bank profitability, and "
        "increases systemic risk — all of which elevate fund stress probability.",
    'NPL Ratio (%)' :
        "A rising share of non-performing loans signals deteriorating bank asset quality. "
        "High NPLs precede bank failures and claims on the deposit insurance fund.",
    'Claims Intensity' :
        "This is the ratio of claims already paid to the total fund balance. "
        "When this ratio rises, the fund is being drawn down faster, directly "
        "signalling stress on the insurance system.",
    'Fund Adequacy Ratio' :
        "A lower fund adequacy ratio (fund balance / insured deposits) means "
        "the fund has less buffer against potential losses. "
        "When this falls, the system is closer to a stress event.",
    'CAR (%)' :
        "Capital Adequacy Ratio measures bank solvency. When CAR falls "
        "across the banking sector, systemic fragility increases and "
        "the likelihood of deposit insurance claims rises.",
    'ROA (%)' :
        "Return on Assets measures banking sector profitability. "
        "When ROA falls, banks are less able to absorb losses internally, "
        "increasing the probability they will fail and trigger claims.",
    'Coverage Ratio' :
        "The coverage limit relative to GDP per capita affects the scope of "
        "potential claims. When coverage is very high or very low relative "
        "to income, it distorts depositor behaviour and fund exposure.",
    'Liquidity Ratio (%)' :
        "Liquidity ratio captures how easily banks can meet short-term obligations. "
        "When it drops, banks are more vulnerable to sudden deposit withdrawals.",
    'Fund Growth Rate' :
        "A slowing or negative fund growth rate signals that premium income "
        "is insufficient to offset claims and administrative costs, "
        "pointing toward medium-term fund adequacy deterioration."
}

for _, row in top5.iterrows():
    feat = row['feature']
    print(f"\n  {feat}  (mean |SHAP| = {row['mean_shap']:.4f})")
    print(f"  → {explanations.get(feat, 'See model output.')}")

# SECTION 5 — Waterfall chart: highest-stress year

# Identify the year with highest predicted stress probability
stress_probs = xgb.predict_proba(X)[:, 1]
df_model = df_model.copy()
df_model['stress_prob'] = stress_probs
highest_stress_idx = stress_probs.argmax()
highest_stress_year = df_model.iloc[highest_stress_idx]['year']

print(f"\n── Highest-stress year for waterfall: {int(highest_stress_year)} ──")
print(f"   Predicted stress probability: {stress_probs[highest_stress_idx]:.4f}")
print(f"   Actual flag: {df_model.iloc[highest_stress_idx]['fund_stress_flag']}")

# SHAP waterfall plot
shap_exp = shap.Explanation(
    values         = shap_vals[highest_stress_idx],
    base_values    = explainer.expected_value,
    data           = X[highest_stress_idx],
    feature_names  = feature_labels
)

fig_wf = plt.figure(figsize=(12, 7))
shap.waterfall_plot(shap_exp, show=False, max_display=11)
plt.title(f'SHAP Waterfall — {int(highest_stress_year)} (Highest Predicted Stress Year)\n'
          f'Predicted probability = {stress_probs[highest_stress_idx]:.4f}',
          fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('resultsfigs/12_shap_waterfall.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 12 saved: SHAP waterfall chart")

# SECTION 6 — SHAP dependence plot: top feature vs stress prob

top_feat_idx = np.argmax(mean_shap)
fig_dep, ax_dep = plt.subplots(figsize=(9, 6))
ax_dep.scatter(
    X[:, top_feat_idx],
    shap_vals[:, top_feat_idx],
    c=stress_probs, cmap='RdYlGn_r', s=100,
    edgecolors='grey', linewidths=0.5, alpha=0.85
)
for i, row in df_model.reset_index(drop=True).iterrows():
    ax_dep.annotate(
        str(int(row['year'])),
        (X[i, top_feat_idx], shap_vals[i, top_feat_idx]),
        textcoords='offset points', xytext=(5, 3),
        fontsize=8.5, color='#444444'
    )
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                             norm=plt.Normalize(stress_probs.min(), stress_probs.max()))
plt.colorbar(sm, ax=ax_dep, label='Predicted Stress Probability')
ax_dep.axhline(0, color='grey', linestyle='--', linewidth=1)
ax_dep.set_xlabel(feature_labels[top_feat_idx], fontsize=12)
ax_dep.set_ylabel(f'SHAP value for {feature_labels[top_feat_idx]}', fontsize=12)
ax_dep.set_title(f'SHAP Dependence Plot — {feature_labels[top_feat_idx]}\nColoured by predicted stress probability',
                 fontsize=13, fontweight='bold')
ax_dep.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('resultsfigs/13_shap_dependence.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 13 saved: SHAP dependence plot")

print("\n✅  Notebook 06 complete. All SHAP plots saved to resultsfigs/")
