# NOTEBOOK 02 — FEATURE ENGINEERING
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# ======================================

import pandas as pd
import numpy as np

df = pd.read_csv('dataclean/nigeria_analytical_frame.csv')
print(f"Frame loaded: {df.shape}")

# 1. Fund Adequacy Ratio (NDIC local calculation) 
# fund_balance_bn and insured_deposits_bn are both in billions NGN
df['local_fund_adequacy_ratio'] = (
    df['fund_balance_bn'] / df['insured_deposits_bn']
).round(4)

# 2. Coverage Ratio: coverage_limit_usd / gdp_per_capita (USD) 
df['coverage_ratio'] = (
    df['coverage_limit_usd'] / df['gdp_per_capita']
).round(4)

# 3. Claims Intensity: claims_paid / fund_balance 
df['claims_intensity'] = (
    df['claims_paid_bn'] / df['fund_balance_bn']
).round(4)

# 4. Premium Efficiency: premium revenue proxy 
# mean_premium_rate_pct × total_deposits_tn as proxy for premiums collected
# (expressed as % of insured deposits)
df['premium_efficiency'] = (
    df['mean_premium_rate_pct'] / 100
).round(6)

# 5. Fund Growth Rate: year-on-year % change 
df['fund_growth_rate'] = (
    df['fund_balance_bn'].pct_change() * 100
).round(4)

# 6. IADI Benchmark Gap: NDIC local FAR minus IADI peer average 
df['fund_adequacy_gap'] = (
    df['local_fund_adequacy_ratio'] - df['iadi_avg_fund_adequacy']
).round(4)

# 7. NDIC vs IADI-reported FAR gap
# fund_adequacy_ratio = IADI-reported figure for Nigeria
df['iadi_reported_gap'] = (
    df['local_fund_adequacy_ratio'] - df['fund_adequacy_ratio']
).round(4)

# 8. Lagged macro variables (prevent data leakage)
df['gdp_growth_lag1']    = df['gdp_growth'].shift(1).round(4)
df['inflation_lag1']     = df['inflation'].shift(1).round(4)
df['exchange_rate_lag1'] = df['exchange_rate'].shift(1).round(4)

# 9. Stress flag: claims_intensity > 0.01 (1% of fund)
# (threshold set low because NDIC claims are consistently small vs fund)
df['fund_stress_flag'] = (df['claims_intensity'] > 0.01).astype(int)

# 10. Print full ratio table 
ratio_cols = [
    'year',
    'local_fund_adequacy_ratio',
    'fund_adequacy_ratio',        # IADI-reported
    'iadi_avg_fund_adequacy',     # peer average
    'fund_adequacy_gap',
    'iadi_reported_gap',
    'coverage_ratio',
    'claims_intensity',
    'premium_efficiency',
    'fund_growth_rate',
    'gdp_growth_lag1',
    'inflation_lag1',
    'fund_stress_flag'
]

print("\n── Computed ratios (all years) ──")
print(df[ratio_cols].to_string(index=False))

# 11. Save model dataset
df.to_csv('dataclean/model_dataset.csv', index=False)
print(f"\n✅  model_dataset.csv saved: {df.shape[0]} rows × {df.shape[1]} columns")

# 12. Print ratio summary for Excel validation
print("\n── 5-year sample for Excel manual verification ──")
sample_years = [2010, 2014, 2018, 2021, 2024]
print(df[df['year'].isin(sample_years)][
    ['year','fund_balance_bn','insured_deposits_bn',
     'local_fund_adequacy_ratio','coverage_limit_usd',
     'gdp_per_capita','coverage_ratio']
].to_string(index=False))
