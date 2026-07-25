import pandas as pd

print("=" * 55)
print("SANITY CHECK — all three clean files")
print("=" * 55)

# ── NDIC ──────────────────────────────────────────────────
ndic = pd.read_csv('dataclean/ndic_annual.csv')
print("\n── NDIC ──────────────────────────────────────────")
print(f"  Rows: {len(ndic)}  |  Columns: {len(ndic.columns)}")
print(f"  Years: {ndic['year'].min()} – {ndic['year'].max()}")
print(f"  Nulls:\n{ndic.isnull().sum()[ndic.isnull().sum() > 0]}")

# NPL ratio cross-check
ndic['npl_check'] = (ndic['npl_tn'] / ndic['total_loans_tn'] * 100).round(2)
mismatch = ndic[abs(ndic['npl_check'] - ndic['npl_ratio_pct']) > 0.5]
print(f"  NPL ratio mismatches (>0.5pp): {len(mismatch)}")

# fund_balance_bn must grow every year
drops = ndic[ndic['fund_balance_bn'].diff() < 0]
print(f"  Fund balance year-on-year drops: {len(drops)}")

# ── IADI ──────────────────────────────────────────────────
iadi = pd.read_csv('dataclean/iadi_survey.csv')
print("\n── IADI ──────────────────────────────────────────")
print(f"  Rows: {len(iadi)}  |  Countries: {iadi['country'].nunique()}")
print(f"  Years: {iadi['year'].min()} – {iadi['year'].max()}")
print(f"  Nulls:\n{iadi.isnull().sum()[iadi.isnull().sum() > 0]}")

iadi['far_check'] = (iadi['fund_balance_usd'] / iadi['insured_deposits_usd']).round(4)
far_fail = iadi[abs(iadi['far_check'] - iadi['fund_adequacy_ratio']) > 0.001]
print(f"  FAR cross-check failures: {len(far_fail)}")

# ── World Bank ─────────────────────────────────────────────
macro = pd.read_csv('dataclean/world_bank_macro.csv')
print("\n── World Bank Macro ──────────────────────────────")
print(f"  Rows: {len(macro)}  |  Countries: {macro['country'].nunique()}")
print(f"  Nulls:\n{macro.isnull().sum()[macro.isnull().sum() > 0]}")

# ── Three-way Nigeria join ─────────────────────────────────
master = pd.read_csv('dataclean/ndic_iadi_nigeria_master.csv')
print("\n── Nigeria master join ───────────────────────────")
print(f"  Rows: {len(master)}  |  Null cells: {master.isnull().sum().sum()}")
print(f"  Columns: {list(master.columns)}")
print("\n✅  Sanity check complete.")
