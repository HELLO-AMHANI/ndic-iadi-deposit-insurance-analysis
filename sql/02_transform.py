import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine('sqlite:///ndic_iadi.db')
os.makedirs('dataclean', exist_ok=True)

# ── 1. NDIC clean export ──────────────────────────────────────────────────────
ndic = pd.read_sql("SELECT * FROM ndic_raw ORDER BY year", engine)
ndic.to_csv('dataclean/ndic_annual.csv', index=False)
print(f"✅  ndic_annual.csv exported:       {len(ndic)} rows")

# ── 2. Nigeria-only IADI rows ─────────────────────────────────────────────────
iadi_nga = pd.read_sql("""
    SELECT * FROM iadi_survey
    WHERE LOWER(country) = 'nigeria'
    ORDER BY year
""", engine)
iadi_nga.to_csv('dataclean/iadi_nigeria.csv', index=False)
print(f"✅  iadi_nigeria.csv exported:      {len(iadi_nga)} rows")

# ── 3. Full IADI survey clean ─────────────────────────────────────────────────
iadi_all = pd.read_sql("SELECT * FROM iadi_survey ORDER BY country, year", engine)
iadi_all.to_csv('dataclean/iadi_survey.csv', index=False)
print(f"✅  iadi_survey.csv exported:       {len(iadi_all)} rows")

# ── 4. Nigeria macro rows ─────────────────────────────────────────────────────
macro_nga = pd.read_sql("""
    SELECT * FROM world_bank_macro
    WHERE LOWER(country) = 'nigeria'
    ORDER BY year
""", engine)
macro_nga.to_csv('dataclean/macro_nigeria.csv', index=False)
print(f"✅  macro_nigeria.csv exported:     {len(macro_nga)} rows")

# ── 5. Full macro clean ───────────────────────────────────────────────────────
macro_all = pd.read_sql("SELECT * FROM world_bank_macro ORDER BY country, year", engine)
macro_all.to_csv('dataclean/world_bank_macro.csv', index=False)
print(f"✅  world_bank_macro.csv exported:  {len(macro_all)} rows")

# ── 6. NDIC joined with Nigeria macro (the master Nigeria table) ──────────────
ndic_macro = pd.read_sql("""
    SELECT
        n.*,
        m.gdp_growth,
        m.inflation,
        m.exchange_rate,
        m.gdp_per_capita
    FROM ndic_raw n
    LEFT JOIN world_bank_macro m
        ON n.year = m.year
        AND LOWER(m.country) = 'nigeria'
    ORDER BY n.year
""", engine)
ndic_macro.to_csv('dataclean/ndic_with_macro.csv', index=False)
print(f"✅  ndic_with_macro.csv exported:   {len(ndic_macro)} rows")

# ── 7. NDIC + IADI Nigeria side-by-side (benchmarking table) ─────────────────
ndic_iadi_nga = pd.read_sql("""
    SELECT
        n.year,
        n.fund_balance_bn,
        n.insured_deposits_bn,
        n.claims_paid_bn,
        n.mean_premium_rate_pct,
        n.num_banks,
        n.npl_ratio_pct,
        n.car_pct,
        n.liquidity_ratio_pct,
        n.roa_pct,
        i.fund_adequacy_ratio   AS iadi_fund_adequacy_ratio,
        i.coverage_limit_usd    AS iadi_coverage_limit_usd,
        m.gdp_growth,
        m.inflation,
        m.exchange_rate,
        m.gdp_per_capita
    FROM ndic_raw n
    LEFT JOIN iadi_survey i
        ON n.year = i.year AND LOWER(i.country) = 'nigeria'
    LEFT JOIN world_bank_macro m
        ON n.year = m.year AND LOWER(m.country) = 'nigeria'
    ORDER BY n.year
""", engine)
ndic_iadi_nga.to_csv('dataclean/ndic_iadi_nigeria_master.csv', index=False)
print(f"✅  ndic_iadi_nigeria_master.csv:   {len(ndic_iadi_nga)} rows")

print("\n🗂️  All clean CSVs saved to data/clean/")
