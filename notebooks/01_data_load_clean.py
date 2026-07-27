# =============================================================
# NOTEBOOK 01 — DATA LOAD, CLEAN & MERGE
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# =============================================================

import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── SECTION 1: Load all clean files ───────────────────────────────────────────

ndic  = pd.read_csv('dataclean/ndic_annual.csv')
iadi  = pd.read_csv('dataclean/iadi_survey.csv')
macro = pd.read_csv('dataclean/world_bank_macro.csv')

print("Files loaded.")
print(f"  NDIC  : {ndic.shape}")
print(f"  IADI  : {iadi.shape}")
print(f"  Macro : {macro.shape}")

# ── SECTION 2: Inspect each file ──────────────────────────────────────────────

print("\n── NDIC dtypes and nulls ──")
print(ndic.dtypes)
print(ndic.isnull().sum())

print("\n── IADI dtypes and nulls ──")
print(iadi.dtypes)
print(iadi.isnull().sum())

print("\n── Macro dtypes and nulls ──")
print(macro.dtypes)
print(macro.isnull().sum())

# ── SECTION 3: Standardize country names with fuzzy matching ──────────────────

from thefuzz import process

# Build canonical country list from IADI
canonical = iadi['country'].dropna().unique().tolist()

def standardize(name, choices, threshold=85):
    match, score = process.extractOne(name, choices)
    return match if score >= threshold else name

# Apply to macro country column (catches 'nigeria' vs 'Nigeria' etc.)
macro['country'] = macro['country'].apply(lambda x: standardize(x, canonical))
iadi['country']  = iadi['country'].apply(lambda x: standardize(x, canonical))

print("\n── Countries after standardization ──")
print("IADI :", sorted(iadi['country'].unique()))
print("Macro:", sorted(macro['country'].unique()))

# ── SECTION 4: Missing value handling rules ───────────────────────────────────

# NDIC — one 'note' column has non-numeric text — keep as-is, exclude from modeling
# IADI — 'note' column has one value (2024 Nigeria flag) — keep as-is
# World Bank — zero nulls confirmed in sanity check
# Rule: drop 'note' columns from analytical frames; all numeric columns complete

ndic_clean  = ndic.drop(columns=['note'], errors='ignore')
iadi_clean  = iadi.drop(columns=['note'], errors='ignore')
macro_clean = macro.copy()

print("\n── Null counts after cleaning ──")
print("NDIC :", ndic_clean.isnull().sum().sum())
print("IADI :", iadi_clean.isnull().sum().sum())
print("Macro:", macro_clean.isnull().sum().sum())

# ── SECTION 5: Add region mapping to IADI (needed for pivot later) ────────────

region_map = {
    'Nigeria': 'Africa', 'Tanzania': 'Africa', 'Uganda': 'Africa',
    'USA': 'Americas', 'Canada': 'Americas', 'Brazil': 'Americas',
    'India': 'Asia', 'Indonesia': 'Asia', 'Malaysia': 'Asia',
    'Philippines': 'Asia', 'Vietnam': 'Asia',
    'Japan': 'Asia', 'South Korea': 'Asia', 'Taiwan': 'Asia'
}
iadi_clean['region'] = iadi_clean['country'].map(region_map)
print("\n── Region mapping check ──")
print(iadi_clean[['country','region']].drop_duplicates().to_string(index=False))

# ── SECTION 6: Compute IADI peer averages by year (all 14 countries) ──────────

iadi_avg = iadi_clean.groupby('year').agg(
    iadi_avg_fund_adequacy = ('fund_adequacy_ratio', 'mean'),
    iadi_med_fund_adequacy = ('fund_adequacy_ratio', 'median'),
    iadi_avg_coverage_usd  = ('coverage_limit_usd',  'mean'),
).reset_index()

print("\n── IADI peer averages by year ──")
print(iadi_avg.to_string(index=False))

# ── SECTION 7: Nigeria macro rows only ────────────────────────────────────────

macro_nga = macro_clean[macro_clean['country'].str.lower() == 'nigeria'].copy()
macro_nga = macro_nga.sort_values('year').reset_index(drop=True)
print(f"\n── Nigeria macro rows: {len(macro_nga)} ──")
print(macro_nga.to_string(index=False))

# ── SECTION 8: Nigeria IADI rows only ─────────────────────────────────────────

iadi_nga = iadi_clean[iadi_clean['country'].str.lower() == 'nigeria'].copy()
iadi_nga = iadi_nga.sort_values('year').reset_index(drop=True)
print(f"\n── Nigeria IADI rows: {len(iadi_nga)} ──")

# ── SECTION 9: Build the master Nigeria analytical frame ──────────────────────

df = ndic_clean.copy()
df = df.sort_values('year').reset_index(drop=True)

# Merge Nigeria macro
df = df.merge(macro_nga[['year','gdp_growth','inflation',
                           'exchange_rate','gdp_per_capita']],
              on='year', how='left')

# Merge Nigeria IADI metrics
df = df.merge(iadi_nga[['year','fund_adequacy_ratio',
                          'coverage_limit_usd',
                          'fund_balance_usd',
                          'insured_deposits_usd']],
              on='year', how='left',
              suffixes=('', '_iadi'))

# Merge IADI peer averages
df = df.merge(iadi_avg, on='year', how='left')

print(f"\n── Master frame shape: {df.shape} ──")
print(df.dtypes)
print(f"Nulls: {df.isnull().sum().sum()}")

# ── SECTION 10: Save intermediate clean frame ──────────────────────────────────

os.makedirs('dataclean', exist_ok=True)
df.to_csv('dataclean/nigeria_analytical_frame.csv', index=False)
print("\n✅  nigeria_analytical_frame.csv saved to dataclean/")
