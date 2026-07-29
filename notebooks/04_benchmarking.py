# =============================================================
# NOTEBOOK 04 — IADI BENCHMARKING ANALYSIS
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('resultsfigs', exist_ok=True)
os.makedirs('results', exist_ok=True)

df   = pd.read_csv('dataclean/model_dataset.csv')
iadi = pd.read_csv('dataclean/iadi_survey.csv')
macro = pd.read_csv('dataclean/world_bank_macro.csv')

# =============================================================
# SECTION 1 — Score NDIC against IADI Core Principles (1–5)
# Based on 2024 data and qualitative public evidence
# =============================================================

# IADI has 7 Core Principles for Effective Deposit Insurance
# We score NDIC 1 (poor) to 5 (best practice) on each

latest = df[df['year'] == df['year'].max()].iloc[0]

# Principle scoring rubric (evidence-based, from public NDIC/IADI data)
principles = {
    'CP1: Public Policy\nObjectives': {
        'score'    : 4,
        'evidence' : 'NDIC mandate clearly defined in NDIC Act 2006 (amended 2019). Covers depositor protection + bank resolution. Loses 1pt: no formal financial stability mandate.',
        'iadi_avg' : 3.5
    },
    'CP2: Mandate &\nPowers': {
        'score'    : 4,
        'evidence' : 'NDIC has payout, risk minimiser, and resolution powers. Formal MOU with CBN. Loses 1pt: limited early intervention powers vs IADI best practice.',
        'iadi_avg' : 3.6
    },
    'CP3: Governance': {
        'score'    : 3,
        'evidence' : 'Governing board includes CBN, Finance Ministry reps. Independence moderate — MD appointed by President. IADI best practice requires operationally independent board.',
        'iadi_avg' : 3.4
    },
    'CP4: Relationships\nwith Safety-Net': {
        'score'    : 4,
        'evidence' : 'Strong CBN-NDIC coordination. Joint examinations. Information sharing MOU in place. Loses 1pt: no formal crisis management committee enshrined in law.',
        'iadi_avg' : 3.5
    },
    'CP5: Membership\n& Coverage': {
        'score'    : 3,
        'evidence' : f"Coverage limit ₦5mn (~USD {latest['coverage_limit_usd']:,.0f}). Coverage ratio = {latest['coverage_ratio']:.2f}x GDP per capita. IADI recommends 1–5x. Pre-2024 limit was historically low. 2024 revision improves score.",
        'iadi_avg' : 3.7
    },
    'CP6: Fund\nManagement': {
        'score'    : 4,
        'evidence' : f"Local FAR = {latest['local_fund_adequacy_ratio']:.4f} vs IADI peer avg {latest['iadi_avg_fund_adequacy']:.4f}. Fund is ex-ante, risk-based premiums. Loses 1pt: no formal target fund ratio enshrined in statute.",
        'iadi_avg' : 3.3
    },
    'CP7: Public\nAwareness': {
        'score'    : 3,
        'evidence' : 'NDIC runs annual depositor awareness campaigns. Reports published. Loses 2pts: low financial literacy penetration; rural depositors largely unaware of coverage.',
        'iadi_avg' : 3.2
    }
}

scores_df = pd.DataFrame([
    {
        'principle' : k,
        'ndic_score': v['score'],
        'iadi_avg'  : v['iadi_avg'],
        'evidence'  : v['evidence']
    }
    for k, v in principles.items()
])

print("── IADI Core Principle Scores ──────────────────────────")
print(scores_df[['principle','ndic_score','iadi_avg']].to_string(index=False))
print(f"\nNDIC overall average : {scores_df['ndic_score'].mean():.2f} / 5")
print(f"IADI peer average    : {scores_df['iadi_avg'].mean():.2f} / 5")

# =============================================================
# SECTION 2 — Gap table: NDIC vs IADI average vs best practice
# =============================================================

iadi_nga = iadi[iadi['country'].str.lower() == 'nigeria'].copy()

# Compute IADI-wide averages and best-practice thresholds per year
iadi_stats = iadi.groupby('year').agg(
    iadi_avg_far = ('fund_adequacy_ratio', 'mean'),
    iadi_max_far = ('fund_adequacy_ratio', 'max'),
    iadi_avg_cov = ('coverage_limit_usd',  'mean'),
    iadi_max_cov = ('coverage_limit_usd',  'max')
).reset_index()

gap_base = df.merge(iadi_nga[['year','fund_adequacy_ratio','coverage_limit_usd']]
                    .rename(columns={
                        'fund_adequacy_ratio':'iadi_nga_far',
                        'coverage_limit_usd' :'iadi_nga_cov'
                    }), on='year', how='left')
gap_base = gap_base.merge(iadi_stats, on='year', how='left')

# Build long-format gap table
gap_records = []
for _, row in gap_base.iterrows():
    # Metric 1: Fund Adequacy Ratio
    gap_records.append({
        'year'       : int(row['year']),
        'metric'     : 'Fund Adequacy Ratio',
        'ndic_value' : round(row['local_fund_adequacy_ratio'], 4),
        'iadi_avg'   : round(row['iadi_avg_far'], 4),
        'iadi_best'  : round(row['iadi_max_far'], 4),
    })
    # Metric 2: Coverage Ratio
    gap_records.append({
        'year'       : int(row['year']),
        'metric'     : 'Coverage Ratio',
        'ndic_value' : round(row['coverage_ratio'], 4),
        'iadi_avg'   : round(row['iadi_avg_cov'] / max(row['gdp_per_capita'],1), 4),
        'iadi_best'  : 5.0,   # IADI recommends 1–5x GDP per capita as best practice
    })
    # Metric 3: Claims Intensity
    gap_records.append({
        'year'       : int(row['year']),
        'metric'     : 'Claims Intensity',
        'ndic_value' : round(row['claims_intensity'], 4),
        'iadi_avg'   : 0.05,   # IADI indicative threshold
        'iadi_best'  : 0.01,   # Lower is better; best practice = minimal claims
    })

gap_df = pd.DataFrame(gap_records)
gap_df['gap_to_iadi_avg']  = (gap_df['ndic_value'] - gap_df['iadi_avg']).round(4)
gap_df['gap_to_best']      = (gap_df['iadi_best']  - gap_df['ndic_value']).round(4)

# For claims intensity, gap_to_best should be positive if NDIC is ABOVE best (worse)
# Adjust sign for claims intensity (lower is better)
ci_mask = gap_df['metric'] == 'Claims Intensity'
gap_df.loc[ci_mask, 'gap_to_best'] = (
    gap_df.loc[ci_mask, 'ndic_value'] - gap_df.loc[ci_mask, 'iadi_best']
).round(4)

gap_df.to_csv('results/gap_table.csv', index=False)
print("\n── Gap table saved to results/gap_table.csv ──")
print(gap_df[gap_df['year'] == gap_df['year'].max()].to_string(index=False))

# =============================================================
# SECTION 3 — Radar chart: NDIC vs IADI average
#             across 6 Core Principle dimensions
# =============================================================

categories  = [p.replace('\n',' ') for p in list(principles.keys())[:6]]
ndic_scores = [principles[k]['score']   for k in list(principles.keys())[:6]]
iadi_scores = [principles[k]['iadi_avg'] for k in list(principles.keys())[:6]]

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ndic_vals = ndic_scores + ndic_scores[:1]
iadi_vals = iadi_scores + iadi_scores[:1]

fig3, ax3 = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

ax3.plot(angles, ndic_vals, 'o-', linewidth=2.5, color='#1D9E75', label='NDIC (Nigeria)')
ax3.fill(angles, ndic_vals, alpha=0.18, color='#1D9E75')

ax3.plot(angles, iadi_vals, 's--', linewidth=2, color='#2980B9', label='IADI Peer Average')
ax3.fill(angles, iadi_vals, alpha=0.10, color='#2980B9')

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax3.set_ylim(0, 5)
ax3.set_yticks([1, 2, 3, 4, 5])
ax3.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9, color='grey')
ax3.yaxis.set_tick_params(labelsize=8)
ax3.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

ax3.set_title('NDIC vs IADI Peer Average\nAcross 6 Core Principle Dimensions',
              fontsize=14, fontweight='bold', pad=25)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)

plt.tight_layout()
plt.savefig('resultsfigs/06_radar_iadi_benchmarking.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 6 saved: radar chart — IADI benchmarking")

# =============================================================
# SECTION 4 — Peer comparison table:
#             Nigeria vs African + emerging-market peers
# =============================================================

peers = ['Nigeria', 'Tanzania', 'Uganda', 'India', 'Indonesia', 'Philippines']

peer_iadi = iadi[iadi['country'].isin(peers)].copy()
peer_macro = macro[macro['country'].isin(peers)].copy()
peer_iadi = peer_iadi.merge(
    peer_macro[['country','year','gdp_per_capita']],
    on=['country','year'], how='left'
)
peer_iadi['coverage_ratio'] = (
    peer_iadi['coverage_limit_usd'] / peer_iadi['gdp_per_capita']
).round(4)

peer_avg = peer_iadi.groupby('country').agg(
    avg_fund_adequacy = ('fund_adequacy_ratio', 'mean'),
    avg_coverage_ratio= ('coverage_ratio',       'mean'),
    avg_coverage_usd  = ('coverage_limit_usd',   'mean'),
    latest_far        = ('fund_adequacy_ratio',   'last'),
    latest_cov        = ('coverage_limit_usd',    'last'),
).round(4).reset_index()

region_map = {
    'Nigeria':'Africa','Tanzania':'Africa','Uganda':'Africa',
    'India':'Asia','Indonesia':'Asia','Philippines':'Asia'
}
peer_avg['region'] = peer_avg['country'].map(region_map)
peer_avg['highlight'] = peer_avg['country'].apply(
    lambda x: '★ Nigeria (NDIC)' if x == 'Nigeria' else x
)

peer_avg = peer_avg.sort_values('avg_fund_adequacy', ascending=False)
peer_avg.to_csv('results/peer_comparison.csv', index=False)

print("\n── Peer Comparison Table ──────────────────────────────────────")
print(peer_avg[['highlight','region','avg_fund_adequacy',
                'avg_coverage_ratio','avg_coverage_usd',
                'latest_far','latest_cov']].to_string(index=False))

# Visual peer comparison table as PNG
fig4, ax4 = plt.subplots(figsize=(14, 5))
ax4.axis('off')

table_cols = ['Country','Region','Avg FAR\n2010–24','Avg Cov\nRatio','Avg Cov Limit\n(USD)','2024 FAR','2024 Cov\nLimit (USD)']
table_data = []
for _, row in peer_avg.iterrows():
    table_data.append([
        row['highlight'],
        row['region'],
        f"{row['avg_fund_adequacy']:.4f}",
        f"{row['avg_coverage_ratio']:.4f}",
        f"${row['avg_coverage_usd']:,.0f}",
        f"{row['latest_far']:.4f}",
        f"${row['latest_cov']:,.0f}"
    ])

tbl = ax4.table(
    cellText=table_data,
    colLabels=table_cols,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

for j in range(len(table_cols)):
    tbl[0, j].set_facecolor('#1B3A6B')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i, row in enumerate(peer_avg.itertuples(), start=1):
    color = '#FFF3CD' if row.country == 'Nigeria' else ('#F0F4F8' if i % 2 == 0 else 'white')
    for j in range(len(table_cols)):
        tbl[i, j].set_facecolor(color)

ax4.set_title('Peer Comparison: Nigeria (NDIC) vs African & Emerging-Market DICs',
              fontsize=13, fontweight='bold', pad=12, y=1.02)
plt.tight_layout()
plt.savefig('resultsfigs/07_peer_comparison_table.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 7 saved: peer comparison table")

print("\n✅  Notebook 04 complete. Files saved:")
print("    results/gap_table.csv")
print("    results/peer_comparison.csv")
print("    resultsfigs/06_radar_iadi_benchmarking.png")
print("    resultsfigs/07_peer_comparison_table.png")
