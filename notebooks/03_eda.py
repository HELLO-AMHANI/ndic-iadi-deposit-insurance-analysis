# NOTEBOOK 03 — EXPLORATORY DATA ANALYSIS
# NDIC + IADI Deposit Insurance Analysis
# Author: Promise O. Amhanesi
# =======================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

os.makedirs('resultsfigs', exist_ok=True)

# Load datasets 
df     = pd.read_csv('dataclean/model_dataset.csv')
iadi   = pd.read_csv('dataclean/iadi_survey.csv')
macro  = pd.read_csv('dataclean/world_bank_macro.csv')

print(f"model_dataset : {df.shape}")
print(f"iadi_survey   : {iadi.shape}")
print(f"macro         : {macro.shape}")
print(f"Years in df   : {df['year'].min()} – {df['year'].max()}")

# CHART 1 — Time-series: Fund Balance vs Claims Paid
#           with banking stress year annotations

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df['year'], y=df['fund_balance_bn'],
    mode='lines+markers',
    name='Fund Balance (₦ bn)',
    line=dict(color='#1D9E75', width=3),
    marker=dict(size=7)
))

fig1.add_trace(go.Scatter(
    x=df['year'], y=df['claims_paid_bn'],
    mode='lines+markers',
    name='Claims Paid (₦ bn)',
    line=dict(color='#C0392B', width=2, dash='dot'),
    marker=dict(size=7)
))

# Stress year annotations (within data range 2010–2024)
stress_years = {
    2016: 'Oil price crash<br>+ FX crisis',
    2020: 'COVID-19<br>shock'
}

for yr, label in stress_years.items():
    fig1.add_vline(
        x=yr, line_width=1.5, line_dash='dash', line_color='#E67E22'
    )
    fig1.add_annotation(
        x=yr, y=df['fund_balance_bn'].max() * 0.92,
        text=label, showarrow=False,
        font=dict(size=11, color='#E67E22'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#E67E22', borderwidth=1
    )

fig1.update_layout(
    title='NDIC Fund Balance vs Claims Paid 2010–2024<br>'
          '<sup>Stress years annotated: 2016 oil/FX crisis | 2020 COVID-19</sup>',
    xaxis_title='Year',
    yaxis_title='₦ Billion',
    legend=dict(x=0.01, y=0.99),
    template='plotly_white',
    font=dict(family='Arial', size=13),
    hovermode='x unified'
)

fig1.write_html('resultsfigs/01_fund_vs_claims_timeseries.html')
fig1.write_image('resultsfigs/01_fund_vs_claims_timeseries.png',
                 width=1100, height=550, scale=2)
print("✅  Chart 1 saved: fund vs claims time-series")

# CHART 2 — Benchmark bar chart:
#           NDIC FAR vs IADI peer average and African peers

# African peers in dataset: Nigeria, Tanzania, Uganda
african = iadi[iadi['country'].isin(['Nigeria','Tanzania','Uganda'])].copy()
african_avg = african.groupby('country')['fund_adequacy_ratio'].mean().reset_index()
african_avg.columns = ['country', 'avg_far']

iadi_global_avg = iadi.groupby('year')['fund_adequacy_ratio'].mean().reset_index()
iadi_overall_avg = iadi_global_avg['fund_adequacy_ratio'].mean()

ndic_avg_far = df['local_fund_adequacy_ratio'].mean()

# Build comparison frame
bar_data = pd.DataFrame({
    'Entity': african_avg['country'].tolist() + ['IADI Global Average'],
    'Average FAR 2010–2024': african_avg['avg_far'].tolist() + [iadi_overall_avg]
})

colors = ['#C0392B' if e == 'Nigeria' else
          '#2980B9' if e == 'IADI Global Average' else
          '#7F8C8D' for e in bar_data['Entity']]

fig2, ax2 = plt.subplots(figsize=(10, 6))
bars = ax2.barh(bar_data['Entity'], bar_data['Average FAR 2010–2024'],
                color=colors, edgecolor='white', height=0.55)

# Add value labels
for bar in bars:
    w = bar.get_width()
    ax2.text(w + 0.002, bar.get_y() + bar.get_height()/2,
             f'{w:.4f}', va='center', ha='left', fontsize=11)

ax2.set_xlabel('Average Fund Adequacy Ratio (2010–2024)', fontsize=12)
ax2.set_title('Fund Adequacy Ratio: Nigeria (NDIC) vs\nAfrican Peers & IADI Global Average',
              fontsize=14, fontweight='bold', pad=15)
ax2.axvline(iadi_overall_avg, color='#2980B9', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'IADI Global Avg = {iadi_overall_avg:.4f}')
ax2.legend(fontsize=10)
ax2.set_xlim(0, bar_data['Average FAR 2010–2024'].max() * 1.2)

legend_patches = [
    mpatches.Patch(color='#C0392B', label='Nigeria (NDIC)'),
    mpatches.Patch(color='#7F8C8D', label='African peers (IADI survey)'),
    mpatches.Patch(color='#2980B9', label='IADI Global Average')
]
ax2.legend(handles=legend_patches, fontsize=10, loc='lower right')
ax2.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('resultsfigs/02_benchmark_bar_far.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 2 saved: benchmark bar chart — FAR")

# CHART 3 — Correlation heatmap:
#           NDIC ratios vs macro variables

heatmap_cols = {
    'local_fund_adequacy_ratio' : 'Fund Adequacy\nRatio (local)',
    'claims_intensity'          : 'Claims\nIntensity',
    'coverage_ratio'            : 'Coverage\nRatio',
    'npl_ratio_pct'             : 'NPL\nRatio (%)',
    'car_pct'                   : 'CAR (%)',
    'roa_pct'                   : 'ROA (%)',
    'gdp_growth_lag1'           : 'GDP Growth\n(lag 1yr)',
    'inflation_lag1'            : 'Inflation\n(lag 1yr)',
    'exchange_rate_lag1'        : 'Exchange Rate\n(lag 1yr)'
}

hmap_df = df[[c for c in heatmap_cols if c in df.columns]].copy()
hmap_df.columns = [heatmap_cols[c] for c in hmap_df.columns]
corr = hmap_df.corr()

fig3, ax3 = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    annot_kws={'size': 10}, ax=ax3,
    cbar_kws={'label': 'Pearson r', 'shrink': 0.8}
)
ax3.set_title('Correlation Heatmap: NDIC Financial Ratios vs\nMacro Variables (Nigeria 2010–2024)',
              fontsize=14, fontweight='bold', pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig('resultsfigs/03_correlation_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 3 saved: correlation heatmap")

# CHART 4 — Scatter: coverage ratio vs fund adequacy
#           All IADI member countries, Nigeria highlighted

# Compute averages per country across all years
scatter_base = iadi.merge(
    macro[['country','year','gdp_per_capita']],
    on=['country','year'], how='left'
)
scatter_base['coverage_ratio'] = (
    scatter_base['coverage_limit_usd'] / scatter_base['gdp_per_capita']
)

scatter_avg = scatter_base.groupby('country').agg(
    avg_coverage_ratio    = ('coverage_ratio',      'mean'),
    avg_fund_adequacy     = ('fund_adequacy_ratio',  'mean')
).reset_index()

region_map = {
    'Nigeria':'Africa','Tanzania':'Africa','Uganda':'Africa',
    'USA':'Americas','Canada':'Americas','Brazil':'Americas',
    'India':'Asia','Indonesia':'Asia','Malaysia':'Asia',
    'Philippines':'Asia','Vietnam':'Asia',
    'Japan':'Asia','South Korea':'Asia','Taiwan':'Asia'
}
scatter_avg['region'] = scatter_avg['country'].map(region_map)

region_colors = {
    'Africa'  : '#C0392B',
    'Americas': '#2980B9',
    'Asia'    : '#27AE60'
}

fig4, ax4 = plt.subplots(figsize=(12, 7))

for region, grp in scatter_avg.groupby('region'):
    is_nga = grp['country'] == 'Nigeria'
    not_nga = grp[~is_nga]
    ax4.scatter(
        not_nga['avg_coverage_ratio'],
        not_nga['avg_fund_adequacy'],
        color=region_colors.get(region, '#888888'),
        s=90, alpha=0.75, label=region, zorder=3
    )
    for _, row in not_nga.iterrows():
        ax4.annotate(row['country'],
                     (row['avg_coverage_ratio'], row['avg_fund_adequacy']),
                     textcoords='offset points', xytext=(7, 3),
                     fontsize=8.5, color='#444444')

# Nigeria highlighted
nga_row = scatter_avg[scatter_avg['country'] == 'Nigeria']
ax4.scatter(
    nga_row['avg_coverage_ratio'],
    nga_row['avg_fund_adequacy'],
    color='#F39C12', s=220, zorder=5,
    edgecolors='#333333', linewidths=1.5,
    marker='*', label='Nigeria (NDIC)'
)
ax4.annotate(
    'Nigeria (NDIC)',
    (nga_row['avg_coverage_ratio'].values[0],
     nga_row['avg_fund_adequacy'].values[0]),
    textcoords='offset points', xytext=(10, -14),
    fontsize=10, fontweight='bold', color='#C0392B',
    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5)
)

ax4.set_xlabel('Average Coverage Ratio\n(Coverage Limit USD / GDP per Capita)', fontsize=12)
ax4.set_ylabel('Average Fund Adequacy Ratio\n(Fund Balance / Insured Deposits)', fontsize=12)
ax4.set_title('Coverage Ratio vs Fund Adequacy Ratio\nAll IADI Member Countries — Average 2010–2024\n'
              '(Nigeria highlighted)', fontsize=13, fontweight='bold', pad=12)
ax4.legend(fontsize=10, title='Region', title_fontsize=10)
ax4.spines[['top','right']].set_visible(False)
ax4.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('resultsfigs/04_scatter_coverage_vs_far.png', dpi=180, bbox_inches='tight')
plt.close()
print("✅  Chart 4 saved: scatter — coverage ratio vs FAR")

# CHART 5 — FAR trend: NDIC local vs IADI-reported vs peer avg
#           (Nigeria in focus over time)

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=df['year'], y=df['local_fund_adequacy_ratio'],
    mode='lines+markers', name='NDIC Local FAR',
    line=dict(color='#1D9E75', width=3),
    marker=dict(size=8, symbol='circle')
))

fig5.add_trace(go.Scatter(
    x=df['year'], y=df['fund_adequacy_ratio'],
    mode='lines+markers', name='IADI-Reported FAR (Nigeria)',
    line=dict(color='#E67E22', width=2, dash='dash'),
    marker=dict(size=7, symbol='diamond')
))

fig5.add_trace(go.Scatter(
    x=df['year'], y=df['iadi_avg_fund_adequacy'],
    mode='lines+markers', name='IADI Peer Average (14 countries)',
    line=dict(color='#2980B9', width=2, dash='dot'),
    marker=dict(size=6, symbol='square')
))

# Stress annotations
for yr, label in {2016: '2016: Oil/FX crisis', 2020: '2020: COVID-19'}.items():
    fig5.add_vline(x=yr, line_width=1.2, line_dash='dash', line_color='#95A5A6')
    fig5.add_annotation(
        x=yr, y=df['local_fund_adequacy_ratio'].max() * 0.88,
        text=label, showarrow=False,
        font=dict(size=10, color='#7F8C8D'),
        bgcolor='rgba(255,255,255,0.85)'
    )

fig5.update_layout(
    title='NDIC Fund Adequacy Ratio vs IADI Member Average 2010–2024<br>'
          '<sup>Three series: NDIC local calculation | IADI-reported | IADI peer average</sup>',
    xaxis_title='Year',
    yaxis_title='Fund Adequacy Ratio',
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.85)'),
    template='plotly_white',
    font=dict(family='Arial', size=13),
    hovermode='x unified'
)

fig5.write_html('resultsfigs/05_far_trend_comparison.html')
fig5.write_image('resultsfigs/05_far_trend_comparison.png',
                 width=1100, height=550, scale=2)
print("✅  Chart 5 saved: FAR trend — NDIC vs IADI average")

# PRINT SUMMARY TABLE for Excel key findings paragraph

print("\n── EDA Summary Table ──────────────────────────────────")
summary = df[['year',
              'local_fund_adequacy_ratio',
              'iadi_avg_fund_adequacy',
              'fund_adequacy_gap',
              'claims_intensity',
              'coverage_ratio',
              'npl_ratio_pct',
              'car_pct',
              'gdp_growth_lag1',
              'inflation_lag1']].copy()
print(summary.to_string(index=False))

ndic_far_mean = df['local_fund_adequacy_ratio'].mean()
iadi_far_mean = df['iadi_avg_fund_adequacy'].mean()
best_far_yr   = df.loc[df['local_fund_adequacy_ratio'].idxmax(), 'year']
worst_far_yr  = df.loc[df['local_fund_adequacy_ratio'].idxmin(), 'year']

print(f"\n── Key stats for findings paragraph ──")
print(f"  NDIC avg local FAR (2010–2024) : {ndic_far_mean:.4f}")
print(f"  IADI peer avg FAR  (2010–2024) : {iadi_far_mean:.4f}")
print(f"  Nigeria FAR above peer avg by  : {(ndic_far_mean - iadi_far_mean):.4f}")
print(f"  Best FAR year                  : {best_far_yr}")
print(f"  Lowest FAR year                : {worst_far_yr}")
print(f"  2024 local FAR                 : {df[df['year']==2024]['local_fund_adequacy_ratio'].values[0]:.4f}")
print(f"  2024 claims intensity          : {df[df['year']==2024]['claims_intensity'].values[0]:.4f}")

print("\n✅  All 5 charts saved to resultsfigs/")
