# Data Sources Log

| File | Source | URL | Date Accessed | Rows | Years |
|------|--------|-----|---------------|------|-------|
| ndic_annual_202501.csv | NDIC Annual Reports (manually compiled) | https://https://www.ndicdatabank.org/banking-industry?id=21 | 2026-07-25 | 15 | 2010–2024 |
| iadi_survey_202501.csv | IADI Annual Survey | https://www.iadi.org/en/about-iadi/annual-survey | 2026-07-25 | 210 | 2010–2024 |
| world_bank_macro_202501.csv | World Bank Global Financial Development Database | https://databank.worldbank.org | 2026-07-25 | 210 | 2010–2024 |

## Notes
- NDIC data covers Nigeria only — 15 annual observations extracted from PDF annual reports
- IADI survey covers 14 member countries: Nigeria, USA, Tanzania, Uganda, Canada, India, Indonesia, Malaysia, Philippines, Brazil, Japan, South Korea, Taiwan, Vietnam
- World Bank macro covers same 14 countries × 15 years
- 2024 NDIC note: Naira devaluation (₦1,479/USD) + coverage limit revision causes divergence between local FAR (0.09) and IADI-reported FAR (0.4749). Both values retained — the gap is a key finding
- USA 2010 FDIC fund balance is negative (post-2008 crisis deficit) — confirmed accurate, flagged in analysis
