import pandas as pd
from sqlalchemy import create_engine
import os

engine = create_engine('sqlite:///ndic_iadi.db')

# 1. NDIC annual data
ndic = pd.read_csv('dataraw/ndic_annual_202501.csv')
ndic.columns = ndic.columns.str.strip().str.lower()
ndic = ndic.rename(columns={'year': 'year'})
ndic.to_sql('ndic_raw', engine, if_exists='replace', index=False)
print(f"✅  ndic_raw loaded:      {len(ndic)} rows | columns: {list(ndic.columns)}")

# 2. IADI survey (14 countries × 15 years)
iadi = pd.read_csv('dataraw/iadi_survey_202501.csv')
iadi.columns = iadi.columns.str.strip().str.lower()
iadi.to_sql('iadi_survey', engine, if_exists='replace', index=False)
print(f"✅  iadi_survey loaded:   {len(iadi)} rows | columns: {list(iadi.columns)}")

# 3. World Bank macro (14 countries × 15 years)
macro = pd.read_csv('dataraw/world_bank_macro_202501.csv')
macro.columns = macro.columns.str.strip().str.lower()
macro.to_sql('world_bank_macro', engine, if_exists='replace', index=False)
print(f"✅  world_bank_macro loaded: {len(macro)} rows | columns: {list(macro.columns)}")

print("\n🗄️  All three tables loaded into ndic_iadi.db")
