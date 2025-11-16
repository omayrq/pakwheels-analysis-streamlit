# sql_integration.py
# --------------------------------------------------------------
# PakWheels → SQL Server (Windows Auth) + 5 Analyses
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import urllib.parse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# ==================== CONFIG ====================
SERVER   = 'localhost'
DATABASE = 'PakWheelsDB'
CSV_FILE = 'pakwheels_featured_cars.csv'
DRIVER   = '{ODBC Driver 18 for SQL Server}'

# ---------- Windows Authentication (Corrected) ----------
params = urllib.parse.quote_plus(
    f"DRIVER={DRIVER};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"      # <<< MUST BE INSIDE THE STRING
    f"Trusted_Connection=yes;"
    f"TrustServerCertificate=yes;"
)

# Engines
master_engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(
        f'DRIVER={DRIVER};SERVER={SERVER};Trusted_Connection=yes;TrustServerCertificate=yes;'
    )}"
)

db_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# ==================== CREATE DATABASE ====================
print("Connecting to SQL Server (master)...")
with master_engine.connect() as conn:
    conn.execution_options(isolation_level="AUTOCOMMIT")
    conn.execute(text(f"""
        IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = '{DATABASE}')
        BEGIN
            CREATE DATABASE [{DATABASE}]
        END
    """))

print(f"Database '{DATABASE}' ready.\n")

# ==================== CREATE TABLES ====================
print("Ensuring tables exist...")

with db_engine.begin() as conn:
    conn.execute(text("""
    IF OBJECT_ID('Cars') IS NOT NULL DROP TABLE Cars;
    CREATE TABLE Cars (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        ScrapedAt DATETIME2,
        Title NVARCHAR(500),
        Year INT,
        Mileage INT,
        Engine INT,
        Fuel NVARCHAR(50),
        Transmission NVARCHAR(50),
        City NVARCHAR(100),
        PricePKR DECIMAL(18,2),
        Link NVARCHAR(500),
        Notes NVARCHAR(1000),
        InsertedAt DATETIME2 DEFAULT GETDATE()
    )
    """))

    conn.execute(text("""
    IF OBJECT_ID('AnalysisResults') IS NOT NULL DROP TABLE AnalysisResults;
    CREATE TABLE AnalysisResults (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        AnalysisType NVARCHAR(100),
        ResultValue NVARCHAR(500),
        Description NVARCHAR(500),
        InsertedAt DATETIME2 DEFAULT GETDATE()
    )
    """))

print("Tables created in PakWheelsDB.\n")

# ==================== LOAD CSV → Cars ====================
print("Loading CSV data...")
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

if 'Scraped At' in df.columns:
    df = df.rename(columns={'Scraped At': 'ScrapedAt'})
if 'Price (PKR)' in df.columns:
    df = df.rename(columns={'Price (PKR)': 'PricePKR'})

def clean_numeric(series, suffix=None):
    s = series.astype(str)
    if suffix:
        s = s.str.replace(suffix, '', regex=False)
    return pd.to_numeric(s.str.replace(',', '').str.strip().replace(['N/A', 'nan', ''], np.nan), errors='coerce')

df['PricePKR'] = clean_numeric(df['PricePKR'])
df['Mileage']  = clean_numeric(df['Mileage'], ' km')
df['Engine']   = clean_numeric(df['Engine'], 'cc')
df['Year']     = pd.to_numeric(df['Year'], errors='coerce')

df['ScrapedAt'] = df['ScrapedAt'].astype(str).str.replace(r'\s+[A-Z]{3,}$', '', regex=True)
df['ScrapedAt'] = pd.to_datetime(df['ScrapedAt'], errors='coerce')

df_clean = df.dropna(subset=['PricePKR', 'Year', 'Mileage']).copy()

# Final load
columns_to_insert = [
    'ScrapedAt','Title','Year','Mileage','Engine','Fuel',
    'Transmission','City','PricePKR','Link','Notes'
]

df_for_sql = df_clean[columns_to_insert].copy()

print(f"Loading {len(df_for_sql)} cars into SQL Server...")
df_for_sql.to_sql(
    'Cars',
    db_engine,
    if_exists='append',
    index=False,
    method='multi',
    chunksize=1000
)
print("DATA LOADED SUCCESSFULLY!\n")

# ==================== ANALYSES ====================
print("Running 5 analyses...")

df_sql = pd.read_sql("SELECT * FROM Cars", db_engine)

# 1. Mean price
mean_lacs = round(df_sql['PricePKR'].mean() / 100_000, 1)
pd.DataFrame([{
    'AnalysisType': 'Price Distribution',
    'ResultValue': str(mean_lacs),
    'Description': 'Mean price (Lacs PKR)'
}]).to_sql('AnalysisResults', db_engine, if_exists='append', index=False)

# 2. Year-Price correlation
corr_year = round(df_sql['Year'].corr(df_sql['PricePKR']), 3)
pd.DataFrame([{
    'AnalysisType': 'Year-Price Correlation',
    'ResultValue': str(corr_year),
    'Description': 'Pearson correlation'
}]).to_sql('AnalysisResults', db_engine, if_exists='append', index=False)

# 3. Mileage-Price correlation
corr_mile = round(df_sql['Mileage'].corr(df_sql['PricePKR']), 3)
pd.DataFrame([{
    'AnalysisType': 'Mileage-Price Correlation',
    'ResultValue': str(corr_mile),
    'Description': 'Pearson correlation'
}]).to_sql('AnalysisResults', db_engine, if_exists='append', index=False)

# 4. Avg by city
city_avg = df_sql.groupby('City')['PricePKR'].mean().div(100_000).round(1).reset_index()
for _, row in city_avg.iterrows():
    pd.DataFrame([{
        'AnalysisType': f'Avg Price {row["City"]}',
        'ResultValue': str(row["PricePKR"]),
        'Description': 'Avg price in Lacs PKR'
    }]).to_sql('AnalysisResults', db_engine, if_exists='append', index=False)

# 5. Most expensive car
top_price = round(df_sql['PricePKR'].max() / 100_000, 1)
pd.DataFrame([{
    'AnalysisType': 'Top Expensive Car (Lacs)',
    'ResultValue': str(top_price),
    'Description': 'Highest price in Lacs PKR'
}]).to_sql('AnalysisResults', db_engine, if_exists='append', index=False)

print("All analyses saved.\n")

# ==================== OUTPUT ====================
print("Sample cars from SQL:")
print(pd.read_sql("SELECT TOP 3 Title, City, PricePKR, Id, InsertedAt FROM Cars", db_engine))

print("\nRecent analysis results:")
print(pd.read_sql("SELECT TOP 10 * FROM AnalysisResults ORDER BY InsertedAt DESC", db_engine))

# ==================== PLOT ====================
avg_city = pd.read_sql("SELECT City, AVG(PricePKR)/100000 AS AvgLacs FROM Cars GROUP BY City", db_engine)

plt.figure(figsize=(10, 6))
sns.barplot(data=avg_city, x='City', y='AvgLacs', hue='City', palette='viridis', legend=False)
plt.title('Average Price by City')
plt.ylabel('Avg Price (Lacs PKR)')
plt.tight_layout()
plt.savefig('sql_analysis_plot.png')
plt.show()

print("\nSQL INTEGRATION COMPLETE!")
