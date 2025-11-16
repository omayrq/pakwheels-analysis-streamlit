# =============================================
#  PakWheels Featured Cars - Cleaning + 5 Analyses
#  Uses: pandas, numpy, matplotlib
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# === CONFIG ===
CSV_FILE = "pakwheels_featured_cars.csv"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Loading data...")
df = pd.read_csv(CSV_FILE)

print(f"Original shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ===================================
# 1. DATA CLEANING (FIXED & ROBUST)
# ===================================
print("\nCleaning data...")

import re

# === SAFE CLEANING FUNCTION ===
def clean_numeric_column(series, remove_text=None, replace_values=['N/A', 'nan', '']):
    """
    Safely clean a column: remove text, commas, handle N/A, empty strings
    """
    s = series.astype(str)  # Ensure string
    
    if remove_text:
        s = s.str.replace(remove_text, '', regex=False)
    
    s = s.str.replace(',', '', regex=False)   # Remove commas
    s = s.str.strip()                         # Remove whitespace
    s = s.replace(replace_values, np.nan)     # Replace invalid with NaN
    return pd.to_numeric(s, errors='coerce')  # Convert safely

# === APPLY CLEANING ===
# Price: "3,600,000" or "N/A" → 3600000.0 or NaN
df['Price (PKR)'] = clean_numeric_column(df['Price (PKR)'], replace_values=['N/A', 'nan', ''])

# Mileage: "115,163 km" → 115163.0
df['Mileage'] = clean_numeric_column(df['Mileage'], remove_text=' km')

# Engine: "996cc" → 996.0
df['Engine'] = clean_numeric_column(df['Engine'], remove_text='cc')

# Year: already numeric or string → int
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# === Extract Grade from Notes ===
# Handles: "3.5 Grade", "9.7/10", "6.9/10 | Managed...", etc.
df['Grade'] = (
    df['Notes']
    .astype(str)
    .str.extract(r'(\d+(?:\.\d+)?)\s*(?:Grade|/10)')
    .iloc[:, 0]
    .astype(float)
)

# === Extract Photo Count ===
df['Photos'] = (
    df['Notes']
    .astype(str)
    .str.extract(r'(\d+)\s*photos?')
    .iloc[:, 0]
    .astype(float)
)

# === Final: Drop rows missing critical data ===
df_clean = df.dropna(subset=['Price (PKR)', 'Year', 'Mileage']).copy()

print(f"After cleaning: {df_clean.shape[0]} valid car listings")
print(f"Price range: PKR {df_clean['Price (PKR)'].min()/1e6:.1f}M – {df_clean['Price (PKR)'].max()/1e6:.1f}M")
print(f"Years: {int(df_clean['Year'].min())} – {int(df_clean['Year'].max())}")

# ===================================
# 2. ANALYSIS 1: Price Distribution (FIXED + SHOWS GRAPH)
# ===================================
print("\nAnalysis 1: Price Distribution")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Auto-adjust bins if data is small
n_cars = len(df_clean)
bins = min(20, max(5, n_cars // 2))  # At least 5, at most 20

plt.figure(figsize=(10, 6))
sns.histplot(
    data=df_clean,
    x='Price (PKR)',
    bins=bins,
    kde=True,
    color='skyblue',
    alpha=0.8
)
plt.title('Distribution of Car Prices (in Lacs PKR)', fontsize=16, fontweight='bold')
plt.xlabel('Price (PKR Lacs)')
plt.ylabel('Number of Cars')

# Format x-axis in Lacs
xticks = plt.gca().get_xticks()
plt.gca().set_xticklabels([f'{int(x/100000)}' for x in xticks])

plt.tight_layout()

# === SAVE + SHOW ===
save_path = f"{OUTPUT_DIR}/1_price_distribution.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {save_path}")

plt.show()        # This shows the graph inline!
plt.close()

# === PRINT STATS IN LACS ===
price_stats = df_clean['Price (PKR)'].describe()
print("\nPrice Stats (in Lacs PKR):")
stats_lacs = price_stats / 100000
print(stats_lacs.round(2))

# ===================================
# 2. ANALYSIS 2: Price vs Year (FIXED + SHOWS GRAPH)
# ===================================
print("\nAnalysis 2: Price vs Year")

plt.figure(figsize=(11, 6))

# Scatter plot
scatter = sns.scatterplot(
    data=df_clean,
    x='Year',
    y='Price (PKR)',
    hue='City',
    s=120,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.5
)

# Format Y-axis in Lacs
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100000:.0f}L'))

# Titles & labels
plt.title('Car Price vs Model Year', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Price (PKR Lacs)', fontsize=12)

# Legend outside
plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# Tight layout with room for legend
plt.tight_layout()

# === ADD CORRELATION ON PLOT ===
corr_year_price = df_clean['Year'].corr(df_clean['Price (PKR)'])
plt.text(
    0.02, 0.95, f'Correlation: {corr_year_price:.3f}',
    transform=plt.gca().transAxes,
    fontsize=11, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7)
)

# === SAVE + SHOW ===
save_path = f"{OUTPUT_DIR}/2_price_vs_year.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {save_path}")

plt.show()        # This shows the graph!
plt.close()

# === PRINT CORRELATION ===
print(f"Correlation (Year vs Price): {corr_year_price:.3f}")
if corr_year_price > 0.7:
    print("Strong positive: Newer cars are much more expensive.")
elif corr_year_price > 0.3:
    print("Moderate: Newer cars tend to cost more.")
else:
    print("Weak or no clear trend.")

# ===================================
# 3. ANALYSIS 3: Mileage vs Price (FIXED + SHOWS GRAPH)
# ===================================
print("\nAnalysis 3: Mileage vs Price")

plt.figure(figsize=(11, 6))

# -------------------------------------------------
# NOTE: 'linewidth' is an alias of 'linewidths' in newer seaborn.
# Use ONLY ONE of them → we use 'linewidths' for scatter points.
# -------------------------------------------------
sns.regplot(
    data=df_clean,
    x='Mileage',
    y='Price (PKR)',
    scatter_kws={
        'alpha': 0.7,
        's': 80,
        'edgecolor': 'black',
        'linewidths': 0.5   # <-- fixed key
    },
    line_kws={
        'color': 'red',
        'linewidth': 2
    },
    color='steelblue'
)

# ---- Y-axis: Price in Lacs ----
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x/100000:.0f}L')
)

# ---- X-axis: Mileage in thousands (e.g. 115k) ----
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}')
)

# ---- Titles & labels ----
plt.title('Mileage vs Price (with Trend Line)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Mileage (km)', fontsize=12)
plt.ylabel('Price (PKR Lacs)', fontsize=12)

# ---- Correlation on plot ----
corr = df_clean['Mileage'].corr(df_clean['Price (PKR)'])
plt.text(
    0.05, 0.95,
    f'Correlation: {corr:.3f}',
    transform=plt.gca().transAxes,
    fontsize=11, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8)
)

plt.tight_layout()

# ---- SAVE + SHOW ----
save_path = f"{OUTPUT_DIR}/3_mileage_vs_price.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {save_path}")

plt.show()          # <-- graph appears inline
plt.close()

# ---- PRINT CORRELATION + INSIGHT ----
print(f"Correlation (Mileage vs Price): {corr:.3f}")
if corr < -0.7:
    print("Strong negative: Higher mileage → much lower price.")
elif corr < -0.3:
    print("Moderate: More mileage reduces price.")
else:
    print("Weak or no clear trend.")

# ===================================
# 4. ANALYSIS 4: Average Price by City (FIXED + SHOWS GRAPH)
# ===================================
print("\nAnalysis 4: Average Price by City")

import matplotlib.pyplot as plt
import pandas as pd

# Group by City: mean price and count
city_price = (
    df_clean.groupby('City')['Price (PKR)']
    .agg(['mean', 'count'])
    .round(0)
)

# Convert mean to Lacs and round
city_price['mean_lacs'] = (city_price['mean'] / 100000).round(1)

# Sort by average price (descending)
city_price = city_price.sort_values('mean', ascending=False)

# Plot
plt.figure(figsize=(10, 6))

# -------------------------------------------------
# FIXED: 'edgehold' → 'edgecolor'
# -------------------------------------------------
bars = plt.bar(
    city_price.index,
    city_price['mean'] / 100000,
    color='coral',
    alpha=0.85,
    edgecolor='black',   # <-- Correct parameter name
    linewidth=0.8
)

# Title & labels
plt.title('Average Car Price by City', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Avg Price (PKR Lacs)', fontsize=12)
plt.xlabel('City', fontsize=12)

# Add text on top of each bar: "XX.XL (n=YY)"
for i, bar in enumerate(bars):
    height = bar.get_height()
    count = int(city_price.iloc[i]['count'])
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.02 * plt.ylim()[1]),  # Dynamic offset
        f'{height:.1f}L\n(n={count})',
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

# Improve layout
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# === SAVE + SHOW ===
save_path = f"{OUTPUT_DIR}/4_avg_price_by_city.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {save_path}")

plt.show()        # <-- graph appears inline
plt.close()

# === PRINT TABLE ===
print("\nAverage Price by City:")
print(city_price[['mean_lacs', 'count']].rename(
    columns={'mean_lacs': 'Avg Price (Lacs)', 'count': 'Listings'}
).to_string(float_format='%.1f'))

# ===================================
# 5. ANALYSIS 5: Top 10 Most Expensive Cars (FIXED + SHOWS GRAPH)
# ===================================
print("\nAnalysis 5: Top 10 Most Expensive Cars")

import matplotlib.pyplot as plt
import pandas as pd

# Get top 10 (or all if less than 10)
top_n = min(10, len(df_clean))
top10 = df_clean.nlargest(top_n, 'Price (PKR)')[['Title', 'Year', 'City', 'Price (PKR)', 'Mileage', 'Grade']].copy()

# Add Price in Lacs
top10['Price (PKR Lacs)'] = (top10['Price (PKR)'] / 100000).round(1)

# Extract short label: Year + Make + Model (first two words after year)
def short_title(row):
    words = row['Title'].split()
    year = int(row['Year'])
    # Find make (usually word 1), model (word 2)
    make = words[1] if len(words) > 1 else ""
    model = words[2] if len(words) > 2 else ""
    return f"{year} {make} {model}"

top10['Label'] = top10.apply(short_title, axis=1)

# Plot
plt.figure(figsize=(11, 1.2 * top_n))  # Dynamic height

bars = plt.barh(
    range(len(top10)),
    top10['Price (PKR)'] / 100000,
    color='gold',
    alpha=0.9,
    edgecolor='black',
    linewidth=0.8
)

# Y-ticks with clean labels
plt.yticks(range(len(top10)), top10['Label'])
plt.gca().invert_yaxis()  # Highest at top

# Labels
plt.xlabel('Price (PKR Lacs)', fontsize=12)
plt.title(f'Top {top_n} Most Expensive Featured Cars', fontsize=16, fontweight='bold', pad=20)

# Add price text on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{width:.1f}L',
        va='center',
        ha='left',
        fontsize=10,
        fontweight='bold',
        color='black'
    )

# Grid for readability
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()

# === SAVE + SHOW ===
save_path = f"{OUTPUT_DIR}/5_top10_expensive.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Graph saved: {save_path}")

plt.show()        # This shows the graph!
plt.close()

# === PRINT TABLE ===
print(f"\nTop {top_n} Expensive Cars:")
print(top10[['Label', 'Year', 'City', 'Price (PKR Lacs)', 'Mileage']].rename(
    columns={'Label': 'Car'}
).to_string(index=False))

# ===================================
# FINAL SUMMARY (SAFE + NO NAMEERROR)
# ===================================
print("\n" + "=" * 55)
print(" " * 18 + "ANALYSIS COMPLETE!")
print("=" * 55)

# -------------------------------------------------
# SAFE CHECK: Use df_clean only if it exists
# -------------------------------------------------
try:
    df_clean  # Just check if it exists
    n_cars = len(df_clean)
except NameError:
    print("Warning: df_clean not found. Run cleaning & analysis first.")
    print("No data to summarize.")
    print("=" * 55)
else:
    # Only run if df_clean exists
    print(f"Cleaned & Valid Data: {n_cars} car{'s' if n_cars != 1 else ''}")

    if n_cars == 0:
        print("No valid car listings after cleaning.")
    else:
        import pandas as pd

        avg_price_lacs = df_clean['Price (PKR)'].mean() / 100000
        oldest_year = int(df_clean['Year'].min())
        highest_mileage = df_clean['Mileage'].max()
        best_grade = df_clean['Grade'].max()

        print(f"Plots saved in: {OUTPUT_DIR}/")
        print("\nKey Insights:")
        print(f"   • Average Price: PKR {avg_price_lacs:,.1f} Lacs")
        print(f"   • Oldest Car: {oldest_year}")
        print(f"   • Highest Mileage: {highest_mileage:,.0f} km")

        if pd.notna(best_grade):
            print(f"   • Best Rated: {best_grade:.1f}/10")
        else:
            print(f"   • Best Rated: No grade data")

        # Bonus: Most common city
        top_city = df_clean['City'].mode()
        if not top_city.empty:
            city_name = top_city.iloc[0]
            city_count = (df_clean['City'] == city_name).sum()
            print(f"   • Most Listings: {city_name} ({city_count} car{'s' if city_count != 1 else ''})")

print("=" * 55)