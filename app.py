# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========================================
# CONFIG
# ========================================
st.set_page_config(page_title="PakWheels Featured Cars Analysis", layout="wide")
st.title("PakWheels Featured Cars - Data Analysis Dashboard")
st.markdown("**Scraped → Cleaned → 5 Visual Insights**")

CSV_FILE = "pakwheels_featured_cars.csv"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# LOAD & CLEAN DATA
# ========================================
@st.cache_data
def load_and_clean():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        st.error(f"CSV file not found: `{CSV_FILE}`. Please run the scraper first.")
        st.stop()

    # --- Safe cleaning ---
    def clean_numeric(series, suffix=None):
        s = series.astype(str)
        if suffix:
            s = s.str.replace(suffix, '', regex=False)
        return pd.to_numeric(
            s.str.replace(',', '').str.strip().replace(['N/A', 'nan', ''], np.nan),
            errors='coerce'
        )

    df['Price (PKR)'] = clean_numeric(df['Price (PKR)'])
    df['Mileage'] = clean_numeric(df['Mileage'], ' km')
    df['Engine'] = clean_numeric(df['Engine'], 'cc')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Grade & Photos
    df['Grade'] = df['Notes'].astype(str).str.extract(r'(\d+\.\d+|\d+)\s*(?:Grade|/10)').iloc[:, 0].astype(float)
    df['Photos'] = df['Notes'].astype(str).str.extract(r'(\d+)\s*photos?').iloc[:, 0].astype(float)

    df_clean = df.dropna(subset=['Price (PKR)', 'Year', 'Mileage']).copy()
    return df_clean

df_clean = load_and_clean()

# ========================================
# SUMMARY
# ========================================
st.sidebar.header("Summary")
st.sidebar.metric("Total Valid Cars", len(df_clean))
if len(df_clean) > 0:
    st.sidebar.metric("Avg Price", f"PKR {df_clean['Price (PKR)'].mean()/100000:,.1f} Lacs")
    st.sidebar.metric("Oldest Year", int(df_clean['Year'].min()))
    st.sidebar.metric("Highest Mileage", f"{df_clean['Mileage'].max():,.0f} km")

# ========================================
# ANALYSIS 1: Price Distribution
# ========================================
st.header("1. Price Distribution")
col1, col2 = st.columns([3, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = min(20, max(5, len(df_clean)//2))
    sns.histplot(df_clean['Price (PKR)']/100000, bins=bins, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Car Prices', fontweight='bold')
    ax.set_xlabel('Price (PKR Lacs)')
    ax.set_ylabel('Count')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("**Stats (Lacs)**")
    stats = (df_clean['Price (PKR)'].describe() / 100000).round(2)
    st.dataframe(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']], use_container_width=True)

# ========================================
# ANALYSIS 2: Price vs Year
# ========================================
st.header("2. Price vs Model Year")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='Year', y='Price (PKR)', hue='City', s=100, alpha=0.8, ax=ax, edgecolor='black')
ax.set_title('Price vs Year', fontweight='bold')
ax.set_ylabel('Price (PKR Lacs)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/100000:.0f}L'))
corr = df_clean['Year'].corr(df_clean['Price (PKR)'])
ax.text(0.02, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        bbox=dict(facecolor='lightgray', alpha=0.8), fontweight='bold')
plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)
plt.close()

# ========================================
# ANALYSIS 3: Mileage vs Price
# ========================================
st.header("3. Mileage vs Price")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(data=df_clean, x='Mileage', y='Price (PKR)',
            scatter_kws={'alpha':0.7, 's':80, 'edgecolor':'black', 'linewidths':0.5},
            line_kws={'color':'red', 'linewidth':2}, color='steelblue', ax=ax)
ax.set_title('Mileage vs Price (Trend Line)', fontweight='bold')
ax.set_xlabel('Mileage (km)')
ax.set_ylabel('Price (PKR Lacs)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/100000:.0f}L'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
corr = df_clean['Mileage'].corr(df_clean['Price (PKR)'])
ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
        bbox=dict(facecolor='lightcoral', alpha=0.8), fontweight='bold')
st.pyplot(fig)
plt.close()

# ========================================
# ANALYSIS 4: Avg Price by City
# ========================================
st.header("4. Average Price by City")
city_stats = df_clean.groupby('City')['Price (PKR)'].agg(['mean', 'count']).round(0)
city_stats['mean_lacs'] = (city_stats['mean'] / 100000).round(1)
city_stats = city_stats.sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(city_stats.index, city_stats['mean']/100000, color='coral', alpha=0.85, edgecolor='black')
ax.set_title('Average Price by City', fontweight='bold')
ax.set_ylabel('Avg Price (PKR Lacs)')
for i, bar in enumerate(bars):
    h = bar.get_height()
    c = int(city_stats.iloc[i]['count'])
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}L\n(n={c})',
            ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
st.pyplot(fig)
plt.close()

st.dataframe(city_stats[['mean_lacs', 'count']].rename(columns={'mean_lacs': 'Avg (Lacs)', 'count': 'Listings'}), use_container_width=True)

# ========================================
# ANALYSIS 5: Top 10 Expensive
# ========================================
st.header("5. Top 10 Most Expensive Cars")
top_n = min(10, len(df_clean))
top10 = df_clean.nlargest(top_n, 'Price (PKR)')[['Title', 'Year', 'City', 'Price (PKR)', 'Mileage']].copy()
top10['Price Lacs'] = (top10['Price (PKR)']/100000).round(1)

def short_label(row):
    words = row['Title'].split()
    return f"{int(row['Year'])} {words[1]} {words[2]}"

top10['Label'] = top10.apply(short_label, axis=1)

fig, ax = plt.subplots(figsize=(10, 1.2 * top_n))
bars = ax.barh(range(len(top10)), top10['Price (PKR)']/100000, color='gold', alpha=0.9, edgecolor='black')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['Label'])
ax.invert_yaxis()
ax.set_xlabel('Price (PKR Lacs)')
ax.set_title(f'Top {top_n} Most Expensive Cars', fontweight='bold')
for i, bar in enumerate(bars):
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}L', va='center', fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
st.pyplot(fig)
plt.close()

st.dataframe(top10[['Label', 'City', 'Price Lacs', 'Mileage']].rename(columns={'Label': 'Car'}), use_container_width=True)

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption(f"Data scraped on: {pd.Timestamp('today').strftime('%Y-%m-%d %I:%M %p')} PKT | Total: {len(df_clean)} cars")