import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime
import os

# === CONFIG ===
URL = "https://www.pakwheels.com/used-cars/search/-/featured_1/"
OUTPUT_CSV = "pakwheels_featured_cars.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}

# === FETCH PAGE ===
print("Fetching data from PakWheels...")
response = requests.get(URL, headers=HEADERS)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# === EXTRACT LISTINGS ===
listings = soup.find_all("li", class_="classified-listing featured-listing")
print(f"Found {len(listings)} featured car(s).")

cars = []

for item in listings:
    car = {}
    
    # Title
    h3 = item.find("h3")
    car["Title"] = h3.get_text(strip=True) if h3 else "N/A"
    
    # JSON-LD Data (most reliable)
    script = item.find("script", type="application/ld+json")
    if script and script.string:
        try:
            data = json.loads(script.string)
            
            car["Year"] = data.get("modelDate", "N/A")
            car["Mileage"] = data.get("mileageFromOdometer", "N/A")
            car["Engine"] = data.get("vehicleEngine", {}).get("engineDisplacement", "N/A")
            car["Fuel"] = data.get("fuelType", "N/A")
            car["Transmission"] = data.get("vehicleTransmission", "N/A")
            
            price = data.get("offers", {}).get("price")
            car["Price (PKR)"] = f"{int(price):,}" if price else "N/A"
            
            # Extract city from description
            desc = data.get("description", "")
            if "for sale in" in desc:
                car["City"] = desc.split("for sale in")[1].strip()
            else:
                car["City"] = "N/A"
                
            car["Link"] = data.get("offers", {}).get("url", "N/A")
        except json.JSONDecodeError:
            car["Year"] = car["Mileage"] = car["Engine"] = car["Fuel"] = car["Transmission"] = car["Price (PKR)"] = car["City"] = car["Link"] = "N/A"
    else:
        car["Year"] = car["Mileage"] = car["Engine"] = car["Fuel"] = car["Transmission"] = car["Price (PKR)"] = car["City"] = car["Link"] = "N/A"

    # === Extra Notes ===
    notes = []
    
    if item.find(string=lambda text: text and "Managed by PakWheels" in text):
        notes.append("Managed by PakWheels")
        
    rating = item.find("span", class_="auction-rating")
    if rating:
        notes.append(rating.get_text(strip=True))
        
    info2 = item.find("ul", class_="search-vehicle-info-2")
    if info2:
        lis = [li.get_text(strip=True) for li in info2.find_all("li")]
        if len(lis) >= 6:
            extra = lis[5]
            if extra and extra not in ["Automatic", "Manual"]:
                notes.append(extra)
    
    photos = item.find("div", class_="total-pictures-bar")
    if photos:
        photo_text = photos.get_text(strip=True).replace('\n', ' ').replace('  ', ' ')
        notes.append(photo_text)

    car["Notes"] = " | ".join(notes) if notes else "N/A"
    
    cars.append(car)

# === CREATE DATAFRAME ===
df = pd.DataFrame(cars)

# === ADD TIMESTAMP ===
df.insert(0, "Scraped At", datetime.now().strftime("%Y-%m-%d %I:%M %p PKT"))

# === SAVE TO CSV ===
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\nSuccess! Data saved to '{OUTPUT_CSV}'")
print(f"Total cars: {len(df)}")
print("\nPreview:")
print(df.head().to_string(index=False))