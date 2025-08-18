import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import seaborn as sns

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== Data Preprocessing ==========
# Read crab data (new data)
crab = pd.read_csv("./crab/distance_results/2017-2018.csv", parse_dates=["time"])
crab = crab.rename(columns={"time": "date", " distance_permin": "distance/mm"})
crab["date"] = crab["date"].dt.normalize()  # Keep only the date part

# Read tide data
tide = pd.read_csv('./tide.csv')

def parse_custom_datetime(day_str, time_str):
    return pd.to_datetime(f"{day_str} {time_str}", format="%d-%b-%y %H:%M:%S")

tide["datetime"] = tide.apply(lambda x: parse_custom_datetime(x["day"], x["time"]), axis=1)

# Calculate daily tidal range and 24-hour tide height
daily_tide = tide.groupby(tide["datetime"].dt.date).agg(
    max_tide=("tidal_range", "max"),
    min_tide=("tidal_range", "min"),
    tide_24h=("tidal_range", lambda x: x.iloc[-1])  # Get the last tide height of each day
).reset_index()
daily_tide["date"] = pd.to_datetime(daily_tide["datetime"])
daily_tide["tidal_range"] = daily_tide["max_tide"] - daily_tide["min_tide"]

# Merge data
merged = pd.merge(crab, daily_tide, on="date", how="outer").sort_values("date")

# ========== Visualization ==========
plt.figure(figsize=(15, 10))

# Upper subplot: Tide data
ax1 = plt.subplot(2, 1, 1)
ax1.plot(merged["date"], merged["tidal_range"], label="Daily Tidal Range", color="blue")
ax1.plot(merged["date"], merged["tide_24h"], label="24h Tide Height", color="cyan", linestyle="--")
ax1.set_ylabel("Tide Height (m)", fontsize=12)
ax1.legend()
ax1.set_title("Tidal Variation Trend", fontsize=14)
ax1.grid(True, alpha=0.3)

# Lower subplot: Crab movement data
ax2 = plt.subplot(2, 1, 2, sharex=ax1)

# Mark missing values (zeros)
missing_dates = merged[merged["distance/mm"] == 0]["date"]
ax2.scatter(missing_dates, [0]*len(missing_dates), color="red", s=30, zorder=3, label="Missing Values")

# Plot valid data and smooth curve
valid_data = merged[merged["distance/mm"] > 0]
x = valid_data["date"].values.astype(np.int64)  # Convert to numeric for interpolation
y = valid_data["distance/mm"]

# Generate smooth curve
x_new = np.linspace(x.min(), x.max(), 500)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_new)

ax2.plot(pd.to_datetime(x_new), y_smooth, color="orange", label="Movement Trend")
ax2.scatter(valid_data["date"], y, color="green", s=40, zorder=2, label="Valid Data")
ax2.set_ylabel("Movement Distance (mm)", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.legend()
ax2.set_title("Crab Movement Distance", fontsize=14)
ax2.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.show()