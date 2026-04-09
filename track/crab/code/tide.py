import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read tide data 
tide = pd.read_csv("tide.csv")
# Custom date and time parsing function
def parse_custom_datetime(day_str, time_str):
    return pd.to_datetime(f"{day_str} {time_str}", 
                        format="%d-%b-%y %H:%M:%S")

# Apply the custom parsing function
tide["datetime"] = tide.apply(
    lambda x: parse_custom_datetime(x["day"], x["time"]), axis=1)

# Calculate daily tidal range
daily_tide = tide.groupby(tide["datetime"].dt.date).agg(
    max_tide=("tidal_range", "max"),
    min_tide=("tidal_range", "min")
).reset_index()

daily_tide["date"] = pd.to_datetime(daily_tide["datetime"])
daily_tide["tidal_range"] = daily_tide["max_tide"] - daily_tide["min_tide"]
# ========== Identify Spring Tide Dates ==========

# Method 1: Determine based on the median tidal range
median_tide = daily_tide["tidal_range"].median()
spring_tides = daily_tide[daily_tide["tidal_range"] > median_tide]

# Method 2: Select the top 10% of days by tidal range 
# top_percentile = daily_tide["tidal_range"].quantile(0.9)
# spring_tides = daily_tide[daily_tide["tidal_range"] > top_percentile]

plt.figure(figsize=(12,6))
ax = sns.lineplot(data=daily_tide,x='date',y='tidal_range',color='blue',label='tidal_range')

# Output the list of spring tide dates
print("=== List of Spring Tide Dates ===")
for date in spring_tides["date"].dt.strftime("%Y-%m-%d"):
    print(date)
# Mark spring tide dates on the plot
spring_dates = spring_tides["date"].tolist()
for date in spring_dates:
    plt.axvline(x=date, color='red', alpha=0.3, linestyle='--')

# Format the date ticks
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
plt.title("Daily Tidal Range Trend (Red Dashed Lines Mark Spring Tides)")
plt.xlabel("Date (Month-Day)")
plt.ylabel("Tidal Range (m)")
plt.xticks(rotation=45)
plt.tight_layout()


plt.show()