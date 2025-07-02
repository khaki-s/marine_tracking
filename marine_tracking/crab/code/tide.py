import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 读取潮水数据，先读取为普通数据框
tide = pd.read_csv("tide.csv")
# 自定义日期时间解析函数
def parse_custom_datetime(day_str, time_str):
    return pd.to_datetime(f"{day_str} {time_str}", 
                        format="%d-%b-%y %H:%M:%S")

# 应用自定义解析
tide["datetime"] = tide.apply(
    lambda x: parse_custom_datetime(x["day"], x["time"]), axis=1)

# 计算每日潮差
daily_tide = tide.groupby(tide["datetime"].dt.date).agg(
    max_tide=("tidal_range", "max"),
    min_tide=("tidal_range", "min")
).reset_index()

daily_tide["date"] = pd.to_datetime(daily_tide["datetime"])
daily_tide["tidal_range"] = daily_tide["max_tide"] - daily_tide["min_tide"]
# ========== 大潮日期识别 ==========
# # 方法1：按潮差中位数判断
median_tide = daily_tide["tidal_range"].median()
spring_tides = daily_tide[daily_tide["tidal_range"] > median_tide]

# 方法2：取潮差最大的前10%天数（可选）
# top_percentile = daily_tide["tidal_range"].quantile(0.9)
# spring_tides = daily_tide[daily_tide["tidal_range"] > top_percentile]

plt.figure(figsize=(12,6))
ax = sns.lineplot(data=daily_tide,x='date',y='tidal_range',color='blue',label='tidal_range')

# 格式化输出大潮日期
print("=== 大潮日期列表 ===")
for date in spring_tides["date"].dt.strftime("%Y-%m-%d"):
    print(date)
# 标记大潮日期
spring_dates = spring_tides["date"].tolist()
for date in spring_dates:
    plt.axvline(x=date, color='red', alpha=0.3, linestyle='--')

# 设置日期刻度格式
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m-%d"))
plt.title("每日潮差趋势 (红色虚线标记大潮日期)")
plt.xlabel("日期 (月-日)")
plt.ylabel("潮差 (m)")
plt.xticks(rotation=45)
plt.tight_layout()


plt.show()