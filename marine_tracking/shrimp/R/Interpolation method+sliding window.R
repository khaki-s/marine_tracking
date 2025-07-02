#load required packages
library(zoo)
library(dplyr)

#-----数据预处理-----
# 1. Read the data
df_raw <- read.csv("D:/khaki/ultralytics-8.3.27/shrimp/distance/run3/2018-2019-1.csv", stringsAsFactors = FALSE)

# 2. 转换 time 列为 Date
df_daily <- df_raw %>%
  mutate(time = as.Date(time)) %>%
  ungroup()

# 2. Generate a complete date sequence from earliest to latest
all_dates <- data.frame(
  time = seq(min(df_daily$time),
             max(df_daily$time),
             by = "day")
)


# 3. Left join, fill missing dates with NA
df_full <- all_dates %>%
  left_join(df_daily, by = "time") %>%
  arrange(time)

# 4. Linear interpolation to fill NA values
df_full$value_interp <- na.approx(df_full$distance_permin_pernumber, x = df_full$time, na.rm = FALSE)

# If there are NAs at the beginning or end, fill them with the last non-NA value
df_full$value_interp <- na.locf(df_full$value_interp, na.rm = FALSE)
df_full$value_interp <- na.locf(df_full$value_interp, fromLast = TRUE)

# 5.sliding window average, right-aligned
window_size <- 1 #set the window size

df_full$value_smooth <- rollapply(
  df_full$value_interp,
  width = window_size,
  FUN = mean,
  align = "right",
  fill = NA
)

# 6. For rows at the beginning that are less than the window size, directly fill with interpolated values
df_full$value_smooth[1:(window_size-1)] <- df_full$value_interp[1:(window_size-1)]



