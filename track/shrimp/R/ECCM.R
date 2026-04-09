library(dplyr)
library(zoo)
library(lubridate)
library(stringr)

# —— 1. Read shrimp data ——#
shrimp_raw <- read.csv("D:/khaki/ultralytics-8.3.27/shrimp/distance/run1/2018-2019-2.csv", stringsAsFactors = FALSE)


# 2. Remove spaces from time, then extract
shrimp <- shrimp_raw %>%
  mutate(
    time_clean = trimws(time),  # Remove leading and trailing spaces
    # Extract date YYYY-MM-DD
    date0 = as.Date(str_extract(time_clean, "^\\d{4}-\\d{2}-\\d{2}"),
                    format = "%Y-%m-%d"),
    # Extract hour (1~2 digits)
    hour0 = as.integer(str_extract(time_clean, 
                                   "(?<=^\\d{4}-\\d{2}-\\d{2} )\\d{1,2}")),
    # Map: 00:00-00:59 -> previous day, others -> current day
    tide_date = if_else(hour0 < 1, date0 - 1, date0),
    distance = distance_permin_pernumber
  ) %>%
  select(tide_date, distance)

# 2. Generate complete date sequence
all_dates <- data.frame(
  date = seq(min(shrimp$tide_date),
             max(shrimp$tide_date),
             by = "day")
)

# 3. Left join, fill missing days with NA
shrimp_full <- all_dates %>%
  left_join(rename(shrimp, date = tide_date), by = "date")

# 4. Interpolate NA values
shrimp_full <- shrimp_full %>%
  arrange(date) %>%
  mutate(
    distance_interp = na.approx(distance, x = date, na.rm = FALSE)
  ) %>%
  # If there are still NA at the beginning or end, fill with the nearest non-NA value
  mutate(
    distance_interp = na.locf(distance_interp, na.rm = FALSE),
    distance_interp = na.locf(distance_interp, fromLast = TRUE)
  )

# 5. Smoothing with sliding window
window_size <- 3
shrimp_full <- shrimp_full %>%
  mutate(
    distance_smooth = rollapply(
      distance_interp,
      width = window_size,
      FUN   = mean,
      align = "right",
      fill  = NA
    ),
    # For the beginning where the window is not full, use the interpolated value instead
    distance_smooth = ifelse(
      row_number() < window_size,
      distance_interp,
      distance_smooth
    )
  )



# —— 2. Read tide data ——#
tide_raw <- read.csv("D:/khaki/ultralytics-8.3.27/shrimp/tide/2018-2019tide.csv", stringsAsFactors = FALSE) %>%
  mutate(
    # day like "8-Sep-16", time like "0:59:59"
    day_time = paste(day, time),
    # Parse datetime automatically using parse_date_time, without specifying tz
    datetime = parse_date_time(day_time, orders = "d-b-y H:M:S"),
    date     = as.Date(datetime),
    height_mm = tidal_range
  ) %>%
  select(datetime, date, height_mm)

# —— 3. Calculate daily tidal range and tidal level ——#
tide_amp <- tide_raw %>%
  group_by(date) %>%
  summarise(tide_amp = max(height_mm) - min(height_mm))

tide_level <- tide_raw %>%
  filter(format(datetime, "%H:%M:%S") == "23:59:59") %>%
  select(date, tide_level = height_mm)

# —— 4. Merge into a daily table and interpolate missing values ——#
daily <- shrimp_full %>%
  left_join(tide_amp,   by ="date") %>%
  left_join(tide_level, by = "date") %>%
  arrange(date) %>%
  mutate(
    distance   = na.approx(distance_smooth,   x = date, na.rm = FALSE),
    tide_amp   = na.approx(tide_amp,   x = date, na.rm = FALSE),
    tide_level = na.approx(tide_level, x = date, na.rm = FALSE)
  )

# 6. Z-score normalization
daily <- daily %>%
  mutate(
    dist_z = scale(distance_smooth),# Use smoothed data
    amp_z  = scale(tide_amp),
    lvl_z  = scale(tide_level)
  )

library(rEDM)
E_best<- 9
tau<-1
#  Create data frame for analysis
df_eccm <- data.frame(
  shrimp = as.numeric(daily$dist_z),
  tide   = as.numeric(daily$amp_z)
)

# —— Scan positive and negative lags tp, calculate cross-map skill of tide → shrimp —— #
lags <- -18:18
eccm_t2s <- sapply(lags, function(tp) {
  out <- ccm(
    df_eccm,
    E           = E_best,
    tau         = tau,
    lib_column  = "tide",
    target_column = "shrimp",
    lib_sizes   = nrow(df_eccm),
    tp          = tp,
    silent      = TRUE
  )
  mean(out$rho)     # Take average rho as the cross-map skill for this tp

})

# ——  Reverse direction: shrimp → tide —— #
eccm_s2t <- sapply(lags, function(tp) {
  out <- ccm(
    df_eccm,
    E           = E_best,
    tau         = tau,
    lib_column  = "shrimp",
    target_column = "tide",
    lib_sizes   = nrow(df_eccm),
    tp          = tp,
    silent      = TRUE
  )
  mean(out$rho)
})

# ——  Plot comparison of two curves —— #
# First merge both sets of rho, calculate y-axis range (with some margin)

all_rho <- c(eccm_t2s, eccm_s2t)
y_min <- min(all_rho, na.rm = TRUE) - 0.05
y_max <- max(all_rho, na.rm = TRUE) + 0.05

plot(lags, eccm_t2s, type="b", pch=16, col="steelblue",
     xlab="Lag (tp, days)", 
     ylab=expression("Cross-map skill ("*rho*")"),
     main="ECCM: tide → shrimp (blue) vs shrimp → tide (red)",
     ylim=c(y_min, y_max),
     xaxt="n")  # Do not draw x-axis ticks first

# Customize x-axis: from -20 to 20, every 5 days
axis(side=1, at=seq(-20, 20, by=5))

# Overlay second curve
lines(lags, eccm_s2t, type="b", pch=16, col="tomato")

# Mark tp=0 and tp=±15
abline(v = c(-15, 0, 15), lty = c(3,2,3), col = "gray40")

# Add legend
legend("topright", legend=c("tide → shrimp", "shrimp → tide"),
       col=c("steelblue", "tomato"), pch=16, bty="n")

rho_forward  <- eccm_t2s  # tide→shrimp
rho_reverse  <- eccm_s2t  # shrimp→tide
neg_idx <- which(lags < 0)
pos_idx <- which(lags > 0)

max_neg_forward <- max(rho_forward[neg_idx])
max_pos_forward <- max(rho_forward[pos_idx])
delta_forward   <- max_neg_forward - max_pos_forward

max_neg_reverse <- max(rho_reverse[neg_idx])
max_pos_reverse <- max(rho_reverse[pos_idx])
delta_reverse   <- max_neg_reverse - max_pos_reverse
