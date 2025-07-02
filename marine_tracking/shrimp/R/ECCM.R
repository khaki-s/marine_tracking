library(dplyr)
library(zoo)
library(lubridate)
library(stringr)

# —— 1. 读入盲虾数据 ——#
shrimp_raw <- read.csv("D:/khaki/ultralytics-8.3.27/shrimp/distance/run1/2018-2019-2.csv", stringsAsFactors = FALSE)


# 2. 去掉 time 前后的空格，再提取
shrimp <- shrimp_raw %>%
  mutate(
    time_clean = trimws(time),  # 去掉前后空格
    # 提取日期 YYYY-MM-DD
    date0 = as.Date(str_extract(time_clean, "^\\d{4}-\\d{2}-\\d{2}"),
                    format = "%Y-%m-%d"),
    # 提取小时 (1~2 位数字)
    hour0 = as.integer(str_extract(time_clean, 
                                   "(?<=^\\d{4}-\\d{2}-\\d{2} )\\d{1,2}")),
    # 映射：00:00-00:59 -> 前一天，其它 -> 当天
    tide_date = if_else(hour0 < 1, date0 - 1, date0),
    distance = distance_permin_pernumber
  ) %>%
  select(tide_date, distance)

# 2. 生成完整的日期序列
all_dates <- data.frame(
  date = seq(min(shrimp$tide_date),
             max(shrimp$tide_date),
             by = "day")
)

# 3. 左连接，把缺失日补成 NA
shrimp_full <- all_dates %>%
  left_join(rename(shrimp, date = tide_date), by = "date")

# 4. 插值填补 NA
shrimp_full <- shrimp_full %>%
  arrange(date) %>%
  mutate(
    distance_interp = na.approx(distance, x = date, na.rm = FALSE)
  ) %>%
  # 如果开头或结尾仍有 NA，就用最近的非 NA 值补齐
  mutate(
    distance_interp = na.locf(distance_interp, na.rm = FALSE),
    distance_interp = na.locf(distance_interp, fromLast = TRUE)
  )

# 5. 滑动窗口平滑
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
    # 开头不足窗口时，用插值值替代
    distance_smooth = ifelse(
      row_number() < window_size,
      distance_interp,
      distance_smooth
    )
  )



# —— 2. 读入潮汐数据 ——#
tide_raw <- read.csv("D:/khaki/ultralytics-8.3.27/shrimp/tide/2018-2019tide.csv", stringsAsFactors = FALSE) %>%
  mutate(
    # day 例如 "8-Sep-16"，time 例如 "0:59:59"
    day_time = paste(day, time),
    # 用 parse_date_time 自动识别本地时间，不给 tz
    datetime = parse_date_time(day_time, orders = "d-b-y H:M:S"),
    date     = as.Date(datetime),
    height_mm = tidal_range
  ) %>%
  select(datetime, date, height_mm)

# —— 3. 计算日潮差和日潮位 ——#
tide_amp <- tide_raw %>%
  group_by(date) %>%
  summarise(tide_amp = max(height_mm) - min(height_mm))

tide_level <- tide_raw %>%
  filter(format(datetime, "%H:%M:%S") == "23:59:59") %>%
  select(date, tide_level = height_mm)

# —— 4. 合并成日频表并补齐插值 ——#
daily <- shrimp_full %>%
  left_join(tide_amp,   by ="date") %>%
  left_join(tide_level, by = "date") %>%
  arrange(date) %>%
  mutate(
    distance   = na.approx(distance_smooth,   x = date, na.rm = FALSE),
    tide_amp   = na.approx(tide_amp,   x = date, na.rm = FALSE),
    tide_level = na.approx(tide_level, x = date, na.rm = FALSE)
  )

# 6. z-score 标准化
daily <- daily %>%
  mutate(
    dist_z = scale(distance_smooth),#用平滑后的数据
    amp_z  = scale(tide_amp),
    lvl_z  = scale(tide_level)
  )

library(rEDM)
E_best<- 9
tau<-1
# 1. 创建 data.frame，用于分析
df_eccm <- data.frame(
  shrimp = as.numeric(daily$dist_z),
  tide   = as.numeric(daily$amp_z)
)

# —— 3. 扫描正负滞后 tp，计算 tide → shrimp 的 cross‐map skill —— #
#    参考教程 “Time Delays with CCM” (§“Time Delays with CCM”) :contentReference[oaicite:0]{index=0}
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
  mean(out$rho)    # 取平均 rho 作为该 tp 的交叉映射技能
})

# —— 4. 反方向：shrimp → tide —— #
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

# —— 5. 绘图比较两条曲线 —— #
# 先合并两组 rho，计算 y 轴范围（留些余量）
all_rho <- c(eccm_t2s, eccm_s2t)
y_min <- min(all_rho, na.rm = TRUE) - 0.05
y_max <- max(all_rho, na.rm = TRUE) + 0.05

plot(lags, eccm_t2s, type="b", pch=16, col="steelblue",
     xlab="Lag (tp, days)", 
     ylab=expression("Cross-map skill ("*rho*")"),
     main="ECCM: tide → shrimp (blue) vs shrimp → tide (red)",
     ylim=c(y_min, y_max),
     xaxt="n")   # 先不画 x 轴刻度

# 自定义 x 轴：-20 到 20，每 5 天一个刻度
axis(side=1, at=seq(-20, 20, by=5))

# 叠加第二条曲线
lines(lags, eccm_s2t, type="b", pch=16, col="tomato")

# 标出 tp=0 以及 tp=±15
abline(v = c(-15, 0, 15), lty = c(3,2,3), col = "gray40")

# 添加图例
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
