library(dplyr)
library(zoo)
library(lubridate)
library(stringr)
library(rEDM)
library(ggplot2)


shrimp_raw <- read.csv(
  "D:/khaki/ultralytics-8.3.27/shrimp/distance/run4/2016-2017-out-for-r.csv",
  stringsAsFactors = FALSE
)

shrimp <- shrimp_raw %>%
  mutate(
    time_clean = trimws(time),
    date0      = as.Date(
      str_extract(time_clean, "^\\d{4}-\\d{2}-\\d{2}"),
      format = "%Y-%m-%d"
    ),
    hour0     = as.integer(
      str_extract(time_clean,
                  "(?<=^\\d{4}-\\d{2}-\\d{2} )\\d{1,2}")
    ),
    tide_date = if_else(hour0 < 1, date0 - 1, date0),
    distance  = distance_permin_pernumber
  ) %>%
  select(tide_date, distance)


all_dates <- data.frame(
  date = seq(min(shrimp$tide_date),
             max(shrimp$tide_date), by = "day")
)

shrimp_full <- all_dates %>%
  left_join(rename(shrimp, date = tide_date), by = "date") %>%
  arrange(date) %>%
  mutate(
    distance_interp = na.approx(distance, x = date, na.rm = FALSE),
    distance_interp = na.locf(distance_interp, na.rm = FALSE),
    distance_interp = na.locf(distance_interp, fromLast = TRUE)
  ) %>%
  mutate(
    distance_smooth = rollapply(
      distance_interp, width = 3, FUN = mean,
      align = "right", fill = NA
    ),
    distance_smooth = ifelse(
      row_number() < 3, distance_interp, distance_smooth
    )
  )


tide_raw <- read.csv(
  "D:/khaki/ultralytics-8.3.27/shrimp/tide/2016-2017tide.csv",
  stringsAsFactors = FALSE
) %>%
  mutate(
    day_time  = paste(day, time),
    datetime  = parse_date_time(day_time, orders = "d-b-y H:M:S"),
    date      = as.Date(datetime),
    height_mm = tidal_range
  ) %>%
  select(datetime, date, height_mm)

tide_amp <- tide_raw %>%
  group_by(date) %>%
  summarise(tide_amp = max(height_mm) - min(height_mm),
            .groups = "drop")

tide_level <- tide_raw %>%
  filter(format(datetime, "%H:%M:%S") == "23:59:59") %>%
  select(date, tide_level = height_mm)

# ── 1-3 合并为日频表 & z-score 标准化 ──
daily <- shrimp_full %>%
  left_join(tide_amp,   by = "date") %>%
  left_join(tide_level, by = "date") %>%
  arrange(date) %>%
  mutate(
    tide_amp   = na.approx(tide_amp,   x = date, na.rm = FALSE),
    tide_level = na.approx(tide_level, x = date, na.rm = FALSE),
    tide_amp   = na.locf(tide_amp,   na.rm = FALSE),
    tide_level = na.locf(tide_level, na.rm = FALSE)
  ) %>%
  mutate(
    dist_z = as.numeric(scale(distance_smooth)),
    amp_z  = as.numeric(scale(tide_amp)),
    lvl_z  = as.numeric(scale(tide_level))
  ) %>%
  filter(!is.na(dist_z))


df_edm <- data.frame(
  time  = seq_len(nrow(daily)),
  value = daily$dist_z
)
N       <- nrow(df_edm)
lib_str <- paste("1", N)


E_list <- 1:10

rho_E <- sapply(E_list, function(e) {
  out <- Simplex(
    dataFrame = df_edm,
    lib       = lib_str,
    pred      = lib_str,
    E         = e,
    tau       = 1,
    Tp        = 1,
    columns   = "value",
    target    = "value",
    verbose   = FALSE
  )
  cor(out$Observations, out$Predictions, use = "complete.obs")
})

simplex_data <- data.frame(E = E_list, rho = rho_E)
E_best       <- E_list[which.max(rho_E)]

cat("=== Simplex 结果 ===\n")
print(simplex_data)
cat("最优 E* =", E_best, "| ρ =", round(max(rho_E), 4), "\n\n")

# 绘图：Simplex
ggplot(simplex_data, aes(x = E, y = rho)) +
  geom_line(color = "steelblue", linewidth = 1, alpha = 0.8) +
  geom_point(size = 3, color = "#c4db86ff") +
  geom_vline(xintercept = E_best, linetype = "dashed",
             color = "tomato", linewidth = 0.9) +
  annotate("text", x = E_best + 0.25, y = min(rho_E) + 0.01,
           label = paste("E* =", E_best),
           color = "tomato", hjust = 0, size = 4) +
  scale_x_continuous(breaks = E_list) +
  labs(
    title = paste("Simplex LOOCV  —  E* =", E_best,
                  " | ρ =", round(max(rho_E), 3)),
    x = "Embedding Dimension  E",
    y = "Forecast skill  ρ"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold"),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10),
    panel.grid.major = element_line(linetype = "dashed",
                                    color = "gray", linewidth = 0.5),
    panel.border     = element_rect(color = "black",
                                    fill = NA, linewidth = 0.8)
  )

#S-map
theta_vals <- c(0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5,
                4, 4.5, 5, 6, 7, 8, 9, 10)
split <- floor(0.7 * N)

lib_str  <- paste("1", split)
pred_str <- paste(split + 1, N)
rho_theta <- sapply(theta_vals, function(th) {
  out <- SMap(
    dataFrame       = df_edm,
    lib             = lib_str,
    pred            = lib_str,
    E               = E_best,
    tau             = 1,
    Tp              = 3,
    theta           = th,
    columns         = "value",
    target          = "value",
    exclusionRadius = 20,   
    verbose         = FALSE
  )
  cor(out$predictions$Observations,
      out$predictions$Predictions,
      use = "complete.obs")
})

smap_data  <- data.frame(E = E_best, theta = theta_vals, rho = rho_theta)
theta_best <- theta_vals[which.max(rho_theta)]
rho_linear <- smap_data$rho[smap_data$theta == 0]
rho_opt    <- max(rho_theta, na.rm = TRUE)
delta_rho  <- rho_opt - rho_linear


print(smap_data[, c("theta", "rho")])

# 绘图：S-map
ggplot(smap_data, aes(x = theta, y = rho)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "#c4db86ff", alpha = 0.8) +
  geom_vline(xintercept = theta_best, linetype = "dashed",
             color = "steelblue", linewidth = 1) +
  geom_hline(yintercept = rho_linear, linetype = "dotted",
             color = "tomato", linewidth = 0.8) +
#  annotate("text",
#           x = max(theta_vals) * 0.55,
#           y = rho_linear + abs(delta_rho) * 0.2,
#           label = sprintf("θ=0 (线性) ρ=%.3f", rho_linear),
#           color = "tomato", size = 3.8) +
#  annotate("text",
#           x = theta_best + 0.3,
#           y = rho_opt - abs(delta_rho) * 0.3,
#           label = sprintf("θ*=%s  ρ=%.3f", theta_best, rho_opt),
#           color = "steelblue", hjust = 0, size = 3.8) +
  scale_x_continuous(breaks = theta_vals) +
  labs(
    title = sprintf("S-map  (E = %d)  —  θ* = %s  |  Δρ = %.3f",
                    E_best, theta_best, delta_rho),
    x = "Theta  θ  (Nonlinearity Parameter)",
    y = "Forecast skill  ρ"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title       = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle    = element_text(hjust = 0.5, color = "gray40"),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10, hjust = 1),
    panel.grid.major = element_line(linetype = "dashed",
                                    color = "gray", linewidth = 0.5),
    panel.border     = element_rect(color = "black",
                                    fill = NA, linewidth = 0.8)
  )
