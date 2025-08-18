#-------Simplex 投影选 E-----
library(rEDM)
library(ggplot2)
# Select required columns
df <- df_full %>% select(starts_with(c("time","value_interp")))
colnames(df) <- c("time", "value")  # rEDM expects column named "value"
# Extract "value" column as input
df_edm <- data.frame(value = df$value)
class(df_edm)
str(df_edm)
head(df_edm)
#Simplex Projection to choose best E,LOOCV
N <- nrow(df_edm)
simplex_out <- simplex(
  time_series  = df_edm$value,
  lib = c(1, N), # Training set indices
  pred = c(1, N),# Prediction set indices
  E = 1:10,# Test different embedding dimensions
  tau = 1#Time delay
  ) 
# Find best E (maximum correlation coefficient rho)
E_best <- simplex_out$E[which.max(simplex_out$rho)]
cat("Best E =", E_best, "\n")

# Define theta scan range (nonlinearity parameter)
theta_vals <- seq(0, 10, by = 0.5)

# Perform S-map nonlinear analysis
smap_out <- s_map(
  time_series = df_edm$value,  
  lib         = c(1, N),
  pred        = c(1, N),
  E           = E_best,
  tau         = 1,
  theta       = theta_vals
)

# View results
print(smap_out)

# Find best theta (maximum rho)
theta_best <- smap_out$theta[which.max(smap_out$rho)]
cat("Best nonlinearity parameter θ* =", theta_best, "\n")


# Convert Simplex LOOCV results to data frame for ggplot
simplex_data <- data.frame(
  E = simplex_out$E,
  rho = simplex_out$rho
)

# Plot Simplex LOOCV using ggplot
ggplot(simplex_data, aes(x = E, y = rho)) +
  geom_point(size = 3, color = "#c4db86ff") +  
  geom_line(color =  "steelblue", size = 1,alpha = 0.8) +               
  labs(
    title = paste("Simplex LOOCV: E_best =",E_best,""),
    x = "Embedding dimension E",
    y = "Forecast skill (rho)"
  ) +
  theme_minimal() +                                   
  theme(
    plot.title = element_text(hjust = 0.5),          
    axis.title.x = element_text(size = 12),          
    axis.title.y = element_text(size = 12),         
    axis.text = element_text(size = 10),             
    panel.grid.major = element_line(linetype = "dashed", color = "gray", size = 0.5) 
  )

# Convert S-map results to data frame for ggplot
smap_data <- data.frame(
  theta = smap_out$theta,
  rho = smap_out$rho
)

# Plot S-map using ggplot
ggplot(smap_data, aes(x = theta, y = rho)) +
  geom_point(size = 3, color = "#c4db86ff", alpha = 0.8) +  
  geom_line(color = "steelblue", size = 1) +               
  geom_vline(aes(xintercept = theta_best), linetype = "dashed", color ="steelblue", size = 1) +  
  labs(
    title = paste("S-map (E =", E_best, ",θ =", theta_best, "): ρ vs θ"),
    x = "Theta (Nonlinearity Parameter)",
    y = "Forecast skill (rho)"
  ) +
  theme_minimal() +                                      
  theme(
    plot.title = element_text(hjust = 0.5),          
    axis.title.x = element_text(size = 12),            
    axis.title.y = element_text(size = 12),            
    axis.text = element_text(size = 10),               
    panel.grid.major = element_line(linetype = "dashed", color = "gray", size = 0.5) 
  )
