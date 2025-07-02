import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import f

# ---------------- Configuration ----------------
# Path settings
tide_file = "D:/khaki/ultralytics-8.3.27/shrimp/tide/2016-2017tide.csv"

# Parse the CSV data
def load_and_prepare_data(tide_file):
    # Parse the CSV content
    tide = pd.read_csv(tide_file)
    #Merge the day and time columns and resolve to datetime
    tide['datetime'] = pd.to_datetime(
    tide['day'] + ' ' + tide['time'],
    format='%d-%b-%y %H:%M:%S'#%d recognizes English abbreviation
    )

    return tide

# Define the cosine function for fitting
def cosine_func(x, amplitude, frequency, phase, offset):
    """
    Cosine function: amplitude * cos(2π * frequency * x + phase) + offset
    
    Parameters:
    - x: time in days
    - amplitude: the peak deviation from the offset
    - frequency: number of cycles per day
    - phase: phase shift in radians
    - offset: vertical shift
    
    Returns: cosine values
    """
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

# Calculate R-squared and p-value
def calculate_fit_statistics(y_true, y_pred, num_params):
    """
    Calculate the coefficient of determination (R-squared) and p-value
    
    Parameters:
    - y_true: actual values
    - y_pred: predicted values
    - num_params: number of parameters in the model
    
    Returns: R-squared value and p-value
    """
    n = len(y_true)  # Number of data points
    
    # Calculate residuals and sum of squares
    residuals = y_true - y_pred
    residual_sum_squares = np.sum(residuals ** 2)
    total_sum_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calculate R-squared
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    # Calculate adjusted R-squared
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - num_params - 1)
    
    # Calculate F-statistic and p-value
    # For regression through the origin, dof is n-p
    # For regression with intercept, dof is n-p-1 (where p is number of params excluding intercept)
    dof_model = num_params  # Degrees of freedom for model (number of parameters)
    dof_residual = n - num_params  # Residual degrees of freedom
    
    # Mean squares
    mean_square_model = (total_sum_squares - residual_sum_squares) / dof_model
    mean_square_residual = residual_sum_squares / dof_residual
    
    # F-statistic
    f_statistic = mean_square_model / mean_square_residual if mean_square_residual != 0 else 0
    
    # p-value
    p_value = 1 - f.cdf(f_statistic, dof_model, dof_residual)
    
    return r_squared, adj_r_squared, f_statistic, p_value

# ---------------- Main Processing ----------------
# Main function to run the analysis
# Load and prepare the data
df = load_and_prepare_data(tide_file)
df['day'] = df['datetime'].dt.date
daily_tidal_range = df.groupby('day')['tidal_range'].agg(lambda x: x.max() - x.min())
daily_tidal_range = daily_tidal_range.reset_index()
daily_tidal_range['days_since_start'] = (pd.to_datetime(daily_tidal_range['day']) - pd.to_datetime(daily_tidal_range['day'].iloc[0])).dt.days
x_days = (df['datetime'] - df['datetime'].iloc[0]).map(lambda x: x.total_seconds()) / (24 * 60 * 60)
# Convert datetime to days since epoch for curve fitting
x = daily_tidal_range['days_since_start']
y = daily_tidal_range['tidal_range']

# Initial guess for the parameters
initial_guess = [y.max() - y.min(), 1/14.76, 0, y.mean()]

# Perform curve fitting
popt, pcov = curve_fit(cosine_func, x, y, p0=initial_guess)

# Extract optimized parameters
amplitude, frequency, phase, offset = popt

# Calculate parameter errors from covariance matrix
param_errors = np.sqrt(np.diag(pcov))
amplitude_err, frequency_err, phase_err, offset_err = param_errors

# Calculate confidence intervals (95%)
confidence = 0.95
alpha = 1.0 - confidence
n = len(y)  # Number of data points
p = len(popt)  # Number of parameters
dof = max(0, n - p)  # Degrees of freedom
tval = stats.t.ppf(1.0 - alpha/2.0, dof)  # Student-t value for the confidence level

# Calculate confidence intervals for each parameter
ci = []
for i, p in enumerate(popt):
    sigma = param_errors[i]
    ci.append((p - sigma * tval, p + sigma * tval))

# Calculate predicted values
y_pred = cosine_func(x, *popt)

# Calculate fit statistics
r_squared, adj_r_squared, f_statistic, p_value = calculate_fit_statistics(y, y_pred, len(popt))

# Print results
print(f"拟合结果：振幅={amplitude:.4f} ± {param_errors[0]:.4f},频率={frequency:.6f} ± {param_errors[1]:.6f},相位={phase:.4f} ± {param_errors[2]:.4f},偏移={offset:.4f} ± {param_errors[3]:.4f}")
print(f"周期是{1/frequency:.2f} ± {frequency_err/(frequency**2):.2f} 天")
print(f"R-squared: {r_squared:.4f}, Adjusted R-squared: {adj_r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.8f}")
print("\n95% Confidence Intervals:")
param_names = ["Amplitude", "Frequency", "Phase", "Offset"]
for i, (param, interval) in enumerate(zip(param_names, ci)):
    print(f"{param}: ({interval[0]:.6f}, {interval[1]:.6f})")
print(f"Period (days): ({1/(frequency + frequency_err*tval):.2f}, {1/(frequency - frequency_err*tval):.2f})")

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original data
plt.subplot(211)
plt.plot(df['datetime'], df['tidal_range'], 'o', label='Original Data', alpha=0.3, markersize=3)
plt.plot(df['datetime'], cosine_func(x_days, *popt), 'r-', 
         label=f'Fitted Curve: Period = {1/frequency:.2f} days, R² = {r_squared:.4f}, p = {p_value:.2e}', 
         linewidth=2)
plt.xlabel('Date')
plt.ylabel('Tidal Range (m)')
plt.title(f'Cosine Fit of Tidal Data')
plt.legend()
plt.grid(True)

# Plot daily data with confidence bands
plt.subplot(212)
plt.plot(daily_tidal_range['day'], daily_tidal_range['tidal_range'], 'o', label='Daily Range', alpha=0.6)
plt.plot(daily_tidal_range['day'], y_pred, 'r-', 
         label=f'Fit: R² = {r_squared:.4f}, p = {p_value:.2e}')

# Generate confidence bands
x_fine = np.linspace(min(x), max(x), 1000)
y_fine = cosine_func(x_fine, *popt)

# Calculate prediction intervals
sigma = np.sqrt(np.sum((y - y_pred)**2) / dof)
pred_error = sigma * tval * np.sqrt(1 + 1/n)

# Generate corresponding dates for plotting
date_0 = pd.to_datetime(daily_tidal_range['day'].iloc[0])
x_dates = [date_0 + pd.Timedelta(days=float(i)) for i in x_fine]

# Plot confidence bands
plt.fill_between(x_dates, 
                 cosine_func(x_fine, *popt) - pred_error, 
                 cosine_func(x_fine, *popt) + pred_error, 
                 color='red', alpha=0.2, label='95% Confidence Band')

plt.xlabel('Date')
plt.ylabel('Daily Tidal Range (m)')
plt.title('Daily Tidal Range with 95% Confidence Bands')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tide_fit_with_confidence.png', dpi=300)
plt.show()