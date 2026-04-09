import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from datetime import datetime

# ---------------- Function Definitions ----------------

# Parse the CSV data
def load_and_prepare_data(file_content):
    """
    Load and prepare the data from CSV content
    The format should have 'time' and 'distance_permin_pernumber' columns
    """
    # Parse the CSV content (Remove values of 0)
    df = file_content[file_content['distance_permin_pernumber'] > 0].reset_index(drop=True)
    
    # Convert time strings to datetime objects
    df['time'] = pd.to_datetime(df['time'])
    
    # Convert datetime to numeric values (days since first observation)
    first_day = df['time'].min()
    df['days'] = [(t - first_day).total_seconds() / (24 * 3600) for t in df['time']]
    
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    
    Parameters:
    - df: DataFrame containing the data
    - column: Column name for which to remove outliers
    - multiplier: IQR multiplier for determining outliers (default: 1.5)
    
    Returns: DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def apply_moving_average(df, column, window_size=3):
    """
    Apply moving average smoothing
    
    Parameters:
    - df: DataFrame containing the data
    - column: Column name to smooth
    - window_size: Size of the moving average window (default: 3)
    
    Returns: DataFrame with smoothed column added
    """
    # Create a proper copy of the DataFrame
    df_copy = df.copy()
    
    # Calculate moving average using .loc
    df_copy.loc[:, f'{column}_smoothed'] = df_copy[column].rolling(window=window_size, center=True).mean()
    
    # Fill NaN values that occur at the edges using .loc
    df_copy.loc[:, f'{column}_smoothed'] = df_copy[f'{column}_smoothed'].fillna(df_copy[column])
    
    return df_copy

# Define the cosine function with fixed period for regression
def cosinor_model(t, M, A, phi, period):
    """
    Cosinor model for regression
    
    Parameters:
    - t: time in days
    - M: MESOR (Midline Estimating Statistic Of Rhythm)
    - A: Amplitude
    - phi: Acrophase (phase shift in radians)
    - period: the period in days
    
    Returns: fitted values
    """
    # Convert time to cycles (normalized by period)
    x = t / period
    return M + A * np.cos(2 * np.pi * x + phi)

# Convert cosinor parameters to beta coefficients (and vice versa)
def cosinor_to_beta(A, phi):
    """Convert amplitude and acrophase to beta coefficients"""
    beta1 = A * np.cos(phi)
    beta2 = -A * np.sin(phi)
    return beta1, beta2

def beta_to_cosinor(beta1, beta2):
    """Convert beta coefficients to amplitude and acrophase"""
    A = np.sqrt(beta1**2 + beta2**2)
    phi = np.arctan2(-beta2, beta1)
    return A, phi

# Perform cosinor analysis following Cornelissen (2014)
def perform_cosinor_analysis(t, y, period):
    """
    Perform cosinor analysis as described in Cornelissen (2014)
    
    Parameters:
    - t: time in days
    - y: measurements
    - period: the period in days
    
    Returns: dictionary of results
    """
    n = len(y)
    
    # Construct design matrix X
    x = t / period  # Convert time to cycles
    X = np.column_stack((
        np.ones(n),
        np.cos(2 * np.pi * x),
        np.sin(2 * np.pi * x)
    ))
    
    # Least squares estimation of parameters
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Extract parameters
    M = beta_hat[0]  # MESOR
    beta1 = beta_hat[1]  # Cosine coefficient
    beta2 = beta_hat[2]  # Sine coefficient
    
    # Convert to amplitude and acrophase
    A, phi = beta_to_cosinor(beta1, beta2)
    
    # Calculate fitted values
    y_fit = X @ beta_hat
    
    # Calculate residuals
    residuals = y - y_fit
    
    # Calculate residual sum of squares
    RSS = np.sum(residuals**2)
    
    # Degrees of freedom
    df_total = n - 1
    df_model = 2  # cosine and sine terms
    df_residual = n - 3  # n - (1 + 2) parameters
    
    # Mean square residual (estimate of variance)
    MSR = RSS / df_residual
    
    # Calculate standard errors for all parameters
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(MSR * XtX_inv))
    
    se_M = se_beta[0]
    se_beta1 = se_beta[1]
    se_beta2 = se_beta[2]
    
    # Calculate standard error for amplitude using error propagation
    # SE(A) ≈ sqrt[(β1²*SE(β2)² + β2²*SE(β1)²) / (β1² + β2²)]
    se_A = np.sqrt((beta1**2 * se_beta2**2 + beta2**2 * se_beta1**2) / (beta1**2 + beta2**2))
    
    # Calculate standard error for acrophase using error propagation
    # SE(φ) ≈ sqrt[(β1²*SE(β2)² + β2²*SE(β1)²) / (β1² + β2²)²]
    se_phi = np.sqrt((beta1**2 * se_beta2**2 + beta2**2 * se_beta1**2) / (beta1**2 + beta2**2)**2)
    
    # Calculate total sum of squares
    TSS = np.sum((y - np.mean(y))**2)
    
    # Calculate R-squared
    r_squared = 1 - RSS / TSS
    
    # Calculate adjusted R-squared
    adj_r_squared = 1 - (RSS / df_residual) / (TSS / df_total)
    
    # F-test for rhythm detection (null hypothesis: no rhythm)
    MSM = (TSS - RSS) / df_model  # Mean square due to model
    F_statistic = MSM / MSR
    p_value = 1 - stats.f.cdf(F_statistic, df_model, df_residual)
    
    # Calculate 95% confidence intervals
    t_crit = stats.t.ppf(0.975, df_residual)  # two-tailed, 95%
    
    CI_M = (M - t_crit * se_M, M + t_crit * se_M)
    CI_A = (A - t_crit * se_A, A + t_crit * se_A)
    CI_phi = (phi - t_crit * se_phi, phi + t_crit * se_phi)
    
    # Package all results
    results = {
        'M': M,
        'A': A,
        'phi': phi,
        'beta1': beta1,
        'beta2': beta2,
        'period': period,
        'se_M': se_M,
        'se_A': se_A,
        'se_phi': se_phi,
        'CI_M': CI_M,
        'CI_A': CI_A,
        'CI_phi': CI_phi,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'F_statistic': F_statistic,
        'p_value': p_value,
        'RSS': RSS,
        'df_residual': df_residual,
        'MSR': MSR
    }
    
    return results

def calculate_confidence_bands(t, results, alpha=0.05):
    """
    Calculate confidence bands for the cosinor model
    
    Parameters:
    - t: time points for evaluation
    - results: results from cosinor analysis
    - alpha: significance level (default: 0.05 for 95% confidence)
    
    Returns: tuple of (y_pred, lower_band, upper_band)
    """
    # Construct design matrix for new time points
    x_cycles = t / results['period']
    X_new = np.column_stack((
        np.ones(len(t)),
        np.cos(2 * np.pi * x_cycles),
        np.sin(2 * np.pi * x_cycles)
    ))
    
    # Original design matrix
    x_original = x / results['period']
    X = np.column_stack((
        np.ones(len(x)),
        np.cos(2 * np.pi * x_original),
        np.sin(2 * np.pi * x_original)
    ))
    
    # Inverse of X'X
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Calculate confidence bands
    y_pred = X_new @ np.array([results['M'], results['beta1'], results['beta2']])
    
    # Standard error of prediction for each point
    se_pred = np.zeros(len(t))
    for i in range(len(t)):
        x_i = X_new[i:i+1, :]
        # Standard error of the fit
        se_fit = np.sqrt(results['MSR'] * (x_i @ XtX_inv @ x_i.T).item())
        # Standard error of prediction (includes uncertainty of future observations)
        se_pred[i] = np.sqrt(results['MSR'] + se_fit**2)
    
    # Calculate confidence bands
    t_crit = stats.t.ppf(1 - alpha/2, results['df_residual'])
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred
    
    return y_pred, lower, upper

# ---------------- Configuration ----------------
data_file = "D:/khaki/ultralytics-8.3.27/shrimp/distance/run4/2016-2017-out.csv"
column_name = "distance_permin_pernumber"
window_size = 3  # Moving average window size

plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = 'Arial'

# ---------------- Main Processing ----------------
print("Loading and preparing data...")
file_content = pd.read_csv(data_file)
df_original = load_and_prepare_data(file_content)

# Apply IQR outlier removal
print("Removing outliers...")
df_no_outliers = remove_outliers_iqr(df_original, column_name)
print(f"Removed {len(df_original) - len(df_no_outliers)} outliers, {len(df_no_outliers)} data points remain")

# Apply moving average smoothing
print("Applying moving average...")
df_smoothed = apply_moving_average(df_no_outliers, column_name, window_size)
smooth_col = f"{column_name}_smoothed"

# Set global x and y for analysis
x = df_smoothed['days'].values
y = df_smoothed[smooth_col].values
# x = df_no_outliers["days"]
# y = df_no_outliers["distance_permin_pernumber"]
# ---------------- Period Optimization ----------------
print("\nPerforming period optimization (10-30 days, step=0.1)...")

# Define period range
periods = np.arange(10.0, 30.0, 0.01)
results_list = []

print(f"Testing {len(periods)} different periods...")

# Detrend using local regression：
import statsmodels.api as sm
lowess = sm.nonparametric.lowess(y, x, frac=0.15)
trend = lowess[:,1]
y_detr = y - trend

for i, period in enumerate(periods):
    if i % 50 == 0:  # Print progress every 50 iterations
        print(f"Progress: {i+1}/{len(periods)} periods tested ({period:.1f} days)")
    
    try:
        results = perform_cosinor_analysis(x, y_detr, period)
        results_list.append({
            'period': period,
            'r_squared': results['r_squared'],
            'adj_r_squared': results['adj_r_squared'],
            'p_value': results['p_value'],
            'amplitude': results['A'],
            'F_statistic': results['F_statistic']
        })
    except Exception as e:
        print(f"Error at period {period:.1f}: {e}")
        continue

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(results_list)

# Find optimal periods
best_r2_idx = results_df['r_squared'].idxmax()
best_p_value_idx = results_df['p_value'].idxmin()

optimal_period = results_df.loc[best_r2_idx, 'period']
optimal_r2 = results_df.loc[best_r2_idx, 'r_squared']
optimal_period_p = results_df.loc[best_p_value_idx, 'period']
optimal_p = results_df.loc[best_p_value_idx, 'p_value']

print(f"\n---------- Optimization Results ----------")
print(f"Best R² = {optimal_r2:.6f} at period = {optimal_period:.1f} days")
print(f"Best p-value = {optimal_p:.2e} at period = {optimal_period_p:.1f} days")

# Check if 14.76 days is close to optimal
target_period = 14.76
closest_idx = np.abs(results_df['period'] - target_period).idxmin()
target_results = results_df.loc[closest_idx]
target_r_squared = target_results['r_squared']
target_p = target_results["p_value"]
print(f"\nResults at {target_period} days:")
print(f"R² = {target_results['r_squared']:.6f}")
print(f"p-value = {target_results['p_value']:.2e}")

# ---------------- Create Plots ----------------
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

# Plot R² vs Period on left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Period (days)')
ax1.set_ylabel('R²', color=color1)
line1 = ax1.plot(results_df['period'], results_df['r_squared'], color=color1, 
                linewidth=2, alpha=0.8, label='R²')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(10, 30)

# Create second y-axis for p-value
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('P-value (log scale)', color=color2)
line2 = ax2.semilogy(results_df['period'], results_df['p_value'], color=color2, 
                    linewidth=2, alpha=0.8, label='P-value')
ax2.tick_params(axis='y', labelcolor=color2)

# Add significance level line
ax2.axhline(y=0.05, color='red', linestyle=':', alpha=0.7, linewidth=1, label='α = 0.05')

# Add vertical lines for optimal periods and target_period days
ax1.axvline(x=optimal_period, color='red', linestyle='--', alpha=0.7, linewidth=2,
           label=f'Max R² at {optimal_period:.2f} days')
ax1.axvline(x=target_period, color='orange', linestyle='--', alpha=0.8, linewidth=2,
           label=f'{target_period} days (lunar cycle)')

# Add grid
ax1.grid(True, alpha=0.3)

# Create combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

# Set title
ax1.set_title(f'Cosinor Analysis: R² and P-value vs Period\n(Max {optimal_period:.2f} days R² = {optimal_r2:.5f}, Min p = {optimal_p:.2e})\n({target_period} days R² = {target_r_squared:.5f}, Min p = {target_p:.2e}))', 
            fontsize=14, pad=20)

plt.tight_layout()
plt.show()

# ---------------- Detailed Analysis at Optimal Period ----------------
print(f"\n---------- Detailed Analysis at Optimal Period ({optimal_period:.2f} days) ----------")
optimal_results = perform_cosinor_analysis(x, y_detr, optimal_period)

# Convert phase from radians to degrees
phi_degrees = np.degrees(optimal_results['phi']) % 360
se_phi_degrees = np.degrees(optimal_results['se_phi'])
CI_phi_degrees = (np.degrees(optimal_results['CI_phi'][0]) % 360, 
                  np.degrees(optimal_results['CI_phi'][1]) % 360)

print(f"MESOR: {optimal_results['M']:.4f} ± {optimal_results['se_M']:.4f}")
print(f"Amplitude: {optimal_results['A']:.4f} ± {optimal_results['se_A']:.4f}")
print(f"Acrophase: {phi_degrees:.2f}° ± {se_phi_degrees:.2f}°")
print(f"R²: {optimal_results['r_squared']:.6f}")
print(f"Adjusted R²: {optimal_results['adj_r_squared']:.6f}")
print(f"F-statistic: {optimal_results['F_statistic']:.4f}")
print(f"p-value: {optimal_results['p_value']:.2e}")

if optimal_results['p_value'] < 0.05:
    print("\nThe rhythm is statistically significant at the optimal period.")
else:
    print("\nThe rhythm is not statistically significant at the optimal period.")

# Save results to CSV
results_df.to_csv('period_optimization_results.csv', index=False)
print(f"\nResults saved to 'period_optimization_results.csv'")

print(f"\n---------- Summary ----------")
print(f"Total periods tested: {len(periods)}")
print(f"Period range: {periods[0]:.2f} - {periods[-1]:.2f} days")
print(f"Step size: 0.1 days")
print(f"Optimal period (highest R²): {optimal_period:.2f} days")
print(f"Maximum R²: {optimal_r2:.6f}")
print(f"Minimum p-value: {optimal_p:.2e} at {optimal_period_p:.2f} days")

# Check how close target_period is to optimal
distance_to_optimal = abs(target_period - optimal_period)
print(f"{target_period} days is {distance_to_optimal:.2f} days away from the optimal period")