import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from datetime import datetime, time
import matplotlib.ticker as ticker
from scipy.signal import find_peaks
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

def process_tide_data(tide_df, window_size=3):
    """
    Processes raw tide data to extract daily tidal range and 23:59:59 tide level,
    then interpolates and smooths these daily series.
    Returns processed daily tidal range and 23:59:59 tide level DataFrames.
    """
    # Ensure datetime column is correct
    tide_df['datetime'] = pd.to_datetime(tide_df['day'] + ' ' + tide_df['time'])
    tide_df['date'] = tide_df['datetime'].dt.normalize()

    # 1. Calculate raw daily tidal range (max - min)
    daily_tidal_range_raw = tide_df.groupby('date')['tidal_range'].agg(lambda x: x.max() - x.min()).reset_index()
    daily_tidal_range_raw.rename(columns={'tidal_range': 'daily_range'}, inplace=True)

    # 2. Get raw 23:59:59 tide level
    tide_level_235959_raw = tide_df[tide_df['datetime'].dt.time == time(23, 59, 59)].copy()
    tide_level_235959_raw = tide_level_235959_raw.rename(columns={'tidal_range': 'level_235959'})[['date', 'level_235959']]

    # Create a complete date range for interpolation
    min_date = min(daily_tidal_range_raw['date'].min(), tide_level_235959_raw['date'].min())
    max_date = max(daily_tidal_range_raw['date'].max(), tide_level_235959_raw['date'].max())
    all_dates = pd.DataFrame({'date': pd.date_range(min_date, max_date, freq='D')})

    # Interpolate and smooth daily tidal range
    daily_tidal_range_processed = pd.merge(all_dates, daily_tidal_range_raw, on='date', how='left')
    daily_tidal_range_processed['daily_range_interp'] = daily_tidal_range_processed['daily_range'].interpolate(method='linear')
    daily_tidal_range_processed['daily_range_interp'] = daily_tidal_range_processed['daily_range_interp'].fillna(method='bfill').fillna(method='ffill')
    daily_tidal_range_processed['daily_range_smoothed'] = daily_tidal_range_processed['daily_range_interp'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Interpolate and smooth 23:59:59 tide level
    tide_level_235959_processed = pd.merge(all_dates, tide_level_235959_raw, on='date', how='left')
    tide_level_235959_processed['level_235959_interp'] = tide_level_235959_processed['level_235959'].interpolate(method='linear')
    tide_level_235959_processed['level_235959_interp'] = tide_level_235959_processed['level_235959'].fillna(method='bfill').fillna(method='ffill')
    tide_level_235959_processed['level_235959_smoothed'] = tide_level_235959_processed['level_235959_interp'].rolling(window=window_size, center=True, min_periods=1).mean()

    return daily_tidal_range_processed, tide_level_235959_processed

# ---------------- Configuration ----------------
# Path settings
data_file = "D:/khaki/ultralytics-8.3.27/shrimp/distance/run1/2016-2017.csv"
tide_file = "D:/khaki/marine_tracking/shrimp/tide/2016-2017tide.csv"  # Add tide file path
output_figure = "shrimp_cosinor_analysis.pdf"
column_name = "distance_permin_pernumber"
period_guess = 14.76  # lunar tidal cycle period in days
window_size = 3 # Moving average window size
plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'Arial'
# ---------------- Main Processing ----------------
# Load and prepare the data
print("Loading data...")
file_content = pd.read_csv(data_file)
df_original = load_and_prepare_data(file_content)

# Load tide data
print("Loading tide data...")
tide_data = pd.read_csv(tide_file)
tide_data['datetime'] = pd.to_datetime(tide_data['day'] + ' ' + tide_data['time'])
tide_data['days'] = (tide_data['datetime'] - df_original['time'].min()).dt.total_seconds() / (24 * 3600)

# Process tide data (interpolation and smoothing)
print("\nProcessing tide data...")
daily_tidal_range_processed, tide_level_235959_processed = process_tide_data(tide_data.copy(), window_size)

# Complete date and interpolate for shrimp data
print("\n1. PREPARING SHRIMP DATA (INTERPOLATION & OUTLIER REMOVAL)")
df_original['date'] = df_original['time'].dt.normalize() # Ensure date column is datetime for merging
all_dates_shrimp = pd.DataFrame({'date': pd.date_range(df_original['date'].min(), df_original['date'].max(), freq='D')})
df_full_shrimp = pd.merge(all_dates_shrimp, df_original, on='date', how='left')
df_full_shrimp['distance_permin_pernumber'] = df_full_shrimp['distance_permin_pernumber'].interpolate(method='linear')
df_full_shrimp['time'] = pd.to_datetime(df_full_shrimp['date'])

# Apply IQR outlier removal
print("\n2. ANALYZING WITH OUTLIER REMOVAL (IQR)")
df_no_outliers = remove_outliers_iqr(df_full_shrimp, column_name) # Apply IQR on interpolated full shrimp data
print(f"Removed {len(df_full_shrimp) - len(df_no_outliers)} outliers, {len(df_no_outliers)} data points remain")

# Apply moving average smoothing
print("\n3. APPLYING MOVING AVERAGE")
df_smoothed = apply_moving_average(df_no_outliers, column_name, window_size)
smooth_col = f"{column_name}_smoothed"

# Re-calculate 'days' column for df_smoothed after interpolation, outlier removal, and smoothing
first_day_smoothed = df_smoothed['time'].min()
df_smoothed['days'] = [(t - first_day_smoothed).total_seconds() / (24 * 3600) for t in df_smoothed['time']]
x = df_smoothed['days'].values # This 'x' is used globally by calculate_confidence_bands
y = df_smoothed[smooth_col].values

# Perform cosinor analysis on the smoothed shrimp data
print("\n4. PERFORMING COSINOR ANALYSIS (SHRIMP DATA)")
results = perform_cosinor_analysis(x, y, period_guess)

# --- Match Tide Data to Shrimp Observations ---
print("\nMatching tide data to shrimp observations...")
def get_tide_lookup_date(shrimp_datetime):
    # If shrimp time > 00:00, use previous day's date
    if shrimp_datetime.time() > time(0, 0, 0):
        return (shrimp_datetime - pd.Timedelta(days=1)).normalize()
    # If shrimp time <= 00:00, use current day's date
    else:
        return shrimp_datetime.normalize()

df_smoothed['tide_lookup_date'] = df_smoothed['time'].apply(get_tide_lookup_date)

# Merge matched tidal range
df_smoothed = pd.merge(df_smoothed, daily_tidal_range_processed[['date', 'daily_range_smoothed']],
                       left_on='tide_lookup_date', right_on='date', how='left', suffixes=('', '_matched_range'))
df_smoothed.rename(columns={'daily_range_smoothed': 'matched_tidal_range'}, inplace=True)
df_smoothed.drop(columns=['date_matched_range'], inplace=True, errors='ignore')

# Merge matched 23:59:59 tide level
df_smoothed = pd.merge(df_smoothed, tide_level_235959_processed[['date', 'level_235959_smoothed']],
                       left_on='tide_lookup_date', right_on='date', how='left', suffixes=('', '_matched_level'))
df_smoothed.rename(columns={'level_235959_smoothed': 'matched_tide_level_235959'}, inplace=True)
df_smoothed.drop(columns=['date_matched_level'], inplace=True, errors='ignore')

# ----------------- Print Results -----------------
# Convert phase from radians to degrees for easier interpretation
phi_degrees = np.degrees(results['phi']) % 360
se_phi_degrees = np.degrees(results['se_phi'])
CI_phi_degrees = (np.degrees(results['CI_phi'][0]) % 360, np.degrees(results['CI_phi'][1]) % 360)

print(f"\n---------- Cosinor Analysis Results (Cornelissen 2014 Method) ----------")
print(f"Period: {period_guess:.2f} days (fixed)")
print(f"MESOR: {results['M']:.4f} ± {results['se_M']:.4f}")
print(f"Amplitude: {results['A']:.4f} ± {results['se_A']:.4f}")
print(f"Acrophase: {phi_degrees:.2f}° ± {se_phi_degrees:.2f}°")
print(f"\n95% Confidence Intervals:")
print(f"MESOR: ({results['CI_M'][0]:.4f}, {results['CI_M'][1]:.4f})")
print(f"Amplitude: ({results['CI_A'][0]:.4f}, {results['CI_A'][1]:.4f})")
print(f"Acrophase: ({CI_phi_degrees[0]:.2f}, {CI_phi_degrees[1]:.2f})°")

print(f"\nModel Statistics:")
print(f"R²: {results['r_squared']:.4f}")
print(f"Adjusted R²: {results['adj_r_squared']:.4f}")
print(f"F({results['df_residual']},{2}): {results['F_statistic']:.4f}")
print(f"p-value: {results['p_value']:.8f}")

if results['p_value'] < 0.05:
    print("\nThe rhythm is statistically significant.")
    print("The null hypothesis of no rhythm is rejected.")
else:
    print("\nThe rhythm is not statistically significant.")
    print("The null hypothesis of no rhythm cannot be rejected.")

# ----------------- Create Plots -----------------
# Create a smooth curve for the fitted function (for shrimp data)
days_smooth = np.linspace(df_smoothed['days'].min(), df_smoothed['days'].max(), 1000)
fitted_curve_shrimp = cosinor_model(days_smooth, results['M'], results['A'], results['phi'], period_guess)

# Calculate confidence bands (for shrimp data)
_, lower_band_shrimp, upper_band_shrimp = calculate_confidence_bands(days_smooth, results)

# Convert days back to dates for plotting
first_day_shrimp_plot = df_smoothed['time'].min()
dates_smooth_shrimp = [first_day_shrimp_plot + pd.Timedelta(days=d) for d in days_smooth]

# --- First Figure: Phase-folded and Original Movement Data ---
plt.figure(figsize=(15, 10))

# Plot 1 (First Figure): Phase-folded data with confidence bands
plt.subplot(211)
# Convert to phase units (0-1)
phase = (df_smoothed['days'] / period_guess) % 1.0
# Sort by phase for clearer visualization
phase_order = np.argsort(phase)
phase_sorted = phase.iloc[phase_order]
y_sorted = df_smoothed[smooth_col].iloc[phase_order]

# Plot original phase-folded data
plt.plot(phase_sorted, y_sorted, 'o', alpha=0.6, label='Observed data')

# Create smooth phase values for the fitted curve
phase_fine = np.linspace(0, 1, 200)
# Calculate model values
y_model_shrimp = results['M'] + results['A'] * np.cos(2 * np.pi * phase_fine + results['phi'])

# Plot fitted curve
plt.plot(phase_fine, y_model_shrimp, 'r-', 
        label=f'Cosinor model (Amp = {results["A"]:.4f}, p = {results["p_value"]:.2e})',
        linewidth=2)

# Calculate confidence bands for the phase-folded data
t_phase = np.linspace(0, period_guess, 200)
_, lower_phase, upper_phase = calculate_confidence_bands(t_phase, results)
phase_fine_plot = (t_phase / period_guess) % 1.0

# Sort by phase
phase_sort_idx = np.argsort(phase_fine_plot)
phase_fine_plot = phase_fine_plot[phase_sort_idx]
lower_phase = lower_phase[phase_sort_idx]
upper_phase = upper_phase[phase_sort_idx]

# Plot confidence bands
plt.fill_between(phase_fine_plot, lower_phase, upper_phase, 
                color='#FF0000', alpha=0.2, label='95% Confidence band')

plt.xlabel('Phase (cycles)')
plt.ylabel("distance permin pernumber")
plt.title('Phase-folded Movement Data with 95% Confidence Bands')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2 (First Figure): Original data with fitted curve
ax_fig1_sub2 = plt.subplot(212)
plt.scatter(df_smoothed['time'], df_smoothed[smooth_col], alpha=0.6, label='Smoothed Data Points')
plt.plot(dates_smooth_shrimp, fitted_curve_shrimp, 'r-', 
        label=f'Cosinor Fit: Period = {results["period"]:.2f} days, R² = {results["r_squared"]:.2f}')

# Add confidence bands
plt.fill_between(dates_smooth_shrimp, lower_band_shrimp, upper_band_shrimp, color='#FF0000', alpha=0.2, label='95% Confidence Band')

# Add labels and title
plt.title(f'Cosinor Analysis for Shrimp Movement (Moving Avg Window={window_size})')
plt.xlabel('Date')
plt.ylabel("distance permin pernumber")
plt.legend()

# Format the date axis (for the first figure's second subplot)
min_date_fig1_sub2 = df_smoothed['time'].min()
max_date_fig1_sub2 = df_smoothed['time'].max()
ax_fig1_sub2.set_xlim([min_date_fig1_sub2, max_date_fig1_sub2]) # Set x-axis limit
ax_fig1_sub2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Combine MonthLocator ticks with start and end dates
monthly_ticks_fig1_sub2 = mdates.MonthLocator().tick_values(min_date_fig1_sub2, max_date_fig1_sub2)
all_ticks_fig1_sub2 = sorted(list(set(monthly_ticks_fig1_sub2.tolist() + [mdates.date2num(min_date_fig1_sub2), mdates.date2num(max_date_fig1_sub2)])))
ax_fig1_sub2.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_fig1_sub2))
ax_fig1_sub2.tick_params(axis='x', rotation=45) # Rotate labels
ax_fig1_sub2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Second Figure: Shrimp Movement Data and Tide Data ---
plt.figure(figsize=(15, 10))

# Define the fixed min and max dates for Figure 2 subplots based on shrimp data
fixed_start_date = df_smoothed['time'].min()
fixed_end_date = df_smoothed['time'].max()

# Plot 1 (Second Figure): Original data with fitted curve
ax1 = plt.subplot(211)
plt.scatter(df_smoothed['time'], df_smoothed[smooth_col], alpha=0.6, label='Smoothed Data Points')
plt.plot(dates_smooth_shrimp, fitted_curve_shrimp, 'r-', 
        label=f'Cosinor Fit: Period = {results["period"]:.2f} days, R² = {results["r_squared"]:.2f}')

# Add confidence bands
plt.fill_between(dates_smooth_shrimp, lower_band_shrimp, upper_band_shrimp, color='#FF0000', alpha=0.2, label='95% Confidence Band')

# Add labels and title
plt.title('Shrimp Movement Data with Cosine Fit')
plt.xlabel('Date')
plt.ylabel("distance permin pernumber")
plt.legend()

# Set x-axis limit for the first subplot using fixed dates
ax1.set_xlim([fixed_start_date, fixed_end_date])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Combine MonthLocator ticks with fixed start and end dates
monthly_ticks_ax1_fig2 = mdates.MonthLocator().tick_values(fixed_start_date, fixed_end_date)
all_ticks_ax1_fig2 = sorted(list(set(monthly_ticks_ax1_fig2.tolist() + [mdates.date2num(fixed_start_date), mdates.date2num(fixed_end_date)])))
ax1.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_ax1_fig2))
ax1.tick_params(axis='x') # Rotate labels
ax1.grid(True, alpha=0.3)

# Plot 2 (Second Figure): Tide data with fitted curve and matched levels
ax2 = plt.subplot(212)
# Plot original tide measurements (all high-frequency points)
# plt.plot(tide_data['datetime'], tide_data['tidal_range'], '.', alpha=0.3, markersize=2, label='Original Tide Measurements (Hourly')

# Plot matched daily tidal range and matched 23:59:59 tide level
plt.plot(df_smoothed['time'], df_smoothed['matched_tidal_range'], 'o', color="lightblue", label='Matched Daily Tidal Range', alpha=0.8)
plt.plot(df_smoothed['time'], df_smoothed['matched_tide_level_235959'], 'o', color="#f2a9a2ff", label='Matched 23:59:59 Tide Level', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Tidal Range (m) / Tide Level (m)') # Updated Y-axis label
plt.title(f'Tide Data with Matched Shrimp Movement Time (Window={window_size})')

# Set x-axis limit for the second subplot using fixed dates
ax2.set_xlim([fixed_start_date, fixed_end_date])
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Combine MonthLocator ticks with fixed start and end dates
monthly_ticks_ax2_fig2 = mdates.MonthLocator().tick_values(fixed_start_date, fixed_end_date)
all_ticks_ax2_fig2 = sorted(list(set(monthly_ticks_ax2_fig2.tolist() + [mdates.date2num(fixed_start_date), mdates.date2num(fixed_end_date)])))
ax2.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_ax2_fig2))
ax2.tick_params(axis='x') # Rotate labels
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.show()