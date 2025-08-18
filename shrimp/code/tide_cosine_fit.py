import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import math

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
def cosine_func(x, M, A, phi):
    """
    Cosine function: M + A * cos(2π * x + phi)
    
    Parameters:
    - x: time in normalized units (cycles)
    - M: MESOR (Midline Estimating Statistic Of Rhythm)
    - A: Amplitude
    - phi: Acrophase (phase shift in radians)
    
    Returns: cosine values
    """
    return M + A * np.cos(2 * np.pi * x + phi)

# Define the cosine function with fixed period for regression
def cosinor_model(t, M, A, phi, period=14.76):
    """
    Cosinor model for regression
    
    Parameters:
    - t: time in days
    - M: MESOR (Midline Estimating Statistic Of Rhythm)
    - A: Amplitude
    - phi: Acrophase (phase shift in radians)
    - period: the period in days (can be fixed or variable)
    
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
    
    # Calculate 95% confidence intervals for the period using bootstrap method
    # (NOTE: this is computationally expensive; for demonstration purposes only)
    # In practice, with known period like tidal cycles, this might be fixed
    
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

# ---------------- Main Processing ----------------
# Load and prepare the data
df = load_and_prepare_data(tide_file)
df['day'] = df['datetime'].dt.date
daily_tidal_range = df.groupby('day')['tidal_range'].agg(lambda x: x.max() - x.min())
daily_tidal_range = daily_tidal_range.reset_index()
daily_tidal_range['days_since_start'] = (pd.to_datetime(daily_tidal_range['day']) - pd.to_datetime(daily_tidal_range['day'].iloc[0])).dt.days
x_days = (df['datetime'] - df['datetime'].iloc[0]).map(lambda x: x.total_seconds()) / (24 * 60 * 60)

# Convert datetime to days since epoch for curve fitting
x = daily_tidal_range['days_since_start'].values
y = daily_tidal_range['tidal_range'].values

# Use known lunar tidal period of approximately 14.76 days
# This corresponds to the fortnightly spring-neap tidal cycle
period = 14.76  # days

# Perform cosinor analysis
results = perform_cosinor_analysis(x, y, period)

# Create predicted values for plotting
t_fine = np.linspace(min(x), max(x), 1000)
y_fine_fit = cosinor_model(t_fine, results['M'], results['A'], results['phi'], period)

# Generate confidence bands for the predicted values
# Calculate prediction intervals (following Cornelissen 2014)
alpha = 0.05  # 95% confidence
t_crit = stats.t.ppf(1 - alpha/2, results['df_residual'])

def calculate_confidence_bands(t, results, alpha=0.05):
    """Calculate confidence bands for the cosinor model"""
    n = len(y)
    
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
        se_fit = np.sqrt(results['MSR'] * (x_i @ XtX_inv @ x_i.T))
        # Standard error of prediction (includes uncertainty of future observations)
        se_pred[i] = np.sqrt(results['MSR'] + se_fit**2)
    
    # Calculate confidence bands
    t_crit = stats.t.ppf(1 - alpha/2, results['df_residual'])
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred
    
    return y_pred, lower, upper

# Calculate confidence bands
y_pred, lower_band, upper_band = calculate_confidence_bands(t_fine, results)

# Convert days to datetime for plotting
t_dates = [pd.to_datetime(daily_tidal_range['day'].iloc[0]) + pd.Timedelta(days=float(t)) for t in t_fine]

# ----------------- Print Results -----------------
# Convert phase from radians to degrees for easier interpretation
phi_degrees = np.degrees(results['phi']) % 360
se_phi_degrees = np.degrees(results['se_phi'])

# Convert CIs from radians to degrees
CI_phi_degrees = (np.degrees(results['CI_phi'][0]) % 360, np.degrees(results['CI_phi'][1]) % 360)

print(f"\n---------- Cosinor Analysis Results (Cornelissen 2014 Method) ----------")
print(f"Period: {period:.2f} days (fixed)")
print(f"MESOR: {results['M']:.4f} ± {results['se_M']:.4f} m")
print(f"Amplitude: {results['A']:.4f} ± {results['se_A']:.4f} m")
print(f"Acrophase: {phi_degrees:.2f}° ± {se_phi_degrees:.2f}°")
print(f"\n95% Confidence Intervals:")
print(f"MESOR: ({results['CI_M'][0]:.4f}, {results['CI_M'][1]:.4f}) m")
print(f"Amplitude: ({results['CI_A'][0]:.4f}, {results['CI_A'][1]:.4f}) m")
print(f"Acrophase: ({CI_phi_degrees[0]:.2f}, {CI_phi_degrees[1]:.2f})°")

print(f"\nModel Statistics:")
print(f"R²: {results['r_squared']:.4f}")
print(f"Adjusted R²: {results['adj_r_squared']:.4f}")
print(f"F({results['df_residual']},{2}): {results['F_statistic']:.4f}")
print(f"p-value: {results['p_value']:.8f}")

if results['p_value'] < 0.05:
    print("\nThe tidal rhythm is statistically significant.")
    print("The null hypothesis of no rhythm is rejected.")
else:
    print("\nThe tidal rhythm is not statistically significant.")
    print("The null hypothesis of no rhythm cannot be rejected.")

# ----------------- Create Plots -----------------
plt.figure(figsize=(15, 10))

# Plot 1: Original data with fitted curve
plt.subplot(211)
plt.plot(df['datetime'], df['tidal_range'], '.', alpha=0.3, markersize=2, label='Original measurements')
plt.xlabel('Date')
plt.ylabel('Tidal Range (m)')
plt.title(f'Tidal Range with Fitted Cosinor Model (Period = {period:.2f} days)')
plt.grid(True, alpha=0.3)

# Plot the original data points
t_days_original = pd.to_datetime(daily_tidal_range['day'])
plt.plot(t_days_original, y, 'bo', label='Daily tidal range', alpha=0.6)

# Plot the fitted curve on original time scale
fitted_original = cosinor_model(x, results['M'], results['A'], results['phi'], period)
plt.plot(t_days_original, fitted_original, 'r-', 
         label=f'Cosinor fit (p = {results["p_value"]:.2e}, R² = {results["r_squared"]:.4f})',
         linewidth=2)
plt.legend()

# Plot 2: Phase-folded data with confidence bands
plt.subplot(212)
# Convert to phase units (0-1)
phase = (x / period) % 1.0
# Sort by phase for clearer visualization
phase_order = np.argsort(phase)
phase_sorted = phase[phase_order]
y_sorted = y[phase_order]

# Plot original phase-folded data
plt.plot(phase_sorted, y_sorted, 'bo', alpha=0.6, label='Observed data')

# Create smooth phase values for the fitted curve
phase_fine = np.linspace(0, 1, 200)
# Calculate model values
y_model = results['M'] + results['A'] * np.cos(2 * np.pi * phase_fine + results['phi'])

# Plot fitted curve
plt.plot(phase_fine, y_model, 'r-', 
         label=f'Cosinor model (Amp = {results["A"]:.4f} m, p = {results["p_value"]:.2e})',
         linewidth=2)

# Calculate confidence bands for the phase-folded data
t_phase = np.linspace(0, period, 200)
_, lower_phase, upper_phase = calculate_confidence_bands(t_phase, results)
phase_fine_plot = (t_phase / period) % 1.0

# Sort by phase
phase_sort_idx = np.argsort(phase_fine_plot)
phase_fine_plot = phase_fine_plot[phase_sort_idx]
lower_phase = lower_phase[phase_sort_idx]
upper_phase = upper_phase[phase_sort_idx]

# Plot confidence bands
plt.fill_between(phase_fine_plot, lower_phase, upper_phase, 
                 color='red', alpha=0.2, label='95% Confidence band')

plt.xlabel('Phase (cycles)')
plt.ylabel('Tidal Range (m)')
plt.title('Phase-folded Tidal Range with 95% Confidence Bands')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('tidal_cosinor_analysis.png', dpi=300)
plt.show()