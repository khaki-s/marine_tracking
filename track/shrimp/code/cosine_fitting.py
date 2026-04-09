import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from datetime import datetime, time
import matplotlib.ticker as ticker
import statsmodels.api as sm

# ---------------- Function Definitions ----------------

def load_and_prepare_data(file_content):
    df = file_content[file_content['distance_permin_pernumber'] > 0].reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    first_day = df['time'].min()
    df['days'] = [(t - first_day).total_seconds() / (24 * 3600) for t in df['time']]
    return df

def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def apply_moving_average(df, column, window_size=3):
    df_copy = df.copy()
    df_copy.loc[:, f'{column}_smoothed'] = df_copy[column].rolling(window=window_size, center=True).mean()
    df_copy.loc[:, f'{column}_smoothed'] = df_copy[f'{column}_smoothed'].fillna(df_copy[column])
    return df_copy

def cosinor_model(t, M, A, phi, period):
    x = t / period
    return M + A * np.cos(2 * np.pi * x + phi)

def cosinor_to_beta(A, phi):
    beta1 = A * np.cos(phi)
    beta2 = -A * np.sin(phi)
    return beta1, beta2

def beta_to_cosinor(beta1, beta2):
    A = np.sqrt(beta1**2 + beta2**2)
    phi = np.arctan2(-beta2, beta1)
    return A, phi

def perform_cosinor_analysis(t, y, period):
    n = len(y)
    x = t / period
    X = np.column_stack((
        np.ones(n),
        np.cos(2 * np.pi * x),
        np.sin(2 * np.pi * x)
    ))
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    M = beta_hat[0]
    beta1 = beta_hat[1]
    beta2 = beta_hat[2]
    A, phi = beta_to_cosinor(beta1, beta2)
    y_fit = X @ beta_hat
    residuals = y - y_fit
    RSS = np.sum(residuals**2)
    df_total = n - 1
    df_model = 2
    df_residual = n - 3
    MSR = RSS / df_residual
    XtX_inv = np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(MSR * XtX_inv))
    se_M = se_beta[0]
    se_beta1 = se_beta[1]
    se_beta2 = se_beta[2]
    se_A = np.sqrt((beta1**2 * se_beta2**2 + beta2**2 * se_beta1**2) / (beta1**2 + beta2**2))
    se_phi = np.sqrt((beta1**2 * se_beta2**2 + beta2**2 * se_beta1**2) / (beta1**2 + beta2**2)**2)
    TSS = np.sum((y - np.mean(y))**2)
    r_squared = 1 - RSS / TSS
    adj_r_squared = 1 - (RSS / df_residual) / (TSS / df_total)
    MSM = (TSS - RSS) / df_model
    F_statistic = MSM / MSR
    p_value = 1 - stats.f.cdf(F_statistic, df_model, df_residual)
    t_crit = stats.t.ppf(0.975, df_residual)
    CI_M = (M - t_crit * se_M, M + t_crit * se_M)
    CI_A = (A - t_crit * se_A, A + t_crit * se_A)
    CI_phi = (phi - t_crit * se_phi, phi + t_crit * se_phi)
    results = {
        'M': M, 'A': A, 'phi': phi, 'beta1': beta1, 'beta2': beta2,
        'period': period, 'se_M': se_M, 'se_A': se_A, 'se_phi': se_phi,
        'CI_M': CI_M, 'CI_A': CI_A, 'CI_phi': CI_phi,
        'r_squared': r_squared, 'adj_r_squared': adj_r_squared,
        'F_statistic': F_statistic, 'p_value': p_value,
        'RSS': RSS, 'df_residual': df_residual, 'MSR': MSR
    }
    return results

def calculate_confidence_bands(t, results, alpha=0.05):
    x_cycles = t / results['period']
    X_new = np.column_stack((
        np.ones(len(t)),
        np.cos(2 * np.pi * x_cycles),
        np.sin(2 * np.pi * x_cycles)
    ))
    x_original = x / results['period']
    X = np.column_stack((
        np.ones(len(x)),
        np.cos(2 * np.pi * x_original),
        np.sin(2 * np.pi * x_original)
    ))
    XtX_inv = np.linalg.inv(X.T @ X)
    y_pred = X_new @ np.array([results['M'], results['beta1'], results['beta2']])
    se_pred = np.zeros(len(t))
    for i in range(len(t)):
        x_i = X_new[i:i+1, :]
        se_fit = np.sqrt(results['MSR'] * (x_i @ XtX_inv @ x_i.T).item())
        se_pred[i] = np.sqrt(results['MSR'] + se_fit**2)
    t_crit = stats.t.ppf(1 - alpha/2, results['df_residual'])
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred
    return y_pred, lower, upper

def process_tide_data(tide_df, window_size=3):
    tide_df['datetime'] = pd.to_datetime(tide_df['day'] + ' ' + tide_df['time'])
    tide_df['date'] = tide_df['datetime'].dt.normalize()
    daily_tidal_range_raw = tide_df.groupby('date')['tidal_range'].agg(lambda x: x.max() - x.min()).reset_index()
    daily_tidal_range_raw.rename(columns={'tidal_range': 'daily_range'}, inplace=True)
    tide_level_235959_raw = tide_df[tide_df['datetime'].dt.time == time(23, 59, 59)].copy()
    tide_level_235959_raw = tide_level_235959_raw.rename(columns={'tidal_range': 'level_235959'})[['date', 'level_235959']]
    min_date = min(daily_tidal_range_raw['date'].min(), tide_level_235959_raw['date'].min())
    max_date = max(daily_tidal_range_raw['date'].max(), tide_level_235959_raw['date'].max())
    all_dates = pd.DataFrame({'date': pd.date_range(min_date, max_date, freq='D')})
    daily_tidal_range_processed = pd.merge(all_dates, daily_tidal_range_raw, on='date', how='left')
    daily_tidal_range_processed['daily_range_interp'] = daily_tidal_range_processed['daily_range'].interpolate(method='linear')
    daily_tidal_range_processed['daily_range_interp'] = daily_tidal_range_processed['daily_range_interp'].fillna(method='bfill').fillna(method='ffill')
    daily_tidal_range_processed['daily_range_smoothed'] = daily_tidal_range_processed['daily_range_interp'].rolling(window=window_size, center=True, min_periods=1).mean()
    tide_level_235959_processed = pd.merge(all_dates, tide_level_235959_raw, on='date', how='left')
    tide_level_235959_processed['level_235959_interp'] = tide_level_235959_processed['level_235959'].interpolate(method='linear')
    tide_level_235959_processed['level_235959_interp'] = tide_level_235959_processed['level_235959'].fillna(method='bfill').fillna(method='ffill')
    tide_level_235959_processed['level_235959_smoothed'] = tide_level_235959_processed['level_235959_interp'].rolling(window=window_size, center=True, min_periods=1).mean()
    return daily_tidal_range_processed, tide_level_235959_processed

# ---------------- Configuration ----------------
data_file = "D:/khaki/ultralytics-8.3.27/shrimp/distance/run4/2016-2017-out.csv"
tide_file = r"D:\khaki\ultralytics-8.3.27\shrimp\tide\2016-2017tide.csv"
output_figure = "shrimp_cosinor_analysis.pdf"
column_name = "distance_permin_pernumber"
period_guess = 14.76
window_size = 6
plt.rcParams["font.size"] = 16
plt.rcParams['font.family'] = 'Arial'

# ---------------- Main Processing ----------------
print("Loading data...")
file_content = pd.read_csv(data_file)
df_original = load_and_prepare_data(file_content)

# Load tide data
print("Loading tide data...")
tide_data = pd.read_csv(tide_file)
tide_data['datetime'] = pd.to_datetime(tide_data['day'] + ' ' + tide_data['time'])
tide_data['days'] = (tide_data['datetime'] - df_original['time'].min()).dt.total_seconds() / (24 * 3600)

# Process tide data
print("\nProcessing tide data...")
daily_tidal_range_processed, tide_level_235959_processed = process_tide_data(tide_data.copy(), window_size)

print("\n1. ANALYZING WITH OUTLIER REMOVAL (IQR)")
df_no_outliers = remove_outliers_iqr(df_original, column_name)
print(f"Removed {len(df_original) - len(df_no_outliers)} outliers, {len(df_no_outliers)} data points remain")

print("\n2. APPLYING MOVING AVERAGE")
df_smoothed = apply_moving_average(df_no_outliers, column_name, window_size)
smooth_col = f"{column_name}_smoothed"

x = df_smoothed['days'].values
y = df_smoothed[smooth_col].values

# Detrend using LOWESS
lowess = sm.nonparametric.lowess(y, x, frac=0.15)
trend = lowess[:, 1]
y_detr = y - trend
df_smoothed['detrended'] = y_detr
df_smoothed['trend'] = trend

y = y_detr

print("\n3. PERFORMING COSINOR ANALYSIS")
results = perform_cosinor_analysis(x, y, period_guess)

# Match tide data to shrimp observations
print("\nMatching tide data to shrimp observations...")
def get_tide_lookup_date(shrimp_datetime):
    if shrimp_datetime.time() > time(0, 0, 0):
        return (shrimp_datetime - pd.Timedelta(days=1)).normalize()
    else:
        return shrimp_datetime.normalize()

df_smoothed['tide_lookup_date'] = df_smoothed['time'].apply(get_tide_lookup_date)

df_smoothed = pd.merge(df_smoothed, daily_tidal_range_processed[['date', 'daily_range_smoothed']],
                       left_on='tide_lookup_date', right_on='date', how='left', suffixes=('', '_matched_range'))
df_smoothed.rename(columns={'daily_range_smoothed': 'matched_tidal_range'}, inplace=True)
df_smoothed.drop(columns=['date_matched_range'], inplace=True, errors='ignore')

df_smoothed = pd.merge(df_smoothed, tide_level_235959_processed[['date', 'level_235959_smoothed']],
                       left_on='tide_lookup_date', right_on='date', how='left', suffixes=('', '_matched_level'))
df_smoothed.rename(columns={'level_235959_smoothed': 'matched_tide_level_235959'}, inplace=True)
df_smoothed.drop(columns=['date_matched_level'], inplace=True, errors='ignore')

# ----------------- Print Results -----------------
phi_degrees = np.degrees(results['phi']) % 360
se_phi_degrees = np.degrees(results['se_phi'])
CI_phi_degrees = (np.degrees(results['CI_phi'][0]) % 360, np.degrees(results['CI_phi'][1]) % 360)

print(f"\n---------- Cosinor Analysis Results ----------")
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
days_smooth = np.linspace(df_smoothed['days'].min(), df_smoothed['days'].max(), 1000)
fitted_curve = cosinor_model(days_smooth, results['M'], results['A'], results['phi'], period_guess)
_, lower_band, upper_band = calculate_confidence_bands(days_smooth, results)

first_day = df_smoothed['time'].min()
dates_smooth = [first_day + pd.Timedelta(days=d) for d in days_smooth]

# --- First Figure: Phase-folded and Detrended Movement Data ---
plt.figure(figsize=(15, 10))

# Plot 1 (First Figure): Phase-folded data with confidence bands
plt.subplot(211)
phase = (df_smoothed['days'] / period_guess) % 1.0
phase_order = np.argsort(phase)
phase_sorted = phase.iloc[phase_order]
y_sorted = df_smoothed['detrended'].iloc[phase_order]

plt.plot(phase_sorted, y_sorted, 'o', alpha=0.6, label='Observed data')

phase_fine = np.linspace(0, 1, 200)
y_model = results['M'] + results['A'] * np.cos(2 * np.pi * phase_fine + results['phi'])
plt.plot(phase_fine, y_model, 'r-',
         label=f'Cosinor model (Amp = {results["A"]:.4f}, p = {results["p_value"]:.2e})',
         linewidth=2)

t_phase = np.linspace(0, period_guess, 200)
_, lower_phase, upper_phase = calculate_confidence_bands(t_phase, results)
phase_fine_plot = (t_phase / period_guess) % 1.0
phase_sort_idx = np.argsort(phase_fine_plot)
phase_fine_plot = phase_fine_plot[phase_sort_idx]
lower_phase = lower_phase[phase_sort_idx]
upper_phase = upper_phase[phase_sort_idx]

plt.fill_between(phase_fine_plot, lower_phase, upper_phase,
                 color='#FF0000', alpha=0.2, label='95% Confidence band')
plt.xlabel('Phase (cycles)')
plt.ylabel("distance permin pernumber")
plt.title('Phase-folded Movement Data with 95% Confidence Bands')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2 (First Figure): Detrended data with Cosinor fit
ax_fig1_sub2 = plt.subplot(212)
plt.scatter(df_smoothed['time'], df_smoothed['detrended'], alpha=0.6, label='Detrended Data Points')
plt.plot(dates_smooth, fitted_curve, 'r-',
         label=f'Cosinor Fit: Period = {results["period"]:.2f} days, R² = {results["r_squared"]:.2f}')
plt.fill_between(dates_smooth, lower_band, upper_band, color='#FF0000', alpha=0.2, label='95% Confidence Band')

plt.title(f'Cosinor Analysis for Shrimp Movement (Moving Avg Window={window_size})')
plt.xlabel('Date')
plt.ylabel("distance permin pernumber")
plt.legend()

min_date_fig1 = df_smoothed['time'].min()
max_date_fig1 = df_smoothed['time'].max()
ax_fig1_sub2.set_xlim([min_date_fig1, max_date_fig1])
ax_fig1_sub2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
monthly_ticks_fig1 = mdates.MonthLocator().tick_values(min_date_fig1, max_date_fig1)
all_ticks_fig1 = sorted(list(set(
    monthly_ticks_fig1.tolist() +
    [mdates.date2num(min_date_fig1), mdates.date2num(max_date_fig1)]
)))
ax_fig1_sub2.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_fig1))
ax_fig1_sub2.tick_params(axis='x', rotation=45)
ax_fig1_sub2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Second Figure: Cosinor Fit and Tide Data ---
plt.figure(figsize=(15, 10))

fixed_start_date = df_smoothed['time'].min()
fixed_end_date = df_smoothed['time'].max()

# Plot 1 (Second Figure): Detrended data with Cosinor fit
ax1 = plt.subplot(211)
plt.scatter(df_smoothed['time'], df_smoothed['detrended'], alpha=0.6, label='Detrended Data Points')
plt.plot(dates_smooth, fitted_curve, 'r-',
         label=f'Cosinor Fit: Period = {results["period"]:.2f} days, R² = {results["r_squared"]:.2f}')
plt.fill_between(dates_smooth, lower_band, upper_band, color='#FF0000', alpha=0.2, label='95% Confidence Band')

plt.title('Shrimp Movement Data with Cosine Fit')
plt.xlabel('Date')
plt.ylabel("distance permin pernumber")
plt.legend()

ax1.set_xlim([fixed_start_date, fixed_end_date])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
monthly_ticks_ax1 = mdates.MonthLocator().tick_values(fixed_start_date, fixed_end_date)
all_ticks_ax1 = sorted(list(set(
    monthly_ticks_ax1.tolist() +
    [mdates.date2num(fixed_start_date), mdates.date2num(fixed_end_date)]
)))
ax1.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_ax1))
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Plot 2 (Second Figure): Tide data
ax2 = plt.subplot(212)
plt.plot(df_smoothed['time'], df_smoothed['matched_tidal_range'], 'o',
         color="lightblue", label='Matched Daily Tidal Range', alpha=0.8)
plt.plot(df_smoothed['time'], df_smoothed['matched_tide_level_235959'], 'o',
         color="#f2a9a2ff", label='Matched 23:59:59 Tide Level', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Tidal Range (m) / Tide Level (m)')
plt.title(f'Tide Data with Matched Shrimp Movement Time (Window={window_size})')

ax2.set_xlim([fixed_start_date, fixed_end_date])
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
monthly_ticks_ax2 = mdates.MonthLocator().tick_values(fixed_start_date, fixed_end_date)
all_ticks_ax2 = sorted(list(set(
    monthly_ticks_ax2.tolist() +
    [mdates.date2num(fixed_start_date), mdates.date2num(fixed_end_date)]
)))
ax2.xaxis.set_major_locator(ticker.FixedLocator(all_ticks_ax2))
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.show()