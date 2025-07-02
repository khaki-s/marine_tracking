import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from datetime import timedelta, time
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import os

#------Load data------
# Load shrimp movement data
shrimp_path = 'D:/khaki/ultralytics-8.3.27/shrimp/distance/run1/2018-2019.csv'
tide_path = 'D:/khaki/ultralytics-8.3.27/shrimp/tide/2018-2019tide.csv'
shrimp = pd.read_csv(shrimp_path, parse_dates=['time'])
shrimp.rename(columns={
    'time': 'datetime',
    'distance_permin_pernumber': 'distance_mm'
}, inplace=True)
#Keep only useful columns and remove null values
shrimp = shrimp[['datetime', 'distance_mm']].dropna()

# Load tide data
tide = pd.read_csv(tide_path)
#Merge the day and time columns and resolve to datetime
tide['datetime'] = pd.to_datetime(
    tide['day'] + ' ' + tide['time'],
    format='%d-%b-%y %H:%M:%S'
)
tide.rename(columns={'tidal_range': 'height_mm'}, inplace=True)
tide = tide[['datetime', 'height_mm']].dropna()
tide.sort_values('datetime', inplace=True)

# Create date columns
tide['date'] = tide['datetime'].dt.date

# Compute daily tide amplitude (max - min)
tide_amp = (
    tide
    .groupby('date', as_index=False)['height_mm']
    .agg(['min','max'])
    .reset_index()
)
tide_amp['tide_amp'] = tide_amp['max'] - tide_amp['min']
tide_amp = tide_amp[['date', 'tide_amp']]

# Extract daily tide level at 23:59:59
tide_level = tide[tide['datetime'].dt.time == time(23, 59, 59)]
tide_level = tide_level[['date', 'height_mm']].rename(columns={'height_mm': 'tide_level'})

# Align shrimp observations to tide dates
def get_tide_date(dt):
    # if shrimp time is between 00:00 and 00:59, use previous day's tide
    if dt.time() < time(1,0):
        return (dt.date() - timedelta(days=1))
    else:
        return dt.date()
shrimp['datetime'] = pd.to_datetime(shrimp['datetime'], infer_datetime_format=True)
shrimp['tide_date'] = shrimp['datetime'].apply(get_tide_date)

# Merge into one DataFrame, keep date for plotting
daily = (
    shrimp[['tide_date', 'distance_mm']]
    .merge(tide_amp,   left_on='tide_date', right_on='date')
    .merge(tide_level, left_on='tide_date', right_on='date')
)
daily.rename(columns={'tide_date': 'date'}, inplace=True)

# Aggregate duplicates
daily = daily.groupby('date', as_index=False).agg({
    'distance_mm': 'mean',
    'tide_amp': 'first',
    'tide_level': 'first'
})
daily['date'] = pd.to_datetime(daily['date'])

#------Test normality and homogeneity of variance------
# 1. Extract year and month
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month
daily['year_month'] = daily['date'].dt.strftime('%Y-%m')  # Create year-month combination as unique identifier

# 2. Get all year-month combinations in chronological order
year_months = sorted(daily['year_month'].unique())
print(f"The year-month combinations included are: {year_months}")

# 3. To perform multiple comparisons, we need to ensure there are enough data points
month_counts = daily.groupby('year_month')['distance_mm'].count()
print("The number of samples per month:")
print(month_counts)

# 4. Check if the data is normally distributed
print("\nNormality test:")
for ym in year_months:
    data = daily.loc[daily['year_month']==ym, 'distance_mm']
    if len(data) >= 8:  # Only when the sample size is large enough can the normality test be performed
        stat, p = stats.shapiro(data)
        print(f"{ym}: Shapiro-Wilk W={stat:.3f}, p={p:.4f} {'(non-normal)' if p<0.05 else '(normal)'}")

# 5. Perform one-way ANOVA or non-parametric test, depending on the data characteristics
# First prepare data
data_list = []
labels = []
for ym in year_months:
    group_data = daily.loc[daily['year_month']==ym, 'distance_mm'].values
    if len(group_data) > 0:  # Ensure
        data_list.append(group_data)
        labels.append(ym)

# Test for homogeneity of variance
all_data = np.concatenate([d for d in data_list if len(d) > 0])
groups = []
for i, d in enumerate(data_list):
    if len(d) > 0:
        groups.extend([i] * len(d))
stat, p = stats.levene(*data_list)
print(f"\nHomogeneity of variance test: Levene's W={stat:.3f}, p={p:.4f} {'(heterogeneity)' if p<0.05 else '(homogeneity)'}")

# According to the results, select the appropriate method
# Here we provide two methods: one suitable for parametric assumptions, one suitable for non-parametric assumptions

# 5a. Parametric method: ANOVA + Tukey HSD
print("\nParametric method: One-way ANOVA")
f_stat, p_anova = stats.f_oneway(*data_list)
print(f"ANOVA: F={f_stat:.3f}, p={p:.4f} {'(at least significant difference)' if p<0.05 else '(no significant difference)'}")

if p_anova < 0.05:
    # Prepare data for tukeyhsd
    all_data = []
    group_labels = []
    for i, (group, label) in enumerate(zip(data_list, labels)):
        all_data.extend(group)
        group_labels.extend([label] * len(group))
    
    # Tukey HSD multiple comparisons
    tukey = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
    print("\nTukey HSD multiple comparisons:")
    print(tukey)

# 5b. Non-parametric method: Kruskal-Wallis + Dunn's test
print("\nNon-parametric method: Kruskal-Wallis test")
h_stat, p_kw = stats.kruskal(*data_list)
print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f} {'(at least significant difference)' if p_kw<0.05 else '(no significant difference)'}")

if p_kw < 0.05:
    # Import Dunn's test
    try:
        from scikit_posthocs import posthoc_dunn
        # Prepare data format
        df_for_dunn = pd.DataFrame({
            'value': np.concatenate(data_list),
            'group': np.concatenate([[i]*len(g) for i, g in enumerate(data_list)])
        })
        # Execute Dunn's test
        dunn_result = posthoc_dunn(df_for_dunn, val_col='value', group_col='group', p_adjust='fdr_bh')
        print("\nDunn's test multiple comparisons (adjusted p-values):")
        # Convert numeric indices back to year-month labels
        dunn_result.index = labels
        dunn_result.columns = labels
        print(dunn_result)
    except ImportError:
        print("Note: scikit-posthocs package is required to execute Dunn's test")
        print("Use the command: pip install scikit-posthocs")
        
        # If scikit-posthocs is not installed, use statsmodels' pairwise_mannwhitneyu as an alternative
        from statsmodels.stats.multitest import multipletests
        print("\nAlternative method: Pairwise Mann-Whitney U test + FDR correction")
        
        # Calculate all possible comparisons
        pairs = [(i, j) for i in range(len(labels)) for j in range(i+1, len(labels))]
        pvals = []
        
        for i, j in pairs:
            stat, p = stats.mannwhitneyu(data_list[i], data_list[j], alternative='two-sided')
            pvals.append(p)
        
        # Apply FDR correction
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        
        # Output results
        print("Significant pairs (p < 0.05, FDR corrected):")
        sig_pairs = []
        for k, ((i, j), p, rej) in enumerate(zip(pairs, pvals_corrected, reject)):
            if rej:
                print(f"{labels[i]} vs {labels[j]}: p_adj = {p:.4f}")
                sig_pairs.append(((labels[i], labels[j]), p))

#------Plot: in chronological order------
# 1. Prepare data, sorted by year-month
monthly_data = []
for ym in year_months:
    group_data = daily.loc[daily['year_month']==ym, 'distance_mm'].values
    if len(group_data) > 0:
        monthly_data.append((ym, group_data))

# 2. Create chart
fig, ax = plt.subplots(figsize=(12, 6))

# 3. Violin plot + scatter plot + mean line
positions = np.arange(len(monthly_data))
violin_data = [data for _, data in monthly_data]

# 3.1 Violin plot
vp = ax.violinplot(
    violin_data,
    positions=positions,
    showmeans=False,
    showmedians=True
)
for body in vp['bodies']:
    body.set_facecolor('lightblue')
vp['cmedians'].set_color('darkred')

# 3.2 Scatter plot (with jitter)
rng = np.random.RandomState(42)
jitter = 0.1
for i, (_, vals) in enumerate(monthly_data):
    x = rng.normal(positions[i], jitter, size=len(vals))
    ax.scatter(x, vals, color='lightblue', s=10, alpha=0.5)

# 3.3 Mean line
means = [vals.mean() for _, vals in monthly_data]
ax.plot(positions, means, '-o', color='#1f76b5ff', linewidth=2, label='Monthly Mean')

# 4. Mark significant differences (if any)
if 'sig_pairs' in locals():
    # Find the maximum value of the y-axis
    y_max = max([arr.max() for _, arr in monthly_data]) * 1.05
    h = y_max * 0.05
    
    for (label1, label2), pval in sig_pairs:
        idx1 = [i for i, (ym, _) in enumerate(monthly_data) if ym == label1][0]
        idx2 = [i for i, (ym, _) in enumerate(monthly_data) if ym == label2][0]
        x1, x2 = positions[idx1], positions[idx2]
        y = y_max + h * (1 + list(sig_pairs).index(((label1, label2), pval)) % 5)
        
        # Draw
        if pval < 0.001:
            stars = '***'
        elif pval < 0.01:
            stars = '**'
        elif pval < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        ax.text((x1+x2)/2, y, stars, ha='center', va='bottom')

# 5.Set axis labels and legend
ax.set_xticks(positions)
ax.set_xticklabels([ym for ym, _ in monthly_data], rotation=45)
ax.set_xlabel('Year-Month')
ax.set_ylabel('Shrimp Distance (mm)')
ax.set_title('Monthly Shrimp Movement')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#------CCF------
# 1. Compute cross-correlation for lags -3 to +3 days
max_lag = 3
lags = np.arange(-max_lag, max_lag + 1)
ccf_amp = [daily['distance_mm'].corr(daily['tide_amp'].shift(l)) for l in lags]
ccf_lvl = [daily['distance_mm'].corr(daily['tide_level'].shift(l)) for l in lags]
best_amp_idx = np.nanargmax(np.abs(ccf_amp))
best_lvl_idx = np.nanargmax(np.abs(ccf_lvl))
best_amp_lag, best_amp_corr = lags[best_amp_idx], ccf_amp[best_amp_idx]
best_lvl_lag, best_lvl_corr = lags[best_lvl_idx], ccf_lvl[best_lvl_idx]

print(f"Best CCF (amp): r = {best_amp_corr:.2f} at lag = {best_amp_lag} days")
print(f"Best CCF (level): r = {best_lvl_corr:.2f} at lag = {best_lvl_lag} days")

# 2. Smooth shrimp distance with cubic interpolation
dates_num = mdates.date2num(daily['date'])
interp_func = interp1d(dates_num, daily['distance_mm'], kind='cubic')
num_smooth = np.linspace(dates_num.min(), dates_num.max(), 300)
dist_smooth = interp_func(num_smooth)
dates_smooth = mdates.num2date(num_smooth)

# 3. Plotting combined figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Upper panel: tide_amp and tide_level
ax1.plot(daily['date'], daily['tide_amp'], label='Tide Amplitude')
ax1.plot(daily['date'], daily['tide_level'], label='Tide Level')
ax1.set_ylabel('Tide (mm)')
ax1.legend()
ax1.set_title('Daily Tide Metrics')
ax1.text(0.02, 0.85,
        f"Tide Amp CCF: r={best_amp_corr:.2f}, lag {best_amp_lag}d\n"
        f"Tide Level CCF: r={best_lvl_corr:.2f}, lag {best_lvl_lag}d",
        transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

# Lower panel: shrimp distance (scatter + smooth curve)
ax2.scatter(daily['date'], daily['distance_mm'], label='Shrimp Distance', s=20, alpha=0.7)
ax2.plot(dates_smooth, dist_smooth, label='Smoothed Distance', linestyle='--')
ax2.set_ylabel('Distance (mm)')
ax2.set_title('Shrimp Movement')

plt.tight_layout()
plt.show()