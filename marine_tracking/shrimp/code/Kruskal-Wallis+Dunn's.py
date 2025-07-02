import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, time
import scipy.stats as stats
from scikit_posthocs import posthoc_dunn

import string

#------Load data------
# Load shrimp movement data
shrimp_path = 'D:/khaki/ultralytics-8.3.27/shrimp/distance/run1/2018-2019.csv'
tide_path = 'D:/khaki/ultralytics-8.3.27/shrimp/tide/2018-2019tide.csv'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 18
# Load and preprocess shrimp data
shrimp = pd.read_csv(shrimp_path, parse_dates=['time'])
shrimp.rename(columns={
    'time': 'datetime',
    'distance_permin_pernumber': 'distance_mm'
}, inplace=True)
# Keep only useful columns and remove null values
shrimp = shrimp[['datetime', 'distance_mm']].dropna()

# Load and preprocess tide data
tide = pd.read_csv(tide_path)
# Merge the day and time columns and convert to datetime
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
    # If shrimp time is between 00:00 and 00:59, use previous day's tide
    if dt.time() < time(1,0):
        return (dt.date() - timedelta(days=1))
    else:
        return dt.date()

shrimp['datetime'] = pd.to_datetime(shrimp['datetime'])
shrimp['tide_date'] = shrimp['datetime'].apply(get_tide_date)

# Merge into one DataFrame, keep date for plotting
daily = (
    shrimp[['tide_date', 'distance_mm']]
    .merge(tide_amp, left_on='tide_date', right_on='date', how='left')
    .merge(tide_level, left_on='tide_date', right_on='date', how='left')
)
daily.rename(columns={'tide_date': 'date'}, inplace=True)

# Aggregate by date but maintain all data points
daily_mean = daily.groupby('date', as_index=False).agg({
    'distance_mm': 'mean',
    'tide_amp': 'first',
    'tide_level': 'first'
})
daily_mean['date'] = pd.to_datetime(daily_mean['date'])

# Extract year and month for both dataframes
daily['year'] = pd.to_datetime(daily['date']).dt.year
daily['month'] = pd.to_datetime(daily['date']).dt.month
daily['year_month'] = pd.to_datetime(daily['date']).dt.strftime('%Y-%m')

daily_mean['year'] = daily_mean['date'].dt.year
daily_mean['month'] = daily_mean['date'].dt.month
daily_mean['year_month'] = daily_mean['date'].dt.strftime('%Y-%m')

#------Statistical Analysis: Kruskal-Wallis + Dunn's Multiple Comparison------
# Retrieve all year-month combinations in chronological order
year_months = sorted(daily['year_month'].unique())
print(f"Year-months included in the data: {year_months}")

# Check the number of samples per month
month_counts = daily.groupby('year_month')['distance_mm'].count()
print("Number of samples per month:")
print(month_counts)

# Prepare data for statistical analysis
data_list = []
labels = []
for ym in year_months:
    # Extract corresponding distance values for each month
    group_data = daily.loc[daily['year_month']==ym, 'distance_mm'].values
    if len(group_data) > 0:  # Ensure there is data available
        data_list.append(group_data)
        labels.append(ym)

# Perform Kruskal-Wallis test
print("\nNonparametric method: Kruskal-Wallis test")
h_stat, p_kw = stats.kruskal(*data_list)  # Perform overall test on grouped data
print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f} {'(Significant differences exist)' if p_kw<0.05 else '(No significant differences)'}")

# If there is a significant difference, perform Dunn's test and letter marking
if p_kw < 0.05:
    # Prepare data format for Dunn's test
    all_values = np.concatenate(data_list)
    all_groups = np.concatenate([[i]*len(g) for i, g in enumerate(data_list)])
    
    df_for_dunn = pd.DataFrame({
        'value': all_values,
        'group': all_groups
    })
    
    # Perform Dunn's test
    dunn_result = posthoc_dunn(df_for_dunn, val_col='value', group_col='group', p_adjust='fdr_bh')
    
    # Convert numeric index back to year-month labels
    dunn_result.index = labels
    dunn_result.columns = labels
    print("\nDunn's test multiple comparisons:")
    print(dunn_result)
    
    # Check if any significant differences exist in pairwise comparisons
    significant_differences_exist = False
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if dunn_result.iloc[i, j] < 0.05:
                significant_differences_exist = True
                break
        if significant_differences_exist:
            break
            
    if not significant_differences_exist:
        print("\nWarning: Although Kruskal-Wallis test showed significant differences,")
        print("no pairwise significant differences were detected by Dunn's test.")
        print("This can happen due to the more conservative nature of pairwise comparisons.")
    
    # Improved letter assignment function
    def assign_letters(p_values_matrix, alpha=0.05):
        n = len(p_values_matrix)
        groups = list(p_values_matrix.index)
        
        # Initialize letter assignments
        letter_assignments = {group: [] for group in groups}
        available_letters = list(string.ascii_lowercase)
        current_letter_idx = 0
        
        # Create an adjacency matrix for groups (1 if not significantly different)
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j or p_values_matrix.iloc[i, j] > alpha:
                    adjacency[i, j] = 1
        
        # Sort groups by their mean values (if means are available)
        group_means = {}
        for group in groups:
            group_data = daily[daily['year_month'] == group]['distance_mm']
            if not group_data.empty:
                group_means[group] = group_data.mean()
        
        # Sort groups by mean if means are available
        if group_means:
            sorted_groups = sorted(groups, key=lambda g: group_means.get(g, 0), reverse=True)
            # Reorder adjacency matrix
            sorted_indices = [groups.index(g) for g in sorted_groups]
            adjacency = adjacency[sorted_indices, :][:, sorted_indices]
            groups = sorted_groups
        
        # Find connected components (groups that are not significantly different)
        components = []
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in range(n):
                if adjacency[node, neighbor] == 1 and neighbor not in visited:
                    dfs(neighbor, component)
        
        for i in range(n):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)
        
        # Assign letters to components
        for component in components:
            letter = available_letters[current_letter_idx]
            current_letter_idx = (current_letter_idx + 1) % len(available_letters)
            
            for node in component:
                letter_assignments[groups[node]].append(letter)
        
        # Format the results
        result = {}
        for group in p_values_matrix.index:
            result[group] = ''.join(sorted(set(letter_assignments[group])))
            
        return result
    
    # Assign letters
    letter_groups = assign_letters(dunn_result)
    
    # Print results
    print("\nLetter labeling based on Dunn's test (same letters indicate no significant difference):")
    for label, letters in letter_groups.items():
        print(f"{label}: {letters}")
        
# ------Plotting: Monthly Shrimp Movement with Statistical Grouping------
# Prepare data, sorted by year-month
monthly_data = []
for ym in year_months:
    group_data = daily.loc[daily['year_month']==ym, 'distance_mm'].values
    if len(group_data) > 0:
        monthly_data.append((ym, group_data))

# Create chart
fig, ax = plt.subplots(figsize=(16, 9))

# Violin chart + scatter plot + mean line
positions = np.arange(len(monthly_data))
violin_data = [data for _, data in monthly_data]

# Draw a violin plot after taking the 5% to 95% percentile of each set of data to avoid the influence of extreme values
violin_data_trimmed = []
for vals in violin_data:
    lower = np.percentile(vals, 5)
    upper = np.percentile(vals, 95)
    trimmed = vals[(vals >= lower) & (vals <= upper)]
    violin_data_trimmed.append(trimmed)

# Violin plot
vp = ax.violinplot(
    violin_data_trimmed,
    positions=positions,
    showmedians=True,
)

#Reduce the color of the violin image
for body in vp['bodies']:
    body.set_facecolor('lightblue')
    body.set_alpha(0.7)
vp['cmedians'].set_color('darkred')

# Scatter plot with jitter
rng = np.random.RandomState(42)
jitter = 0.1
for i, (_, vals) in enumerate(monthly_data):
    x = rng.normal(positions[i], jitter, size=len(vals))
    ax.scatter(x, vals, color='#4c93c2ff', s=10, alpha=0.5)

# Mean line
means = [vals.mean() for _, vals in monthly_data]
ax.plot(positions, means, '-o', color='lightblue', linewidth=2, label='Monthly Mean', alpha=0.5)

# Annotate letter grouping
if 'letter_groups' in locals():
    # Find the maximum value for y-axis scaling
    y_max = max([np.max(vals) for _, vals in monthly_data]) * 1.02
    
    # Find all unique letter combinations
    unique_letters = set(letter_groups.values())
    
    # Check if all groups have the same letter (no significant differences found)
    all_same_letter = len(unique_letters) == 1
    
    # Only add letter annotations if there are actual differences
    if not all_same_letter:
        # Create color map for each unique letter combination
        import matplotlib.cm as cm
        
        # Color mapping
        color_map = cm.get_cmap('tab10', len(unique_letters))
        
        # Create mapping from letter combinations to colors
        letter_to_color = {}
        for i, letter in enumerate(unique_letters):
            letter_to_color[letter] = color_map(i)
        
        # Mark letters above each month with corresponding colors
        for i, (ym, _) in enumerate(monthly_data):
            if ym in letter_groups:
                letter = letter_groups[ym]
                color = letter_to_color[letter]
                ax.text(positions[i], y_max, letter, 
                    ha='center', va='bottom',  fontweight='bold',
                    color=color,fontsize = 25)
        
        # Set y-axis limits to accommodate labels
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(current_ymin, y_max * 1.05)
        
    else:
        print("All groups have the same letter. No significant differences found in pairwise comparisons.")

# Label axes and add title/legend
ax.set_xticks(positions)
ax.set_xticklabels([ym for ym, _ in monthly_data])
ax.set_xlabel('Year-Month')
ax.set_ylabel('Shrimp Distance (mm)')
ax.set_title('Monthly Shrimp Movement with Statistical Grouping')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
#plt.savefig('monthly_shrimp_movement.png', dpi=300)
plt.show()




# #------Cross-Correlation Function (CCF) Analysis------
# # Compute cross-correlation for lags -3 to +3 days
# max_lag = 3
# lags = np.arange(-max_lag, max_lag + 1)
# ccf_amp = [daily_mean['distance_mm'].corr(daily_mean['tide_amp'].shift(l)) for l in lags]
# ccf_lvl = [daily_mean['distance_mm'].corr(daily_mean['tide_level'].shift(l)) for l in lags]

# # Find best correlations
# best_amp_idx = np.nanargmax(np.abs(ccf_amp))
# best_lvl_idx = np.nanargmax(np.abs(ccf_lvl))
# best_amp_lag, best_amp_corr = lags[best_amp_idx], ccf_amp[best_amp_idx]
# best_lvl_lag, best_lvl_corr = lags[best_lvl_idx], ccf_lvl[best_lvl_idx]

# print(f"Best CCF (amplitude): r = {best_amp_corr:.2f} at lag = {best_amp_lag} days")
# print(f"Best CCF (level): r = {best_lvl_corr:.2f} at lag = {best_lvl_lag} days")

# # Smooth shrimp distance with cubic interpolation
# dates_num = mdates.date2num(daily_mean['date'])
# # Ensure data is sorted by date
# sort_idx = np.argsort(dates_num)
# dates_num = dates_num[sort_idx]
# distances = daily_mean['distance_mm'].values[sort_idx]

# # Create interpolation function
# interp_func = interp1d(dates_num, distances, kind='cubic')

# # Generate smooth points
# num_smooth = np.linspace(dates_num.min(), dates_num.max(), 300)
# dist_smooth = interp_func(num_smooth)
# dates_smooth = mdates.num2date(num_smooth)

# # Plot combined figure
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # Upper panel: tide_amp and tide_level
# ax1.plot(daily_mean['date'], daily_mean['tide_amp'], label='Tide Amplitude')
# ax1.plot(daily_mean['date'], daily_mean['tide_level'], label='Tide Level')
# ax1.set_ylabel('Tide (mm)')
# ax1.legend()
# ax1.set_title('Daily Tide Metrics')
# ax1.text(0.02, 0.85,
#         f"Tide Amplitude CCF: r={best_amp_corr:.2f}, lag {best_amp_lag}d\n"
#         f"Tide Level CCF: r={best_lvl_corr:.2f}, lag {best_lvl_lag}d",
#         transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

# # Lower panel: shrimp distance (scatter + smooth curve)
# ax2.scatter(daily_mean['date'], daily_mean['distance_mm'], label='Shrimp Distance', s=20, alpha=0.7)
# ax2.plot(dates_smooth, dist_smooth, label='Smoothed Distance', linestyle='--')
# ax2.set_ylabel('Distance (mm)')
# ax2.set_xlabel('Date')
# ax2.set_title('Shrimp Movement')
# ax2.legend()

# plt.tight_layout()
# plt.savefig('shrimp_tide_correlation.png', dpi=300)
# plt.show()

