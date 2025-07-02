import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multitest import multipletests
import string
import scipy.stats as stats

# Function to assign significance letters
def assign_letters(p_values_matrix, labels, alpha=0.05):
    n = len(labels)
    # Initialize letters for each group
    group_letters = {label: set() for label in labels}
    available_letters = list(string.ascii_lowercase)
    
    # Check all pairs of groups
    for i in range(n):
        for j in range(i+1, n):
            label_i, label_j = labels[i], labels[j]
            p_val = p_values_matrix.loc[label_i, label_j]
            
            # If no significant difference, assign the same letter
            if p_val > alpha:
                # Shared alphabet logic
                if not group_letters[label_i] and not group_letters[label_j]:
                    # If neither group has letters, assign a new one
                    next_letter = available_letters.pop(0) if available_letters else 'z+'
                    group_letters[label_i].add(next_letter)
                    group_letters[label_j].add(next_letter)
                elif group_letters[label_i] and not group_letters[label_j]:
                    # If only group i has letters, share with group j
                    group_letters[label_j].update(group_letters[label_i])
                elif not group_letters[label_i] and group_letters[label_j]:
                    # If only group j has letters, share with group i
                    group_letters[label_i].update(group_letters[label_j])
                else:
                    # If both have letters, merge them
                    common_letters = group_letters[label_i].union(group_letters[label_j])
                    group_letters[label_i] = common_letters
                    group_letters[label_j] = common_letters
    
    # Ensure each group has at least one letter
    for label in labels:
        if not group_letters[label] and available_letters:
            group_letters[label].add(available_letters.pop(0))
        elif not group_letters[label]:
            group_letters[label].add('z+')
    
    # Format results
    result = {}
    for label in labels:
        sorted_letters = ''.join(sorted(group_letters[label]))
        result[label] = sorted_letters
        
    return result

# ------ Load data ------
# Load shrimp movement data (replace with your actual path)
shrimp_path = 'D:/khaki/ultralytics-8.3.27/shrimp/distance/all.csv'  
shrimp = pd.read_csv(shrimp_path, parse_dates=['time'])
shrimp.rename(columns={
    'time': 'datetime',
    'distance_permin_pernumber': 'distance_mm'
}, inplace=True)

# Clean data
shrimp = shrimp[['datetime', 'distance_mm']].dropna()
shrimp['datetime'] = pd.to_datetime(shrimp['datetime'], infer_datetime_format=True)
shrimp['year'] = shrimp['datetime'].dt.year

# Group data by year
year_data = shrimp.groupby('year')['distance_mm'].apply(list).reset_index()
years = sorted(year_data['year'].unique())
print(f"Years in dataset: {years}")
x = 0
for i in years:
    j=year_data["year"]
    y = year_data["distance_mm"]
    print(f"{i}:{len(y[x])} ")
    x = x+1
# ------ Statistical Analysis: Kruskal-Wallis + Dunn's Multiple Comparison ------
# Prepare data for Kruskal-Wallis test
data_list = [year_data[year_data['year'] == year]['distance_mm'].iloc[0] for year in years]
labels = [str(year) for year in years]

# Perform Kruskal-Wallis test
print("\nNonparametric method: Kruskal-Wallis test")
h_stat, p_kw = stats.kruskal(*data_list)
print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_kw:.4f} {'(Significant differences exist)' if p_kw<0.05 else '(No significant differences)'}")

# If significant, perform post-hoc Dunn's test
if p_kw < 0.05:
    # Prepare the data format for Dunn's test
    df_for_dunn = pd.DataFrame({
        'value': np.concatenate(data_list),
        'group': np.concatenate([[i]*len(g) for i, g in enumerate(data_list)])
    })
    
    # Dunn's test
    dunn_result = posthoc_dunn(df_for_dunn, val_col='value', group_col='group', p_adjust='fdr_bh')
    
    # Convert numeric index back to year labels
    dunn_result.index = labels
    dunn_result.columns = labels
    print("\nDunn's test multiple comparisons:")
    print(dunn_result)
    
    # Create letter grouping
    letter_groups = assign_letters(dunn_result, labels)
    
    # Print results
    print("\nLetter labeling based on Dunn's test (same letters indicate no significant difference):")
    for label, letters in letter_groups.items():
        print(f"{label}: {letters}")

# ------ Drawing: Violin plot with significance letters ------
fig, ax = plt.subplots(figsize=(12, 6))

# Create positions for each year
positions = np.arange(len(years))

# Prepare data for violin plot
violin_data = [year_data[year_data['year'] == year]['distance_mm'].iloc[0] for year in years]

# Create violin plot
vp = ax.violinplot(
    violin_data,
    positions=positions,
    showmeans=False,
    showmedians=True
)
for body in vp['bodies']:
    body.set_facecolor('lightblue')
vp['cmedians'].set_color('darkred')

# Add scatter points with jitter
rng = np.random.RandomState(42)
jitter = 0.1
for i, vals in enumerate(violin_data):
    x = rng.normal(positions[i], jitter, size=len(vals))
    ax.scatter(x, vals, color='lightblue', s=10, alpha=0.5)

# Add mean line
means = [np.mean(vals) for vals in violin_data]
ax.plot(positions, means, '-o', color='lightblue', linewidth=2, label='Yearly Mean',alpha = 0.5)

# Add letter annotations if statistical test was performed
if p_kw < 0.05 and 'letter_groups' in locals():
    # Find maximum y value for annotation placement
    y_max = max([np.max(vals) for vals in violin_data]) * 1.02
    
    # Get unique letter combinations
    unique_letters = set(letter_groups.values())
    
    # Create color map
    color_map = plt.cm.get_cmap('tab10', len(unique_letters))
    letter_to_color = {letter: color_map(i) for i, letter in enumerate(unique_letters)}
    
    # Add letter annotations
    for i, year in enumerate(years):
        year_str = str(year)
        if year_str in letter_groups:
            letter = letter_groups[year_str]
            color = letter_to_color[letter]
            ax.text(positions[i], y_max, letter, 
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=color)
    
    # Adjust y-axis to accommodate annotations
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(current_ymin, y_max * 1.05)

# Format plot
ax.set_xticks(positions)
ax.set_xticklabels(years)
ax.set_xlabel('Year')
ax.set_ylabel('Shrimp Distance (mm)')
ax.set_title('Yearly Shrimp Movement Analysis with Statistical Grouping')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('yearly_shrimp_analysis.png', dpi=300)
plt.show()