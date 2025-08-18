import pandas as pd
import matplotlib.pyplot as plt
#  the font size and family for plots
plt.rcParams["font.size"] = 16
plt.rcParams['font.family'] = 'Arial'
# Read Simplex results
simplex_path = 'D:/khaki/ultralytics-8.3.27/shrimp/R/2016-2017simplex.csv'
simplex_df = pd.read_csv(simplex_path, usecols=['E', 'rho'])
simplex_df = simplex_df.dropna()
simplex_df['E'] = simplex_df['E'].astype(int)
simplex_df['rho'] = simplex_df['rho'].astype(float)
E_best = simplex_df.loc[simplex_df['rho'].idxmax(), 'E']

# Read S-map results
smap_path = 'D:/khaki/ultralytics-8.3.27/shrimp/R/2016-2017smap.csv'
smap_df = pd.read_csv(smap_path, usecols=['theta', 'rho'])
smap_df = smap_df.dropna()
smap_df['theta'] = smap_df['theta'].astype(float)
smap_df['rho'] = smap_df['rho'].astype(float)
theta_best = smap_df.loc[smap_df['rho'].idxmax(), 'theta']

# Plot Simplex graph
plt.figure(figsize=(10, 5))
plt.plot(simplex_df['E'], simplex_df['rho'], '-o', color='steelblue', markerfacecolor='#c4db86', markersize=8, alpha=0.7)
plt.axvline(x=E_best, color='olive', linestyle='dashed', linewidth=2, alpha=0.7)
plt.title(f'Simplex LOOCV: E_best = {E_best}')
plt.xlabel('Embedding dimension E')
plt.ylabel('Forecast skill (rho)')
plt.grid(True, linestyle='dashed', alpha=0.5)
plt.tight_layout()
plt.savefig('simplex_py.pdf', dpi=150)
plt.show()

# Plot S-map graph
plt.figure(figsize=(10, 5))
plt.plot(smap_df['theta'], smap_df['rho'], '-o', color='steelblue', markerfacecolor='#c4db86', markersize=8, alpha=0.7)
plt.axvline(x=theta_best, color='olive', linestyle='dashed', linewidth=2, alpha=0.7)
plt.title(f'S-map (E = {E_best}, θ = {theta_best}): ρ vs θ')
plt.xlabel('Theta (Nonlinearity Parameter)')
plt.ylabel('Forecast skill (rho)')
plt.grid(True, linestyle='dashed', alpha=0.5)
plt.tight_layout()
plt.savefig('smap_py.pdf', dpi=150)
plt.show()