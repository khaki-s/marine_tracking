import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

def load_and_prepare_data(file_content, focus_column='distance_permin_pernumber'):
    """
    Load data from CSV content and prepare for analysis.
    """
    # Parse the CSV content
    df = file_content
    
    # Convert time strings to datetime objects
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values('time')
    
    # Convert datetime to numeric values (days since first observation)
    first_day = df['time'].min()
    df['days'] = [(t - first_day).total_seconds() / (24 * 3600) for t in df['time']]
    
    # Extract month for seasonal analysis
    df['month'] = df['time'].dt.month
    
    # Calculate day of year for annual cycle analysis
    df['day_of_year'] = df['time'].dt.dayofyear
    
    return df

def single_component_cosine(x, amplitude, frequency, phase, offset):
    """
    Single cosine component for fitting.
    """
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

def dual_component_cosine(x, amp1, freq1, phase1, amp2, freq2, phase2, offset):
    """
    Dual cosine component model (e.g., for combined tidal patterns).
    """
    return (amp1 * np.cos(2 * np.pi * freq1 * x + phase1) + 
            amp2 * np.cos(2 * np.pi * freq2 * x + phase2) + offset)

def detrended_cosine(x, amplitude, frequency, phase, offset, slope):
    """
    Cosine with linear trend removal.
    """
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset + slope * x

def seasonal_cosine(x, amp_annual, amp_tide, freq_tide, phase_tide, offset):
    """
    Combined annual cycle and tidal rhythm.
    """
    # Annual cycle (365.25 days)
    annual_freq = 1/365.25
    annual_component = amp_annual * np.cos(2 * np.pi * annual_freq * x)
    
    # Tidal component
    tidal_component = amp_tide * np.cos(2 * np.pi * freq_tide * x + phase_tide)
    
    return annual_component + tidal_component + offset

def perform_cosinor_analysis(df, column_name='distance_permin_pernumber'):
    """
    Perform comprehensive cosinor-based rhythmometry on the data.
    """
    x_data = df['days'].values
    y_data = df[column_name].values
    
    # Calculate basic statistics
    mean_value = np.mean(y_data)
    amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2
    
    # Define potential periods to test (in days)
    periods_to_test = [
        # Tidal cycles
        #0.517,  # Lunar quarter-diurnal tide (~6.2 hours)
        #0.538,  # Solar quarter-diurnal tide (~6.5 hours)
        #1.0,    # Daily cycle
        #12.42,  # Lunar semidiurnal tide (~12.42 hours, converted to days)
        14.8,  # Spring-neap cycle
        27.46,  # Tropical lunar month
        #29.53,  # Synodic lunar month (full moon to full moon)
        
        # Seasonal/annual
        #182.62, # Half-year cycle
        #365.25  # Annual cycle
    ]
    
    # Convert periods to frequencies
    frequencies = [1/period for period in periods_to_test]
    
    # Dictionary to store results
    all_results = {}
    
    # 1. Single component cosinor models (basic)
    print("Testing single component cosinor models...")
    single_results = []
    
    for period, freq in zip(periods_to_test, frequencies):
        try:
            # Initial parameter guesses [amplitude, frequency, phase, offset]
            p0 = [amplitude_guess, freq, 0, mean_value]
            
            # Perform curve fitting
            params, covariance = curve_fit(single_component_cosine, x_data, y_data, 
                                          p0=p0, maxfev=10000)
            
            # Calculate predictions and R²
            y_pred = single_component_cosine(x_data, *params)
            ss_total = np.sum((y_data - mean_value)**2)
            ss_residual = np.sum((y_data - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate adjusted R² to account for number of parameters
            n = len(y_data)
            p = 4  # number of parameters
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # Calculate AIC and BIC for model comparison
            resid = y_data - y_pred
            sse = np.sum(resid**2)
            aic = n * np.log(sse/n) + 2*p
            bic = n * np.log(sse/n) + p * np.log(n)
            
            # Calculate MESOR (Midline Estimating Statistic Of Rhythm)
            mesor = params[3]
            
            # Calculate acrophase (time of peak in radians, then convert to days)
            acrophase_rad = -params[2]  # Negative because of how we defined our cosine function
            acrophase_days = (acrophase_rad / (2 * np.pi)) * (1/freq)
            
            single_results.append({
                'model_type': 'single_component',
                'period_days': period,
                'fitted_period': 1/params[1],
                'amplitude': params[0],
                'frequency': params[1],
                'phase_rad': params[2],
                'offset': params[3],
                'mesor': mesor,
                'acrophase_days': acrophase_days,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'aic': aic,
                'bic': bic,
                'params': params
            })
            
            print(f"  Tested period: {period:.2f} days -> R² = {r_squared:.4f}, Adj. R² = {adj_r_squared:.4f}")
            
        except Exception as e:
            print(f"  Failed fitting for period {period:.2f} days: {str(e)}")
    
    # Sort by adjusted R²
    single_results.sort(key=lambda x: x['adj_r_squared'], reverse=True)
    all_results['single_component'] = single_results
    
    # 2. Detrended cosinor model (for long-term trend removal)
    print("\nTesting detrended cosinor models...")
    detrended_results = []
    
    for period, freq in zip(periods_to_test, frequencies):
        try:
            # Initial parameter guesses [amplitude, frequency, phase, offset, slope]
            p0 = [amplitude_guess, freq, 0, mean_value, 0]
            
            # Perform curve fitting
            params, covariance = curve_fit(detrended_cosine, x_data, y_data, 
                                          p0=p0, maxfev=10000)
            
            # Calculate predictions and R²
            y_pred = detrended_cosine(x_data, *params)
            ss_total = np.sum((y_data - mean_value)**2)
            ss_residual = np.sum((y_data - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Calculate adjusted R²
            n = len(y_data)
            p = 5  # number of parameters
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # Calculate AIC and BIC
            resid = y_data - y_pred
            sse = np.sum(resid**2)
            aic = n * np.log(sse/n) + 2*p
            bic = n * np.log(sse/n) + p * np.log(n)
            
            detrended_results.append({
                'model_type': 'detrended',
                'period_days': period,
                'fitted_period': 1/params[1],
                'amplitude': params[0],
                'frequency': params[1],
                'phase_rad': params[2],
                'offset': params[3],
                'slope': params[4],
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'aic': aic,
                'bic': bic,
                'params': params
            })
            
            print(f"  Tested period with trend: {period:.2f} days -> R² = {r_squared:.4f}, Adj. R² = {adj_r_squared:.4f}")
            
        except Exception as e:
            print(f"  Failed detrended fitting for period {period:.2f} days: {str(e)}")
    
    # Sort by adjusted R²
    detrended_results.sort(key=lambda x: x['adj_r_squared'], reverse=True)
    all_results['detrended'] = detrended_results
    
    # 3. Try dual component models (combining two rhythms)
    print("\nTesting dual component models...")
    dual_results = []
    
    # Test combinations of the top 3 single component periods
    if len(single_results) >= 2:
        top_periods = [result['period_days'] for result in single_results[:3]]
        for i, period1 in enumerate(top_periods):
            for period2 in top_periods[i+1:]:
                try:
                    freq1 = 1/period1
                    freq2 = 1/period2
                    
                    # Initial guesses [amp1, freq1, phase1, amp2, freq2, phase2, offset]
                    p0 = [amplitude_guess/2, freq1, 0, amplitude_guess/2, freq2, 0, mean_value]
                    
                    # Perform curve fitting
                    params, covariance = curve_fit(dual_component_cosine, x_data, y_data, 
                                                  p0=p0, maxfev=10000)
                    
                    # Calculate predictions and R²
                    y_pred = dual_component_cosine(x_data, *params)
                    ss_total = np.sum((y_data - mean_value)**2)
                    ss_residual = np.sum((y_data - y_pred)**2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Calculate adjusted R²
                    n = len(y_data)
                    p = 7  # number of parameters
                    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
                    
                    # Calculate AIC and BIC
                    resid = y_data - y_pred
                    sse = np.sum(resid**2)
                    aic = n * np.log(sse/n) + 2*p
                    bic = n * np.log(sse/n) + p * np.log(n)
                    
                    dual_results.append({
                        'model_type': 'dual_component',
                        'period1_days': period1,
                        'period2_days': period2,
                        'fitted_period1': 1/params[1],
                        'fitted_period2': 1/params[4],
                        'amplitude1': params[0],
                        'frequency1': params[1],
                        'phase1_rad': params[2],
                        'amplitude2': params[3],
                        'frequency2': params[4],
                        'phase2_rad': params[5],
                        'offset': params[6],
                        'r_squared': r_squared,
                        'adj_r_squared': adj_r_squared,
                        'aic': aic,
                        'bic': bic,
                        'params': params
                    })
                    
                    print(f"  Tested periods {period1:.2f} and {period2:.2f} days -> R² = {r_squared:.4f}, Adj. R² = {adj_r_squared:.4f}")
                    
                except Exception as e:
                    print(f"  Failed dual component fitting for periods {period1:.2f} and {period2:.2f} days: {str(e)}")
    
    # Sort by adjusted R²
    dual_results.sort(key=lambda x: x['adj_r_squared'], reverse=True)
    all_results['dual_component'] = dual_results
    
    # 4. Try seasonal model with tidal component
    print("\nTesting seasonal model with tidal components...")
    seasonal_results = []
    
    # Get the best tidal period from single component results
    tidal_periods = [r['period_days'] for r in single_results if r['period_days'] < 31]  # Less than monthly
    
    if tidal_periods:
        for tidal_period in tidal_periods[:3]:  # Try top 3 tidal periods
            try:
                tidal_freq = 1/tidal_period
                
                # Initial guesses [amp_annual, amp_tide, freq_tide, phase_tide, offset]
                p0 = [amplitude_guess/2, amplitude_guess/2, tidal_freq, 0, mean_value]
                
                # Perform curve fitting
                params, covariance = curve_fit(seasonal_cosine, x_data, y_data, 
                                              p0=p0, maxfev=10000)
                
                # Calculate predictions and R²
                y_pred = seasonal_cosine(x_data, *params)
                ss_total = np.sum((y_data - mean_value)**2)
                ss_residual = np.sum((y_data - y_pred)**2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Calculate adjusted R²
                n = len(y_data)
                p = 5  # number of parameters
                adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
                
                # Calculate AIC and BIC
                resid = y_data - y_pred
                sse = np.sum(resid**2)
                aic = n * np.log(sse/n) + 2*p
                bic = n * np.log(sse/n) + p * np.log(n)
                
                seasonal_results.append({
                    'model_type': 'seasonal_with_tidal',
                    'annual_amplitude': params[0],
                    'tidal_period_days': 1/params[2],
                    'tidal_amplitude': params[1],
                    'tidal_frequency': params[2],
                    'tidal_phase_rad': params[3],
                    'offset': params[4],
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'aic': aic,
                    'bic': bic,
                    'params': params
                })
                
                print(f"  Tested seasonal model with tidal period {tidal_period:.2f} days -> R² = {r_squared:.4f}, Adj. R² = {adj_r_squared:.4f}")
                
            except Exception as e:
                print(f"  Failed seasonal model fitting with tidal period {tidal_period:.2f} days: {str(e)}")
    
    # Sort by adjusted R²
    seasonal_results.sort(key=lambda x: x['adj_r_squared'], reverse=True)
    all_results['seasonal_with_tidal'] = seasonal_results
    
    # Find overall best model
    best_models = {}
    for model_type, results in all_results.items():
        if results:
            best_models[model_type] = results[0]
    
    # Sort models by adjusted R²
    best_models = dict(sorted(best_models.items(), key=lambda x: x[1]['adj_r_squared'], reverse=True))
    
    return all_results, best_models

def plot_cosinor_results(df, column_name, results, best_models):
    """
    Plot the results of cosinor analysis.
    """
    x_data = df['days'].values
    y_data = df[column_name].values
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(best_models), 1, figsize=(14, 4*len(best_models)))
    
    if len(best_models) == 1:
        axes = [axes]  # Make iterable for single plot case
    
    # Plot each model type
    for i, (model_type, model) in enumerate(best_models.items()):
        ax = axes[i]
        
        # Plot original data
        ax.scatter(df['time'], y_data, alpha=0.6, label='Original Data', s=20)
        
        # Create smooth curve for fitted function
        days_smooth = np.linspace(x_data.min(), x_data.max(), 1000)
        dates_smooth = [df['time'].min() + pd.Timedelta(days=d) for d in days_smooth]
        
        # Calculate fitted values based on model type
        if model_type == 'single_component':
            fitted_curve = single_component_cosine(days_smooth, *model['params'])
            title = f"Single Component: Period = {model['fitted_period']:.2f} days"
            
        elif model_type == 'detrended':
            fitted_curve = detrended_cosine(days_smooth, *model['params'])
            title = f"Detrended: Period = {model['fitted_period']:.2f} days, Slope = {model['slope']:.6f}"
            
        elif model_type == 'dual_component':
            fitted_curve = dual_component_cosine(days_smooth, *model['params'])
            title = f"Dual Component: Periods = {model['fitted_period1']:.2f} & {model['fitted_period2']:.2f} days"
            
        elif model_type == 'seasonal_with_tidal':
            fitted_curve = seasonal_cosine(days_smooth, *model['params'])
            title = f"Seasonal + Tidal: Annual + {model['tidal_period_days']:.2f} days"
        
        # Plot fitted curve
        ax.plot(dates_smooth, fitted_curve, 'r-', linewidth=2, 
                label=f'Fitted Curve: R² = {model["r_squared"]:.4f}, Adj. R² = {model["adj_r_squared"]:.4f}')
        
        # Add labels and title
        ax.set_title(f"{title}")
        ax.set_xlabel('Date')
        ax.set_ylabel(column_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        
    plt.tight_layout()
    fig.autofmt_xdate()
    
    return fig

def decompose_time_series(df, column_name, period=None):
    """
    Perform time series decomposition to extract trend, seasonal, and residual components.
    """
    # If data isn't evenly spaced, resample to daily frequency
    if not df['time'].diff().iloc[1:].std().total_seconds() < 10:  # Allow small variation
        # Create a temporary dataframe with a daily date range
        date_range = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='D')
        temp_df = pd.DataFrame({'date': date_range})
        
        # Merge with original data
        merged = pd.merge_asof(temp_df.sort_values('date'), 
                              df.rename(columns={'time': 'date'}).sort_values('date'),
                              on='date', direction='nearest')
        
        # Interpolate missing values
        ts = merged[column_name].interpolate()
    else:
        ts = df[column_name]
    
    # If no period specified, try to use best period from cosinor analysis
    if period is None:
        # Use average number of days per month as default
        period = 30
    
    # Perform decomposition
    try:
        result = seasonal_decompose(ts, model='additive', period=int(period))
        
        # Create plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        # Original data
        axes[0].plot(ts.index, ts.values)
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        axes[1].plot(result.trend.index, result.trend.values)
        axes[1].set_title('Trend Component')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        axes[2].plot(result.seasonal.index, result.seasonal.values)
        axes[2].set_title(f'Seasonal Component (Period = {period})')
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        axes[3].plot(result.resid.index, result.resid.dropna().values)
        axes[3].set_title('Residual Component')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return result, fig
    except Exception as e:
        print(f"Decomposition failed: {str(e)}")
        return None, None

def analyze_monthly_patterns(df, column_name):
    """
    Analyze patterns based on month.
    """
    # Group by month and calculate statistics
    monthly_stats = df.groupby('month')[column_name].agg(['mean', 'std', 'min', 'max', 'count'])
    
    # Plot monthly patterns
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plot for means with error bars
    ax.bar(monthly_stats.index, monthly_stats['mean'], yerr=monthly_stats['std'], 
           alpha=0.7, capsize=5, color='skyblue', edgecolor='black')
    
    # Add data points count
    for i, count in enumerate(monthly_stats['count']):
        ax.text(i+1, monthly_stats['mean'].iloc[i] + monthly_stats['std'].iloc[i] + 0.5, 
                f'n={count}', ha='center')
    
    # Formatting
    ax.set_title(f'Monthly Pattern Analysis for {column_name}')
    ax.set_xlabel('Month')
    ax.set_ylabel(f'Average {column_name}')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3)
    
    return monthly_stats, fig

def analyze_shrimp_data(file_content):
    """
    Main function to analyze shrimp movement data.
    """
    # Load and prepare data
    df = load_and_prepare_data(file_content)
    
    # Focus column
    column_name = 'distance_permin_pernumber'
    
    print(f"\nAnalyzing {column_name} for rhythmic patterns...")
    
    # 1. Perform comprehensive cosinor analysis
    all_results, best_models = perform_cosinor_analysis(df, column_name)
    
    # 2. Plot the best models
    fig_cosinor = plot_cosinor_results(df, column_name, all_results, best_models)
    
    # 3. Analyze monthly patterns
    monthly_stats, fig_monthly = analyze_monthly_patterns(df, column_name)
    
    # 4. Try time series decomposition
    # Use the period from the best single component model if available
    if all_results['single_component']:
        best_period = all_results['single_component'][0]['fitted_period']
        # Round to nearest integer for decomposition
        decomp_period = int(round(best_period))
    else:
        decomp_period = 30  # Default to monthly
    
    # Perform decomposition
    decomp_result, fig_decomp = decompose_time_series(df, column_name, period=decomp_period)
    
    # Summary of findings
    print("\n=== SUMMARY OF FINDINGS ===")
    print(f"Best models for {column_name}:")
    
    for model_type, model in best_models.items():
        print(f"\n{model_type.upper()} MODEL:")
        
        if model_type == 'single_component':
            print(f"  Period: {model['fitted_period']:.2f} days")
            print(f"  Amplitude: {model['amplitude']:.4f}")
            print(f"  MESOR (mean): {model['mesor']:.4f}")
            print(f"  Acrophase: {model['acrophase_days']:.2f} days")
            
        elif model_type == 'detrended':
            print(f"  Period: {model['fitted_period']:.2f} days")
            print(f"  Amplitude: {model['amplitude']:.4f}")
            print(f"  Linear slope: {model['slope']:.6f} units/day")
            
        elif model_type == 'dual_component':
            print(f"  Period 1: {model['fitted_period1']:.2f} days (Amplitude: {model['amplitude1']:.4f})")
            print(f"  Period 2: {model['fitted_period2']:.2f} days (Amplitude: {model['amplitude2']:.4f})")
            
        elif model_type == 'seasonal_with_tidal':
            print(f"  Annual amplitude: {model['annual_amplitude']:.4f}")
            print(f"  Tidal period: {model['tidal_period_days']:.2f} days")
            print(f"  Tidal amplitude: {model['tidal_amplitude']:.4f}")
        
        print(f"  R²: {model['r_squared']:.4f}")
        print(f"  Adjusted R²: {model['adj_r_squared']:.4f}")
        print(f"  AIC: {model['aic']:.2f}")
        print(f"  BIC: {model['bic']:.2f}")
    
    print("\nMonthly statistics:")
    print(monthly_stats)
    
    # Show all the figures
    plt.show()
    
    return {
        'all_results': all_results,
        'best_models': best_models,
        'monthly_stats': monthly_stats,
        'figures': {
            'cosinor': fig_cosinor,
            'monthly': fig_monthly,
            'decomposition': fig_decomp
        }
    }

# For demonstration purposes
if __name__ == "__main__":
    # Read the file content
    shrimp_path = "D:/khaki/ultralytics-8.3.27/shrimp/distance/2018-2019.csv"
    file_content =pd.read_csv(shrimp_path)
    results = analyze_shrimp_data(file_content)