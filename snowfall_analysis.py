import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

RESORTS = ["Vail", "Breckenridge", "Winter Park", "Steamboat", "Copper Mountain"]

def process_snowfall_data(data):
    """Process raw NOAA data into monthly snowfall totals by ski season."""
    if not data or 'results' not in data or not data['results']:
        return pd.DataFrame()
    
    df = pd.DataFrame(data['results'])
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    df['season'] = df['year']
    df.loc[df['month'] <= 6, 'season'] = df['year'] - 1
    
    df = df[
        ((df['month'] >= 7) & (df['season'] == df['year'])) |  
        ((df['month'] <= 6) & (df['season'] == df['year'] - 1))
    ]
        
    df['value'] = df['value'] / 25.4  #Coververt to inches
    
    monthly_totals = df.groupby(['season', 'month'])['value'].sum().reset_index()
        
    return monthly_totals

def calculate_monthly_predictions(data):
    """Calculate predicted snowfall for each month of the 2025-2026 ski season 
    using historical data patterns."""

    ski_season_months = [10, 11, 12, 1, 2, 3, 4, 5]
    monthly_predictions = {}
    
    for month in ski_season_months:
        month_data = data[data['month'] == month]
        
        if month_data.empty:
            print(f"No data available for month {month}")
            monthly_predictions[month] = 0
            continue
        
        seasons = month_data['season'].values
        values = month_data['value'].values
        
        if len(seasons) > 1:
            min_season = min(seasons)
            max_season = max(seasons)
            season_range = max_season - min_season
            
            if season_range > 0:
                normalized_seasons = (seasons - min_season) / season_range
                weights = np.exp(2 * normalized_seasons)
                weights = weights / weights.sum()
                
                prediction = np.sum(values * weights) * 1.1 
            else:
                prediction = np.mean(values) * 1.1
        else:
            prediction = values[0] * 1.1
            
        monthly_predictions[month] = prediction
    
    return monthly_predictions

def lagrange_interpolation_function(x_points, y_points):
    """Create a Lagrange polynomial interpolation function for the given points."""
    n = len(x_points)
    
    def L(x, i):
        """Calculate the Lagrange basis polynomial L_i(x)."""
        result = 1.0
        for j in range(n):
            if j != i:
                result *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return result
    
    def lagrange_polynomial(x):
        """Calculate the Lagrange polynomial at point x."""
        result = 0.0
        for i in range(n):
            result += y_points[i] * L(x, i)
        return result
    
    return lagrange_polynomial

def calculate_total_snowfall(interp_func, start, end, n = 1000):
    """Integrate under the curve using simpons rule."""
    x = np.linspace(start, end, n+1)
    y = np.array([max(0, interp_func(xi)) for xi in x])

    if n % 2 == 1:
            n += 1
            x = np.linspace(start, end, n+1)
            y = np.array([max(0, interp_func(xi)) for xi in x])
        
    return (end - start) / (3 * n) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

def plot_resort_prediction(resort, monthly_data, monthly_predictions):
    """Create plots for a single resort with predictions and Lagrange interpolation."""
    print(f"\nProcessing {resort}...")
    
    if not monthly_predictions:
        print(f"No prediction data available for {resort}")
        return
    
    prediction_months = list(monthly_predictions.keys())
    prediction_values = list(monthly_predictions.values())
    
    x_month_mapping = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    x_prediction = [x_month_mapping[m] for m in prediction_months]
    
    sorted_indices = np.argsort(x_prediction)
    x_prediction_sorted = [x_prediction[i] for i in sorted_indices]
    y_prediction_sorted = [prediction_values[i] for i in sorted_indices]
    months_sorted = [prediction_months[i] for i in sorted_indices]
    
    x_dense = np.linspace(0, 7, 500)
    
    lagrange_func = lagrange_interpolation_function(x_prediction_sorted, y_prediction_sorted)
    y_lagrange = [lagrange_func(x) for x in x_dense]
    
    y_lagrange = [max(0, y) for y in y_lagrange]
    
    total_snowfall = calculate_total_snowfall(lagrange_func, 0, 7)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'{resort} Snowfall Prediction', fontsize=16, y=0.95)
    
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    
    # Plot 1: Monthly Historical Data
    if not monthly_data.empty:
        hist_monthly_avg = monthly_data.groupby('month')['value'].mean().reindex(prediction_months)
        hist_monthly_max = monthly_data.groupby('month')['value'].max().reindex(prediction_months)
        hist_monthly_min = monthly_data.groupby('month')['value'].min().reindex(prediction_months)
        
        for i, month in enumerate(months_sorted):
            ax1.bar(i, hist_monthly_avg[month], color='lightblue', alpha=0.7, width=0.6)
    
    # Plot predicted values on top of historical data
    ax1.plot(range(len(months_sorted)), y_prediction_sorted, 'ro-', markersize=8, label='2025-2026 Prediction')
    ax1.set_title('Monthly Snowfall Prediction vs. Historical Average', fontsize=14)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Snowfall (inches)')
    ax1.set_xticks(range(len(months_sorted)))
    ax1.set_xticklabels([month_names[x_month_mapping[m]] for m in months_sorted])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    for i, (x, y) in enumerate(zip(range(len(months_sorted)), y_prediction_sorted)):
        ax1.annotate(f"{y:.1f}\"", (x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    
    # Plot 2: Lagrange Interpolation Curve and Integration
    ax2.plot(x_dense, y_lagrange, 'b-', linewidth=2, label='Lagrange Interpolation')
    ax2.fill_between(x_dense, 0, y_lagrange, alpha=0.3, color='skyblue')
    ax2.plot(x_prediction_sorted, y_prediction_sorted, 'ro', markersize=8, label='Monthly Predictions')
    
    for i in range(8):
        ax2.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    
    ax2.set_title(f'Snowfall Prediction for 2025-2026 Season\nTotal: {total_snowfall:.1f} inches', 
                 fontsize=14)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Snowfall (inches)')
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(month_names)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    #Add text for total snowfall
    ax2.text(0.02, 0.95, f'Integrated Total: {total_snowfall:.1f} inches',
             transform=ax2.transAxes, fontsize=14, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    print("\nPredicted Monthly Snowfall for 2025-2026 Season (inches):")
    for month in sorted(monthly_predictions.keys(), key=lambda m: x_month_mapping[m]):
        month_name = month_names[x_month_mapping[month]]
        print(f"{month_name}: {monthly_predictions[month]:.1f}")
    print(f"\nPredicted total snowfall: {total_snowfall:.1f} inches")
    
    plt.tight_layout()
    
    os.makedirs('snowfall_predictions', exist_ok=True)
    plt.savefig(f'snowfall_predictions/{resort.lower().replace(" ", "_")}_prediction.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return total_snowfall

def main():

    data_dir = 'snowfall_data'
    if not os.path.exists(data_dir):
        print("Error: Data directory not found. Please run data_downloader.py first.")
        return
    
    try:
        with open(f'{data_dir}/date_info.json', 'r') as f:
            date_info = json.load(f)
        print(f"Analyzing data from {date_info['start_date']} to {date_info['end_date']}")
        print(f"Data downloaded on: {date_info['download_date']}")
    except FileNotFoundError:
        print("Date info file not found. Data may be incomplete.")
    
    resort_rankings = pd.DataFrame(columns=['Resort', 'Predicted Snowfall (inches)'])
    
    for i, resort in enumerate(RESORTS):
        data_file = f'{data_dir}/{resort.lower().replace(" ", "_")}_raw.json'
        
        try:
            with open(data_file, 'r') as f:
                raw_data = json.load(f)
                
            if raw_data and 'results' in raw_data:
                print(f"\nProcessing {resort}... ({len(raw_data['results'])} records)")
                
                monthly_data = process_snowfall_data(raw_data)
                
                if monthly_data.empty:
                    print(f"No processed data available for {resort}")
                    continue
                    
                monthly_predictions = calculate_monthly_predictions(monthly_data)
                total_snowfall = plot_resort_prediction(resort, monthly_data, monthly_predictions)
                
                resort_rankings.loc[i] = [resort, total_snowfall]
            else:
                print(f"No data available for {resort}")
        except FileNotFoundError:
            print(f"Data file for {resort} not found. Please download the data first.")
    
    if not resort_rankings.empty:
        resort_rankings = resort_rankings.sort_values('Predicted Snowfall (inches)', ascending=False)
        
        print("\n" + "="*50)
        print("RESORT RANKINGS BY PREDICTED 2025-2026 SNOWFALL")
        print("="*50)
        for i, (index, row) in enumerate(resort_rankings.iterrows()):
            print(f"{i+1}. {row['Resort']}: {row['Predicted Snowfall (inches)']:.1f} inches")
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(resort_rankings['Resort'], resort_rankings['Predicted Snowfall (inches)'], 
                       color='skyblue')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{height:.1f}"', ha='center', fontsize=10)
        
        plt.title('Colorado Ski Resorts: Predicted Snowfall for 2025-2026 Season', fontsize=16)
        plt.ylabel('Predicted Snowfall (inches)', fontsize=12)
        plt.xlabel('Resort', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('snowfall_predictions/resort_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        resort_rankings.to_csv('snowfall_predictions/resort_rankings.csv', index=False)
    
    print("\nAnalysis complete! Check the snowfall_predictions directory for results.")

if __name__ == "__main__":
    main()