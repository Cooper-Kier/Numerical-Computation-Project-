import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from snowfall_analysis import process_snowfall_data, calculate_monthly_predictions, lagrange_interpolation_function, calculate_total_snowfall

RESORTS = ["Vail", "Breckenridge", "Winter Park", "Steamboat", "Copper Mountain"]

def validate_prediction_with_actual(resort, data_file):
    """Compare the prediction for 2023-2024 season with actual data."""
    print(f"\nValidating prediction for {resort}...")
    
    try:
        with open(data_file, 'r') as f:
            raw_data = json.load(f)
            
        if not raw_data or 'results' not in raw_data:
            print(f"No data available for {resort}")
            return None
            
        # Get all processed monthly data
        monthly_data = process_snowfall_data(raw_data)
        
        if monthly_data.empty:
            print(f"No processed data available for {resort}")
            return None
            
        # Filter data to exclude the 2023-2024 season
        training_data = monthly_data[monthly_data['season'] < 2023]
        
        if training_data.empty:
            print(f"No training data available for {resort}")
            return None
            
        # Calculate predictions for 2023-2024 using only prior seasons' data
        monthly_predictions_2023 = calculate_monthly_predictions(training_data)
        
        # Get actual data for 2023-2024 season
        actual_data_2023 = monthly_data[monthly_data['season'] == 2023]
        
        if actual_data_2023.empty:
            print(f"No actual data available for 2023-2024 season for {resort}")
            return None
            
        # Create a mapping of predictions and actual values
        ski_season_months = [10, 11, 12, 1, 2, 3, 4, 5]
        comparison = {
            'month': [],
            'predicted': [],
            'actual': []
        }
        
        for month in ski_season_months:
            if month in monthly_predictions_2023:
                comparison['month'].append(month)
                comparison['predicted'].append(monthly_predictions_2023[month])
                
                # Get actual value for this month
                actual_month_data = actual_data_2023[actual_data_2023['month'] == month]
                actual_value = actual_month_data['value'].sum() if not actual_month_data.empty else 0
                comparison['actual'].append(actual_value)
        
        comparison_df = pd.DataFrame(comparison)
        
        # Calculate error metrics
        comparison_df['error'] = comparison_df['predicted'] - comparison_df['actual']
        comparison_df['percent_error'] = (comparison_df['error'] / comparison_df['actual']) * 100
        comparison_df['absolute_error'] = abs(comparison_df['error'])
        
        # Calculate prediction curve and total predicted snowfall
        x_month_mapping = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
        x_prediction = [x_month_mapping[m] for m in comparison_df['month']]
        y_prediction = comparison_df['predicted'].tolist()
        
        indices = np.argsort(x_prediction)
        x_sorted = [x_prediction[i] for i in indices]
        y_sorted = [y_prediction[i] for i in indices]
        
        # Calculate total predicted snowfall using Lagrange interpolation
        if len(x_sorted) > 1:
            lagrange_func = lagrange_interpolation_function(x_sorted, y_sorted)
            total_predicted = calculate_total_snowfall(lagrange_func, 0, 7)
        else:
            print(f"Not enough data points for interpolation")
            total_predicted = sum(y_prediction)
        
        # Calculate total actual snowfall
        total_actual = comparison_df['actual'].sum()
        
        # Plot comparison
        plot_validation_comparison(resort, comparison_df, total_predicted, total_actual)
        
        return {
            'resort': resort,
            'comparison': comparison_df,
            'total_predicted': total_predicted,
            'total_actual': total_actual,
            'rmse': np.sqrt(np.mean(comparison_df['error'] ** 2)),
            'total_error_percent': (total_predicted - total_actual) / total_actual * 100 if total_actual > 0 else float('inf')
        }
    
    except FileNotFoundError:
        print(f"Data file for {resort} not found.")
        return None
    except Exception as e:
        print(f"Error processing {resort}: {str(e)}")
        return None

def plot_validation_comparison(resort, comparison_df, total_predicted, total_actual):
    """Plot comparison between predicted and actual snowfall."""
    month_names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
    x_month_mapping = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    
    # Sort by month order
    comparison_df['month_order'] = comparison_df['month'].map(lambda m: x_month_mapping[m])
    comparison_df = comparison_df.sort_values('month_order')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'{resort} - 2023-2024 Season Prediction Validation', fontsize=16, y=0.95)
    
    # Plot 1: Bar chart comparing predicted vs actual
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comparison_df['predicted'], width, label='Predicted', color='#6495ED')
    bars2 = ax1.bar(x + width/2, comparison_df['actual'], width, label='Actual', color='#FF7F50')
    
    ax1.set_title('Monthly Snowfall: Prediction vs. Actual', fontsize=14)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Snowfall (inches)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([month_names[x_month_mapping[m]] for m in comparison_df['month']])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add data labels
    for i, bars in enumerate([bars1, bars2]):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}\"",
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    # Plot 2: Percent error by month
    ax2.bar(x, comparison_df['percent_error'], color=comparison_df['percent_error'].apply(
        lambda x: '#ff9999' if x > 0 else '#66b3ff'))
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Prediction Error by Month', fontsize=14)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Percent Error (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([month_names[x_month_mapping[m]] for m in comparison_df['month']])
    ax2.grid(True, alpha=0.3)
    
    # Add data labels
    for i, v in enumerate(comparison_df['percent_error']):
        ax2.annotate(f"{v:.1f}%",
                    xy=(i, v),
                    xytext=(0, 5 if v >= 0 else -15),
                    textcoords="offset points",
                    ha='center',
                    fontsize=8)
    
    # Add text with overall metrics
    ax2.text(0.02, 0.95, 
             f"Total Predicted: {total_predicted:.1f}\"\n"
             f"Total Actual: {total_actual:.1f}\"\n"
             f"Total Error: {(total_predicted - total_actual):.1f}\" "
             f"({(total_predicted - total_actual) / total_actual * 100:.1f}%)\n",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('snowfall_validation', exist_ok=True)
    plt.savefig(f'snowfall_validation/{resort.lower().replace(" ", "_")}_validation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data_dir = 'snowfall_data'
    if not os.path.exists(data_dir):
        print("Error: Data directory not found. Please run data_downloader.py first.")
        return
    
    validation_results = []
    
    for resort in RESORTS:
        data_file = f'{data_dir}/{resort.lower().replace(" ", "_")}_raw.json'
        result = validate_prediction_with_actual(resort, data_file)
        
        if result:
            validation_results.append(result)
    
    if validation_results:
        # Create summary comparison table
        summary_df = pd.DataFrame([
            {
                'Resort': r['resort'],
                'Predicted Snowfall (inches)': r['total_predicted'],
                'Actual Snowfall (inches)': r['total_actual'],
                'Error (inches)': r['total_predicted'] - r['total_actual'],
                'Error (%)': r['total_error_percent'],
                'RMSE': r['rmse']
            } for r in validation_results
        ])
        
        summary_df = summary_df.sort_values('Actual Snowfall (inches)', ascending=False)
        
        # Print results
        print("\n" + "="*80)
        print("PREDICTION VALIDATION SUMMARY FOR 2023-2024 SEASON")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
        
        # Save summary table
        summary_df.to_csv('snowfall_validation/validation_summary.csv', index=False)
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        x = np.arange(len(summary_df))
        width = 0.4
        
        plt.bar(x - width/2, summary_df['Predicted Snowfall (inches)'], width, label='Predicted', color='#6495ED')
        plt.bar(x + width/2, summary_df['Actual Snowfall (inches)'], width, label='Actual', color='#FF7F50')
        
        plt.title('Colorado Ski Resorts: 2023-2024 Season Prediction vs. Actual', fontsize=16)
        plt.ylabel('Snowfall (inches)', fontsize=12)
        plt.xlabel('Resort', fontsize=12)
        plt.xticks(x, summary_df['Resort'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add data labels
        for i, resort in enumerate(summary_df['Resort']):
            predicted = summary_df.iloc[i]['Predicted Snowfall (inches)']
            actual = summary_df.iloc[i]['Actual Snowfall (inches)']
            error_pct = summary_df.iloc[i]['Error (%)']
            
            plt.annotate(f"{predicted:.1f}\"",
                        xy=(i - width/2, predicted),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
                        
            plt.annotate(f"{actual:.1f}\"",
                        xy=(i + width/2, actual),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
                        
            plt.annotate(f"{error_pct:.1f}%",
                        xy=(i, min(predicted, actual) / 2),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center',
                        fontsize=9, color='black')
        
        plt.tight_layout()
        plt.savefig('snowfall_validation/validation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nValidation complete! Check the snowfall_validation directory for results.")

if __name__ == "__main__":
    main()