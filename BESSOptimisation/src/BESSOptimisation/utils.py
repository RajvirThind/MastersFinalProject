import matplotlib.pyplot as plt
import pandas as pd



def plot_long_term_appraisal(summary_df):
    """Generates a dynamic view of revenue streams and battery health based on actual data range."""
    if summary_df is None or summary_df.empty:
        print("Summary DataFrame is empty. Cannot plot.")
        return

    # Ensure Date is datetime objects
    summary_df['Date'] = pd.to_datetime(summary_df['Date'])
    
    # Check if we have enough data to resample; if not, use raw daily data
    # If the simulation is short (< 60 days), we use daily steps. 
    # If long, we resample to weekly to keep the bars readable.
    if len(summary_df) > 60:
        plot_df = summary_df.resample('W', on='Date').sum()
        plot_df['SOH'] = summary_df.resample('W', on='Date').mean()['SOH']
    else:
        plot_df = summary_df.set_index('Date')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Subplot 1: Stacked Revenue Streams ---
    revenue_cols = ['Arbitrage_Profit', 'Ancillary_Low_Profit', 'Ancillary_High_Profit']
    
    # Filter to only existing columns to avoid errors
    existing_cols = [c for c in revenue_cols if c in plot_df.columns]
    
    # Plotting stacked bars
    plot_df[existing_cols].plot(kind='bar', stacked=True, ax=ax1, width=0.8, alpha=0.8)
    
    ax1.set_ylabel('Profit (£)')
    ax1.set_title('BESS Revenue Breakdown (Actual Generated Dates)', fontsize=14)
    ax1.legend(loc='upper left', fontsize='small')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 2: SOH vs Throughput ---
    ax2_twin = ax2.twinx()
    
    # Plot SOH (State of Health) as a line
    ax2.plot(plot_df.index.astype(str), plot_df['SOH'] * 100, color='red', marker='o', label='SOH %', linewidth=2)
    ax2.set_ylabel('State of Health (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot Throughput on the twin axis
    ax2_twin.bar(plot_df.index.astype(str), plot_df['Throughput_MWh'], color='gray', alpha=0.3, label='Throughput (MWh)')
    ax2_twin.set_ylabel('Energy Throughput (MWh)', color='gray')

    ax2.set_title('Degradation and Physical Utilization')
    
    # Adjust X-axis labels to prevent crowding
    plt.xticks(rotation=45)
    
    # Reduce the number of x-ticks displayed if there are too many
    n = max(1, len(plot_df) // 10)
    ax2.set_xticks(ax2.get_xticks()[::n])

    plt.tight_layout()
    plt.show()