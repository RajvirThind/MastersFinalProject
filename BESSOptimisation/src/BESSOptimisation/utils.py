import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



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

def plot_soc_preview(dispatch_df):
    """Plots the State of Charge (SOC) preview from the dispatch DataFrame."""
    if dispatch_df is None or dispatch_df.empty:
        print("Dispatch DataFrame is empty. Cannot plot SOC preview.")
        return
    first_week_preview_df = dispatch_df.head(336)  # First Week
    last_week_preview_df = dispatch_df.tail(336)  # Last Week
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plotting first week preview
    ax1.plot(first_week_preview_df.index, first_week_preview_df['SOC'], label='SOC', color='blue', marker='o')
    ax1.set_title('State of Charge (SOC) - First Week')
    ax1.set_ylabel('SOC (MWh)')
    ax1.grid(True)
    ax1.legend()

    # Plotting last week preview
    ax2.plot(last_week_preview_df.index, last_week_preview_df['SOC'], label='SOC', color='blue', marker='o')
    ax2.set_title('State of Charge (SOC) - Last Week')
    ax2.set_ylabel('SOC (MWh)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def generate_block_skip_matrix(df, markets, p_skip=0.02, block_size=8):
    """
    Ensures that if a skip happens, it lasts for the whole 4-hour block.
    """
    skip_df = pd.DataFrame(0, index=df.index, columns=markets)
    
    # Iterate through each 4-hour window
    for start in range(0, len(df), block_size):
        for m in markets:
            # Determine if this specific market block is skipped
            if np.random.rand() < p_skip:
                end = min(start + block_size, len(df))
                skip_df.iloc[start:end, skip_df.columns.get_loc(m)] = 1
                
    return skip_df