import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy_financial as npf # Industry standard for NPV/IRR

def calculate_investment_metrics(summary_df, battery_params):
    # 1. Setup Parameters
    p_max_mw = battery_params.get('max_power', 10)
    cap_mwh = battery_params.get('capacity', 20)
    capex_per_mwh = 135000
    opex_per_mw_year = 6000
    discount_rate = 0.09
    
    # 2. Initial Investment (Year 0)
    initial_investment = cap_mwh * capex_per_mwh
    
    # 3. Aggregate Daily Data to Annual
    yearly_data = summary_df.resample('YE', on='Date').agg({
        'Total_Profit': 'sum',
        'SOH': 'mean'
    })
    
    # 4. Cash Flow Calculation
    # We subtract Fixed OPEX from the Operational Profit
    yearly_data['Net_Cash_Flow'] = yearly_data['Total_Profit'] - (p_max_mw * opex_per_mw_year)
    
    # List of cash flows starting with negative CAPEX at Year 0
    cash_flows = [-initial_investment] + yearly_data['Net_Cash_Flow'].tolist()
    
    # 5. Financial Metrics
    npv = npf.npv(discount_rate, cash_flows)
    irr = npf.irr(cash_flows)
    
    # Calculate Payback Period
    cum_cf = np.cumsum(cash_flows)
    payback_year = next((i for i, v in enumerate(cum_cf) if v > 0), None)
    
    return {
        "npv": npv,
        "irr": irr,
        "payback_year": payback_year,
        "cum_cash_flow": cum_cf,
        "yearly_cf": cash_flows
    }




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

def generate_hybrid_skip_matrix(df, markets, p_site_fault=0.005, p_ancillary_fault=0.02, p_bm_skip=0.80, block_size=8):
    """
    Implements a hybrid skip logic with specific BM constraints:
    1. Site-wide faults: Battery offline for ALL markets.
    2. Ancillary-specific faults: Battery skips only ancillary markets.
    3. BM Skip Rate: 80% chance of being bypassed by the ESO Control Room.
    """
    skip_df = pd.DataFrame(0, index=df.index, columns=markets)
    ancillary_markets = ['DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh']
    
    for start in range(0, len(df), block_size):
        end = min(start + block_size, len(df))
        current_indices = df.index[start:end]

        # 1. Site-wide outage (All markets)
        if np.random.rand() < p_site_fault:
            skip_df.loc[current_indices, :] = 1
            continue  
            
        # 2. Category-wide ancillary outage (DCDM, DR)
        if np.random.rand() < p_ancillary_fault:
            cols_to_skip = [m for m in markets if m in ancillary_markets]
            skip_df.loc[current_indices, cols_to_skip] = 1
        
        # 3. Balancing Mechanism (BM) specific skip rate
        # This applies an 80% skip rate specifically to the BM market
        if 'BM' in markets:
            if np.random.rand() < p_bm_skip:
                skip_df.loc[current_indices, 'BM'] = 1
                
    return skip_df