import pandas as pd
import numpy as np 
from BESSOptimisation import BESS_Optimiser
from BESSOptimisation.src.utils import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_weekly_soc(dispatch_data_list, week_start_day=0):
    """
    Plots the SOC for a 7-day period to show weekly cycling behavior.
    week_start_day determines which day in the list to start the 7-day slice.
    """
    if len(dispatch_data_list) < week_start_day + 7:
        print("Not enough data to plot a full 7-day week from the specified start day.")
        return

    # Concatenate 7 consecutive days of dispatch data
    week_data = pd.concat(dispatch_data_list[week_start_day : week_start_day + 7])

    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot the SOC line
    ax.plot(week_data.index, week_data['SOC'], color='tab:green', linewidth=2, label='State of Charge (MWh)')
    
    # Formatting
    ax.set_title("Typical Weekly Battery State of Charge (SOC)", fontsize=14, fontweight='bold')
    ax.set_ylabel("SOC (MWh)", fontweight='bold')
    ax.set_ylim(bottom=0) # Keeps the y-axis grounded at 0
    
    # Format the X-axis to show days clearly
    days_locator = mdates.DayLocator()
    days_formatter = mdates.DateFormatter('%a %d %b') # e.g., "Mon 01 Jan"
    ax.xaxis.set_major_locator(days_locator)
    ax.xaxis.set_major_formatter(days_formatter)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_finance(fin_data):
    """Generates the J-Curve for NPV and IRR."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fin_data['years'], fin_data['cum_cf'], marker='o', color='navy', label='Cumulative Cash Flow')
    
    # Calculate yearly bars (delta of cumulative)
    annual_flow = np.diff(fin_data['cum_cf'], prepend=fin_data['cum_cf'][0])
    ax.bar(fin_data['years'][1:], annual_flow[1:], alpha=0.3, label='Annual Net Flow')
    
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f"BESS Investment: NPV £{fin_data['npv']:,.0f} | IRR {fin_data['irr']*100:.2f}%")
    ax.set_ylabel("GBP (£)")
    ax.set_xlabel("Project Year")
    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def simulate_bess_operation():
    print("Loading price data...")
    full_price_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
    full_price_df = full_price_df.head(20000) 

    battery_params = {
        'time_interval': 0.5, 
        'max_power': 10, 
        'capacity': 20, 
        'rte': 0.9, 
        'soc_min_factor': 0.1, 
        'soc_max_factor': 0.9,
        'soc_initial_factor': 0.5,
        'cycle_limit': 1.1, 
        'deg_per_mwh': 0.00001, 
        'utilisation_factor': 0.02
    }

    current_soh = 1.0
    last_soc = battery_params['soc_initial_factor'] * battery_params['capacity']
    
    daily_summaries = []
    dispatch_data_list = []

    steps_per_day = 48
    lookahead_steps = 48 

    print("Starting simulation loop...")
    for start_idx in range(0, len(full_price_df), steps_per_day):
        end_idx = min(start_idx + steps_per_day + lookahead_steps, len(full_price_df))
        day_df = full_price_df.iloc[start_idx : end_idx]
        
        if len(day_df) < steps_per_day: 
            break
        
        battery_params['skip_rates'] = generate_hybrid_skip_matrix(day_df, day_df.columns)
        
        opt = BESS_Optimiser(day_df, battery_params, current_soh, last_soc)
        opt.define_variables()
        opt.set_objective()
        opt.set_constraints()
        results = opt.solve_and_collect()

        if results is not None:
            comm_res = results.iloc[:steps_per_day]
            
            # --- FIX: Calculate profit breakdowns required by plot_long_term_appraisal ---
            arb_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.arbitrage_markets)
            anc_low_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.ancillary_low)
            anc_high_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.ancillary_high)
            
            daily_throughput = comm_res['Throughput_MWh'].sum()
            current_soh -= (daily_throughput * battery_params['deg_per_mwh'])
            last_soc = comm_res['SOC'].iloc[-1]

            daily_summaries.append({
                'Date': comm_res.index[0],
                'Total_Profit': comm_res['Total Hourly Profit'].sum(),
                'Arbitrage_Profit': arb_profit,
                'Ancillary_Low_Profit': anc_low_profit,
                'Ancillary_High_Profit': anc_high_profit,
                'SOH': current_soh,
                'Throughput_MWh': daily_throughput
            })
            
            dispatch_data_list.append(comm_res)
        else:
            print(f"Solver failed for period starting {day_df.index[0]}. Resetting SOC.")
            last_soc = (battery_params['capacity'] * current_soh) * 0.5
            # Add dummy data to summary_df to prevent plotting crashes
            daily_summaries.append({
                'Date': day_df.index[0], 'Total_Profit': 0, 'Arbitrage_Profit': 0,
                'Ancillary_Low_Profit': 0, 'Ancillary_High_Profit': 0,
                'SOH': current_soh, 'Throughput_MWh': 0
            })

    
    summary_df = pd.DataFrame(daily_summaries)
    
    if not summary_df.empty:
        # Long Term Appraisal (SOH and Operational Profit Breakdown)
        BESS_Optimiser.plot_long_term_appraisal(summary_df) 

        # Financial Appraisal (NPV/IRR J-Curve)
        fin_results = BESS_Optimiser.run_financial_analysis(summary_df, battery_params)
        plot_finance(fin_results)

    # Operational Preview (Last full day of dispatch)
    if dispatch_data_list:
        last_day_results = dispatch_data_list[-1]
        opt.plot_operation(last_day_results)

        print("Generating weekly SOC chart...")
        plot_weekly_soc(dispatch_data_list, week_start_day=0)

    print(f"Final SOH: {current_soh:.2%}")
    print("Simulation and Financial Analysis complete.")

if __name__ == "__main__":
    simulate_bess_operation()