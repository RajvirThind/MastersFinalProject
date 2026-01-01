import pandas as pd
import numpy as np 
from BESSOptimisation import BESS_Optimiser

def simulate_bess_operation():
    print("Loading price data...")
    full_price_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv')
    full_price_df = full_price_df.head(1440)
    full_price_df["Date"] = pd.to_datetime(full_price_df["Date"])
    full_price_df.set_index("Date", inplace=True)

    battery_params = {
        'time_interval': 0.5,
        'max_power': 10,
        'rte': 0.9,
        'capacity': 20,         # Nameplate capacity (MWh)
        'soc_min_factor': 0.1,
        'soc_max_factor': 0.9,
        'soc_initial_factor': 0.5,
        'cycle_limit': 2.0,
        'deg_per_mwh': 0.00001, # Loss in SOH per MWh discharged
        'utilisation_factor': 0.02
    }

    current_soh = 1.0
    last_soc = battery_params['soc_initial_factor'] * battery_params['capacity']
    daily_summaries = []

    steps_per_day = 48
    lookahead_steps = 8 # 4-hour lookahead

    for start_idx in range(0, len(full_price_df), steps_per_day):
        end_idx = min(start_idx + steps_per_day + lookahead_steps, len(full_price_df))
        if (end_idx - start_idx) < steps_per_day: break
        
        day_df = full_price_df.iloc[start_idx:end_idx]
        
        # 1. Instantiate and Solve
        opt = BESS_Optimiser(day_df, battery_params, current_soh, last_soc)
        opt.define_variables()
        opt.set_objective()
        opt.set_constraints()
        results_ext = opt.solve_and_collect()

        # 2. Resiliency Logic
        if results_ext is not None:
            comm_res = results_ext.iloc[:steps_per_day]
            
            # Market Breakdown Calculation
            arb_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.arbitrage_markets)
            anc_low_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.ancillary_low)
            anc_high_profit = sum(comm_res[f'Profit_{m}'].sum() for m in opt.ancillary_high)
            
            daily_profit = comm_res['Total Hourly Profit'].sum()
            daily_throughput = comm_res['Throughput_MWh'].sum()
            
            current_soh -= (daily_throughput * battery_params['deg_per_mwh'])
            last_soc = comm_res['SOC'].iloc[-1]

            daily_summaries.append({
                'Date': comm_res.index[0],
                'Total_Profit': daily_profit,
                'Arbitrage_Profit': arb_profit,
                'Ancillary_Low_Profit': anc_low_profit,
                'Ancillary_High_Profit': anc_high_profit,
                'SOH': current_soh,
                'Throughput_MWh': daily_throughput,
                'Capacity_MWh': opt.current_capacity
            })
        else:
            print(f"Day starting {day_df.index[0]} Infeasible. Resetting SOC to 50%.")
            last_soc = (battery_params['capacity'] * current_soh) * 0.5
            daily_summaries.append({'Date': day_df.index[0], 'Profit': 0, 'SOH': current_soh, 'Throughput_MWh': 0})

    # 3. Export
    summary_df = pd.DataFrame(daily_summaries)
    summary_df.to_csv("Data/Ten_Year_Appraisal_Summary.csv", index=False)
    print("Simulation Complete.")

if __name__ == "__main__":
    simulate_bess_operation()