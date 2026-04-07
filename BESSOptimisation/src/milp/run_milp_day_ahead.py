import pandas as pd
from milp_optimiser_day_ahead import BESS_Optimiser

def simulate():
    #Load Data
    full_price_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
    
    battery_params = {
        'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 'rte': 0.9,
        'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'soc_initial_factor': 0.5,
        'cycle_limit': 1.1, 'deg_per_mwh': 0.00001
    }

    current_soh = 1.0
    last_soc = battery_params['soc_initial_factor'] * battery_params['capacity']
    all_dispatch_data = [] 
    
    steps_per_day = 48
    sim_days = 10 

    print(f"--- Running Day Ahead Simulation for {sim_days} days ---")

    for i in range(sim_days):
        start_idx = i * steps_per_day
        day_df = full_price_df.iloc[start_idx : start_idx + steps_per_day]
        if len(day_df) < steps_per_day: break

        opt = BESS_Optimiser(day_df, battery_params, current_soh, last_soc)
        opt.define_variables()
        opt.set_objective()
        opt.set_constraints()
        results = opt.solve_and_collect()

        if results is not None:
            # Updating SOH and SOC for the next day
            # This line matches the 'Throughput' column in the class
            current_soh -= (results['Throughput'].sum() * battery_params['deg_per_mwh'])
            last_soc = results['SOC'].iloc[-1]
            
            all_dispatch_data.append(results)
            print(f"Processed Day {i+1}: {day_df.index[0].date()}")
        else:
            print(f"Solver failed on Day {i+1}. Skipping.")

    # Final combined output
    if all_dispatch_data:
        full_results = pd.concat(all_dispatch_data)
        
        total_profit = full_results['Hourly_Profit'].sum()
        print("\n" + "="*30)
        print(f"TOTAL PROFIT:  £{total_profit:,.2f}")
        print(f"FINAL SOH:     {current_soh:.2%}")
        print("="*30)

        # Output the single combined graph
        BESS_Optimiser.plot_full_period(full_results)

if __name__ == "__main__":
    simulate()