import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DayAheadMILP_optimiser import BESS_Optimiser # Your MILP Class
from dayahead_agent import BESSEnv   # Your RL Env
from stable_baselines3 import SAC

def run_comparison(test_df, battery_params, rl_model_path):
    """Runs MILP and RL on the same data and plots the results together."""
    
    # --- 1. RUN MILP (The Benchmark) ---
    print("Running MILP Optimisation...")
    milp_opt = BESS_Optimiser(test_df, battery_params)
    milp_opt.define_variables()
    milp_opt.set_objective()
    milp_opt.set_constraints()
    milp_results = milp_opt.solve_and_collect()
    
    if milp_results is None:
        print("MILP failed to find a solution.")
        return

    # --- 2. RUN RL AGENT (The Challenger) ---
    print("Running RL Agent Inference...")
    env = BESSEnv(test_df, battery_params)
    
    # Load the trained model
    if not os.path.exists(rl_model_path):
        print(f"RL Model not found at {rl_model_path}")
        return
        
    model = SAC.load(rl_model_path, env=env)
    
    obs, _ = env.reset()
    for _ in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done: break
    
    rl_results = pd.DataFrame(env.history)
    rl_results.index = test_df.index[:len(rl_results)]

    # --- 3. CONSOLIDATED PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    
    # Plot 1: SOC Comparison
    ax1.plot(milp_results.index, milp_results['SOC'], label='MILP (Optimal)', color='black', linewidth=2)
    ax1.plot(rl_results.index, rl_results['SOC'], label='RL Agent', color='tab:blue', linestyle='--')
    ax1.set_ylabel('SOC (MWh)')
    ax1.set_title('Strategy Comparison: State of Charge (SOC)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Price vs Power (Dispatch Correlation)
    ax2_price = ax2.twinx()
    ax2_price.plot(test_df.index, test_df['DayAhead'], color='gray', alpha=0.2, label='Price')
    
    # Using step plots to see discrete battery actions clearly
    ax2.step(milp_results.index, milp_results['Power'], where='post', label='MILP Power (MW)', color='black', alpha=0.7)
    ax2.step(rl_results.index, rl_results['Power_DayAhead'], where='post', label='RL Power (MW)', color='tab:blue', alpha=0.6)
    
    ax2.set_ylabel('Power (MW)')
    ax2_price.set_ylabel('Price (£/MWh)')
    ax2.set_title('Dispatch Timing vs. Price')
    ax2.legend(loc='upper left')

    # Plot 3: Cumulative Profit
    # Use 'Hourly_Profit' from MILP and 'Revenue' (net of penalty) from RL
    milp_cum_profit = milp_results['Hourly_Profit'].cumsum()
    rl_cum_profit = (rl_results['Revenue'] - rl_results['Penalty']).cumsum()
    
    ax3.plot(milp_results.index, milp_cum_profit, label='MILP Profit', color='black', linewidth=2)
    ax3.plot(rl_results.index, rl_cum_profit, label='RL Profit', color='tab:orange', linestyle='--')
    ax3.set_ylabel('Cumulative Net Profit (£)')
    ax3.set_title('Revenue Capture Efficiency')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- 4. NUMERICAL SUMMARY ---
    total_milp = milp_cum_profit.iloc[-1]
    total_rl = rl_cum_profit.iloc[-1]
    efficiency = (total_rl / total_milp) * 100

    print("\n" + "="*40)
    print(f"COMPARISON SUMMARY")
    print("="*40)
    print(f"MILP Total Profit:  £{total_milp:,.2f}")
    print(f"RL Total Profit:    £{total_rl:,.2f}")
    print(f"Agent Efficiency:   {efficiency:.2f}% of Optimal")
    print(f"MILP Cycles:        {milp_results['Throughput'].sum() / battery_params['capacity']:.2f}")
    print(f"RL Cycles:          {rl_results['Throughput'].sum() / battery_params['capacity']:.2f}")
    print("="*40)

if __name__ == "__main__":
    import os
    # Load your shared data
    full_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
    
    # Select a specific test window (e.g., 7 days of data not used heavily in training)
    # 11640 is roughly 8 months into the year
    test_window = full_df.iloc[11640 : 11640 + (48 * 7)] 
    
    params = {
        'time_interval': 0.5, 
        'max_power': 10, 
        'capacity': 20, 
        'rte': 0.9,
        'soc_min_factor': 0.1, 
        'soc_max_factor': 0.9, 
        'soc_initial_factor': 0.5,
        'deg_per_mwh': 0.00001
    }
    
    run_comparison(test_window, params, "bess_sac_lookahead.zip")