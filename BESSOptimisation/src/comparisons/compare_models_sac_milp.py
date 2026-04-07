
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BESSOptimisation.src.milp.milp_optimiser_day_ahead import BESS_Optimiser #MILP
from BESSOptimisation.src.rl.dayahead_agent import BESSEnv   #RL

def run_comparison(test_df, battery_params, rl_model_path, vec_norm_path):
    """Runs MILP and RL on the same data and plots the results together."""
    
    # Running MILP
    print("Running MILP Optimisation...")
    milp_opt = BESS_Optimiser(test_df, battery_params)
    milp_opt.define_variables()
    milp_opt.set_objective()
    milp_opt.set_constraints()
    milp_results = milp_opt.solve_and_collect()
    
    if milp_results is None:
        print("MILP failed to find a solution.")
        return

    # Running RL Agent
    print("Running RL Agent Inference...")
    
    # Verify files exist
    if not os.path.exists(rl_model_path):
        print(f"RL Model not found at {rl_model_path}")
        return
    if not os.path.exists(vec_norm_path):
        print(f"Normalization file not found at {vec_norm_path}")
        return

    # Wrap the environment so it can scale the inputs exactly like it did during training
    eval_env = DummyVecEnv([lambda: BESSEnv(test_df, battery_params)])
    eval_env = VecNormalize.load(vec_norm_path, eval_env)
    eval_env.training = False # Don't update scaling metrics during evaluation
    eval_env.norm_reward = False # Keep rewards in true £ values
    
    # Load the trained model
    model = SAC.load(rl_model_path, env=eval_env)
    
    obs = eval_env.reset()
    
    # Stop at len(test_df) - 2 to prevent DummyVecEnv from auto-resetting and wiping history
    for _ in range(len(test_df) - 2):
        action, _ = model.predict(obs, deterministic=True)
        # DummyVecEnv step returns 4 values, and done is an array
        obs, reward, done, info = eval_env.step(action)
        if done[0]: break
    
    # Extract history from the underlying environment
    rl_results = pd.DataFrame(eval_env.envs[0].history)
    rl_results.index = test_df.index[:len(rl_results)]

    # Consolidated Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
    
    # SOC Comparison
    ax1.plot(milp_results.index, milp_results['SOC'], label='MILP (Optimal)', color='black', linewidth=2)
    ax1.plot(rl_results.index, rl_results['SOC'], label='RL Agent', color='tab:blue', linestyle='--')
    ax1.set_ylabel('SOC (MWh)')
    ax1.set_title('Strategy Comparison: State of Charge (SOC)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Price vs Power (Dispatch Correlation)
    ax2_price = ax2.twinx()
    ax2_price.plot(test_df.index, test_df['DayAhead'], color='gray', alpha=0.2, label='Price')
    
    # Using step plots to see discrete battery actions clearly
    ax2.step(milp_results.index, milp_results['Power'], where='post', label='MILP Power (MW)', color='black', alpha=0.7)
    ax2.step(rl_results.index, rl_results['Power_DayAhead'], where='post', label='RL Power (MW)', color='tab:blue', alpha=0.6)
    
    ax2.set_ylabel('Power (MW)')
    ax2_price.set_ylabel('Price (£/MWh)')
    ax2.set_title('Dispatch Timing vs. Price')
    ax2.legend(loc='upper left')

    # Cumulative Profit
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

    # Numerical Summary
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
    
    full_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
    
    # Select a specific test window
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
    
    # Pass the normalised model name and the vec_normalize.pkl file
    run_comparison(test_window, params, "models/bess_sac_lookahead_normalized.zip", "models/vec_normalize.pkl")