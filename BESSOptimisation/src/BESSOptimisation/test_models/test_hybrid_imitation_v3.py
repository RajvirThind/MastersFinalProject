import sys
import os

# 1. Get the directory of the current script (test_models)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the parent directory (BESSOptimisation)
parent_dir = os.path.dirname(current_dir)

# 3. Add the parent directory to Python's search path
sys.path.append(parent_dir)

# --- NOW perform your normal imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import SAC

# Python will now successfully find this in the parent folder!
from BESSOptimisation.milp.DayAheadMILP_optimiser import BESS_Optimiser 
from BESSOptimisation.src.BESSOptimisation.test_models.dayahead_agent_v3 import BESSEnv # Make sure this name matches your actual agent file name

def run_hybrid_test(df, battery_params, model_path, num_days=7):
    print(f"--- Running Hybrid Validation for {num_days} Days ---")
    
    #SETUP ENV AND MODEL
    # We use the same lookahead window (48 steps)
    test_env = BESSEnv(df, battery_params)
    model = SAC.load(model_path, env=test_env)
    
    #RUN RL INFERENCE (The Student)
    obs, _ = test_env.reset()
    for _ in range(num_days * 48):
        # Deterministic=True ensures the agent uses the 'mean' action it learned
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if done: break
    
    rl_results = pd.DataFrame(test_env.history)
    rl_results.index = df.index[:len(rl_results)]

    #RUN MILP OPTIMISATION (The Teacher)
    # We run the MILP in a single block for the whole test period for comparison
    milp_opt = BESS_Optimiser(df.iloc[:num_days*48], battery_params)
    milp_opt.define_variables()
    milp_opt.set_objective()
    milp_opt.set_constraints()
    milp_results = milp_opt.solve_and_collect()

    # 4. CONSOLIDATED GRAPHS (Refined for 7-Day View)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # Common X-axis formatting for all subplots
    days_locator = mdates.DayLocator()
    days_formatter = mdates.DateFormatter('%a %d %b') # e.g., "Mon 01 Jan"

    # Plot 1: SOC Overlap
    ax1.plot(milp_results.index, milp_results['SOC'], label='MILP Expert (Teacher)', color='black', linewidth=2.5)
    ax1.plot(rl_results.index, rl_results['SOC'], label='RL Hybrid (Student)', color='tab:cyan', linestyle='--', alpha=0.9)
    ax1.set_ylabel('SOC (MWh)', fontweight='bold')
    ax1.set_title('Strategic Alignment: Did the RL Agent learn the MILP Strategy?', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Plot 2: Dispatch Timing vs Price
    ax2_price = ax2.twinx()
    # Fill between for price makes the dispatch bars stand out more
    ax2_price.fill_between(df.index[:num_days*48], 0, df['DayAhead'].iloc[:num_days*48], color='gray', alpha=0.1, label='Market Price')
    
    # Use step(where='post') to accurately reflect half-hourly settlement periods
    ax2.step(milp_results.index, milp_results['Power'], where='post', label='MILP Power', color='black', alpha=0.7, linewidth=1.5)
    ax2.step(rl_results.index, rl_results['Power_DayAhead'], where='post', label='RL Hybrid Power', color='tab:cyan', alpha=0.8, linewidth=1.2)
    
    ax2.set_ylabel('Power (MW)', fontweight='bold')
    ax2_price.set_ylabel('Price (£/MWh)', color='gray')
    ax2.set_title('Dispatch Timing Comparison', fontsize=12)
    ax2.legend(loc='upper left')

    # Plot 3: Financial Performance
    milp_cum = milp_results['Hourly_Profit'].cumsum()
    rl_cum = rl_results['Revenue'].cumsum()
    
    ax3.plot(milp_results.index, milp_cum, label='MILP Profit', color='black', linewidth=2)
    ax3.plot(rl_results.index, rl_cum, label='RL Hybrid Profit', color='tab:orange', linestyle='--', linewidth=2)
    ax3.set_ylabel('Cumulative Profit (£)', fontweight='bold')
    ax3.set_title('Financial Performance Comparison', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Final touch: Clean up the dates on the bottom
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(days_locator)
        ax.xaxis.set_major_formatter(days_formatter)
    
    plt.xticks(rotation=0) # Keeps day labels horizontal for better readability
    plt.tight_layout()
    plt.show()

    #SUMMARY STATS
    m_profit = milp_results['Hourly_Profit'].sum()
    r_profit = rl_results['Revenue'].sum()
    print("\n" + "="*40)
    print(f"HYBRID PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Expert (MILP) Profit:  £{m_profit:,.2f}")
    print(f"Student (RL) Profit:   £{r_profit:,.2f}")
    print(f"Imitation Accuracy:    {(r_profit/m_profit)*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Load same data used in training
    full_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
    full_df = full_df[['DayAhead']]
    
    params = {
        'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 'rte': 0.9,
        'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'soc_initial_factor': 0.5,
        'deg_per_mwh': 0.00001
    }
    
    run_hybrid_test(full_df, params, "bess_hybrid_imitation.zip")