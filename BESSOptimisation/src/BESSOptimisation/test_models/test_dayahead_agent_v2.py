import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO

# Import your custom environment
from BESSOptimisation.src.BESSOptimisation.environments.dayahead_bess_env import BESS_RLEnv

# --- CONFIGURATION ---
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
MODEL_PATH = "bess_ppo_agent.zip"
TEST_DAYS = 7
TRAIN_DAYS = 180  # To ensure we test on UNSEEN data

params = {
    'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 'rte': 0.9,
    'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'soc_initial_factor': 0.5,
    'cycle_limit': 1.1 
}

def run_test():
    print("--- 1. Loading Test Data ---")
    full_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    full_df = full_df[['DayAhead']]
    
    # Slice the dataframe to start exactly where training ended
    start_idx = TRAIN_DAYS * 48
    end_idx = start_idx + (TEST_DAYS * 48)
    test_df = full_df.iloc[start_idx:end_idx].copy()
    
    print(f"Testing from {test_df.index[0]} to {test_df.index[-1]}")

    print("--- 2. Setting up Environment & Model ---")
    # CRITICAL: is_training=False so the environment records the history!
    test_env = BESS_RLEnv(test_df, params, is_training=False)
    
    try:
        model = PPO.load(MODEL_PATH, env=test_env)
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Did you run the training script?")
        return

    print("--- 3. Running Inference ---")
    obs, _ = test_env.reset()
    done = False
    
    while not done:
        # deterministic=True forces the agent to use its best learned strategy
        # without adding random exploration noise
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    print("--- 4. Generating Dashboard ---")
    # Convert the environment's internal history log into a Pandas DataFrame
    results_df = pd.DataFrame(test_env.history)
    results_df.index = test_df.index[:len(results_df)] # Align timestamps

    plot_dashboard(results_df)
    print_summary(results_df)

def plot_dashboard(df):
    """Generates a 3-panel visualization of the agent's behavior."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Common X-axis formatting
    days_locator = mdates.DayLocator()
    days_formatter = mdates.DateFormatter('%a %d %b')

    # --- Plot 1: State of Charge (SOC) ---
    ax1.plot(df.index, df['SOC'], color='tab:blue', linewidth=2.5, label='SOC (MWh)')
    ax1.axhline(2.0, color='red', linestyle='--', alpha=0.3, label='Min SOC')
    ax1.axhline(18.0, color='green', linestyle='--', alpha=0.3, label='Max SOC')
    ax1.set_ylabel('SOC (MWh)', fontweight='bold')
    ax1.set_title('Battery State of Charge', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 2: Price and Trading Action (The "Stock Market" View) ---
    ax2.plot(df.index, df['Price'], color='black', alpha=0.7, linewidth=1.5, label='Day Ahead Price')
    ax2.set_ylabel('Price (£/MWh)', fontweight='bold')
    
    # Track if we've added the label to the legend yet
    bought_label_added = False
    sold_label_added = False
    
    # Overlay Buy/Sell markers based on the Agent's power dispatch
    for idx, row in df.iterrows():
        power = row['Power']
        if power < -1.0: # Charging (Buying Energy)
            ax2.plot(idx, row['Price'] - 5, 'g^', markersize=8, 
                     label='Charge (Buy)' if not bought_label_added else "")
            bought_label_added = True
        elif power > 1.0: # Discharging (Selling Energy)
            ax2.plot(idx, row['Price'] + 5, 'rv', markersize=8, 
                     label='Discharge (Sell)' if not sold_label_added else "")
            sold_label_added = True

    # Add a secondary axis just to show the power envelope lightly in the background
    ax2_power = ax2.twinx()
    ax2_power.fill_between(df.index, 0, df['Power'], step='post', color='gray', alpha=0.15)
    ax2_power.set_ylabel('Power (MW)', color='gray')
    ax2_power.set_ylim(-15, 15) # Keep it scaled neatly
    
    ax2.set_title('Market Price vs. Agent Dispatch Timing', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # --- Plot 3: Cumulative Financials ---
    cum_profit = df['Hourly_Profit'].cumsum()
    ax3.plot(df.index, cum_profit, color='tab:orange', linewidth=2.5, label='Cumulative Arbitrage Profit')
    ax3.set_ylabel('Profit (£)', fontweight='bold')
    ax3.set_title('Financial Performance', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # Apply date formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(days_locator)
        ax.xaxis.set_major_formatter(days_formatter)
    
    plt.tight_layout()
    plt.show()

def print_summary(df):
    total_revenue = df['Hourly_Profit'].sum()
    total_throughput = df['Throughput'].sum()
    total_deg_cost = total_throughput * 35.00 # Matching the environment logic
    net_profit = total_revenue - total_deg_cost
    
    print("\n" + "="*40)
    print("PPO AGENT PERFORMANCE SUMMARY (7 DAYS)")
    print("="*40)
    print(f"Gross Arbitrage Revenue:  £{total_revenue:,.2f}")
    print(f"Total Throughput:         {total_throughput:,.2f} MWh")
    print(f"Estimated Deg Costs:      £{total_deg_cost:,.2f} (@ £35/MWh)")
    print("-" * 40)
    print(f"NET PROFIT:               £{net_profit:,.2f}")
    print("="*40)

if __name__ == "__main__":
    run_test()        