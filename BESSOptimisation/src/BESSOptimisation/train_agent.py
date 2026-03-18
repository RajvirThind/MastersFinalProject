import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from RLBESSOptimisation import BESSEnv


MODEL_NAME = "bess_sac_optimized"
TRAIN_STEPS = 150000
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'

battery_params = {
    'time_interval': 0.5, 
    'max_power': 10, 
    'capacity': 20, 
    'rte': 0.9, 
    'soc_initial_factor': 0.5, 
    'soc_min_factor': 0.1, 
    'soc_max_factor': 0.9,
    'utilisation_factor': 0.02,
    'deg_per_mwh': 0.00001
}

# --- 2. DATA LOADING & PRE-PROCESSING ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Ensure your price data is at {DATA_PATH}")

prices_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)

# Splitting data: Train on first 6 months, Eval on next 2 months
train_df = prices_df.iloc[:8760] 
eval_df = prices_df.iloc[8760:11640] 

# --- 3. TRAINING ---
env = BESSEnv(train_df, battery_params)

# Policy architecture: 128x128 neurons to capture market correlations
policy_kwargs = dict(net_arch=[128, 128])

if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"--- Loading existing model: {MODEL_NAME} ---")
    model = SAC.load(MODEL_NAME, env=env)
else:
    print(f"--- Starting fresh training for {TRAIN_STEPS} steps ---")
    model = SAC("MlpPolicy", env, 
                verbose=1, 
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                ent_coef='0.5', 
                batch_size=256, 
                gamma=0.99) 
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_NAME)

# --- 4. EVALUATION ---
print("--- Running Evaluation on Unseen Data ---")
eval_env = BESSEnv(eval_df, battery_params)
obs, _ = eval_env.reset()

for _ in range(len(eval_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    if done: break

# --- 5. RESULTS & PLOTTING ---
res_df = pd.DataFrame(eval_env.history)
res_df.index = eval_df.index[:len(res_df)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot 1: SOC and Physical Constraints
ax1.plot(res_df.index, res_df['SOC'], label='Agent SOC (MWh)', color='tab:blue', linewidth=2)
ax1.axhline(eval_env.soc_min, color='red', linestyle='--', alpha=0.5, label='Min SOC')
ax1.axhline(eval_env.soc_max, color='green', linestyle='--', alpha=0.5, label='Max SOC')
ax1.set_ylabel('MWh')
ax1.set_title("BESS Operational State (RL Agent)")
ax1.legend(loc='upper right')

# Plot 2: Market Dispatch (Sample of Arbitrage vs Ancillary)
ax2.plot(res_df.index, res_df['Power_DayAhead'], label='DayAhead MW', alpha=0.6)
ax2.plot(res_df.index, res_df['Power_DCDMLow'], label='DC Low MW', alpha=0.6)
ax2.plot(res_df.index, res_df['Power_Net'], color='black', label='Net Physical Flow', linewidth=1, linestyle='--')
ax2.set_ylabel('MW')
ax2.set_title("Market Power Allocation & Stacking")
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Plot 3: Financials & Penalty Tracking
ax3.plot(res_df.index, res_df['Revenue'].cumsum(), color='green', label='Cumulative Revenue (£)')
ax3_p = ax3.twinx()
ax3_p.plot(res_df.index, res_df['Penalty'].cumsum(), color='red', label='Cumulative Penalties (£)', linestyle=':')
ax3.set_ylabel('Revenue (£)', color='green')
ax3_p.set_ylabel('Penalties (£)', color='red')
ax3.set_title("Financial Performance & Contractual Compliance")
ax3.legend(loc='upper left')
ax3_p.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- 6. FINAL METRICS ---
final_net_profit = res_df['Revenue'].sum() - res_df['Penalty'].sum()
print("\n" + "="*40)
print(f"RL AGENT PERFORMANCE SUMMARY")
print("="*40)
print(f"Total Revenue:      £{res_df['Revenue'].sum():,.2f}")
print(f"Total Penalties:    £{res_df['Penalty'].sum():,.2f}")
print(f"Net Profit:         £{final_net_profit:,.2f}")
print(f"Avg Daily Cycles:   {(res_df['Throughput_MWh'].sum() / eval_env.capacity) / (len(res_df)*0.5/24):.2f}")
print("="*40)