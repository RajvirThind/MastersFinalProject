import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from BESSOptimisation.src.BESSOptimisation.dayahead_agent import BESSEnv

MODEL_NAME = "bess_sac_lookahead"
TRAIN_STEPS = 250000 # Increased to account for larger observation space
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'

battery_params = {
    'time_interval': 0.5, 
    'max_power': 10, 
    'capacity': 20, 
    'rte': 0.9, 
    'soc_min_factor': 0.1, 
    'soc_max_factor': 0.9,
    'deg_per_mwh': 0.00001
}

# --- Load Data ---
prices_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
prices_df = prices_df[['DayAhead']]

train_df = prices_df.iloc[:8760] 
eval_df = prices_df.iloc[8760:11640] 

# --- Environment Setup ---
env = BESSEnv(train_df, battery_params)

# Policy architecture (Increased width for lookahead data)
policy_kwargs = dict(net_arch=[256, 256])

# --- Training ---
if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"--- Loading model: {MODEL_NAME} ---")
    model = SAC.load(MODEL_NAME, env=env)
else:
    print(f"--- Training model for {TRAIN_STEPS} steps ---")
    model = SAC("MlpPolicy", env, 
                verbose=1, 
                policy_kwargs=policy_kwargs,
                ent_coef='auto')
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(MODEL_NAME)

# --- Evaluation ---
eval_env = BESSEnv(eval_df, battery_params)
obs, _ = eval_env.reset()

for _ in range(len(eval_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = eval_env.step(action)
    if done: break

# --- Plotting ---
res_df = pd.DataFrame(eval_env.history)
res_df.index = eval_df.index[:len(res_df)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
ax1.plot(res_df.index, res_df['SOC'], color='tab:blue')
ax1.set_title("BESS SOC (Lookahead RL Agent)")
ax2.plot(res_df.index, eval_df['DayAhead'].iloc[:len(res_df)], color='black', alpha=0.3)
ax2_twin = ax2.twinx()
ax2_twin.bar(res_df.index, res_df['Power_DayAhead'], width=0.01, color='green', alpha=0.5)
ax3.plot(res_df.index, res_df['Revenue'].cumsum(), color='orange')
ax3.set_ylabel("Cumulative Profit (£)")

plt.tight_layout()
plt.show()

print(f"Total Net Profit: £{res_df['Revenue'].sum() - res_df['Penalty'].sum():,.2f}")
print(f"Final SOH: {res_df['SOH'].iloc[-1]:.2%}")