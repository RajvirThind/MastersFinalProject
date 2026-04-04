import time  # New import for timing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback  # Callback import

MODEL_NAME = "bess_dqn_original_competitive"
VEC_NORM_PATH = "vec_normalize_dqn_original.pkl"
TRAIN_STEPS = 500000 
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
env = DummyVecEnv([lambda: DiscreteBESSWrapper(BESSEnv(train_df, battery_params), num_bins=21)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

policy_kwargs = dict(net_arch=[256, 256])

# --- Training with Callback ---
if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"--- Loading model: {MODEL_NAME} ---")
    model = DQN.load(MODEL_NAME, env=env)
    env = VecNormalize.load(VEC_NORM_PATH, env)
else:
    print(f"--- Training model for {TRAIN_STEPS} steps ---")
    model = DQN("MlpPolicy", env, 
                gamma=0.999, 
                exploration_fraction=0.5, 
                exploration_final_eps=0.05, 
                verbose=1, 
                policy_kwargs=policy_kwargs)
    
    # Initialize and apply the timer callback
    timer_callback = TimeTrackingCallback()
    model.learn(total_timesteps=TRAIN_STEPS, callback=timer_callback)
    
    model.save(MODEL_NAME)
    env.save(VEC_NORM_PATH)

# --- Evaluation ---
eval_env = DummyVecEnv([lambda: DiscreteBESSWrapper(BESSEnv(eval_df, battery_params), num_bins=21)])
eval_env = VecNormalize.load(VEC_NORM_PATH, eval_env)
eval_env.training = False 
eval_env.norm_reward = False 

obs = eval_env.reset()

for _ in range(len(eval_df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if done[0]: break

# --- Plotting ---
res_df = pd.DataFrame(eval_env.envs[0].env.eval_history)
res_df.index = eval_df.index[:len(res_df)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
ax1.plot(res_df.index, res_df['SOC'], color='tab:blue')
ax1.set_title("BESS SOC (High-Res DQN Agent)")
ax2.plot(res_df.index, eval_df['DayAhead'].iloc[:len(res_df)], color='black', alpha=0.3)
ax2_twin = ax2.twinx()
ax2_twin.step(res_df.index, res_df['Power_DayAhead'], where='post', color='green', alpha=0.5)

net_profit = res_df['Revenue'] - res_df['Penalty']
ax3.plot(res_df.index, net_profit.cumsum(), color='orange')
ax3.set_ylabel("Cumulative Net Profit (£)")

plt.tight_layout()
plt.show()

print(f"Total Net Profit: £{net_profit.sum():,.2f}")
print(f"Final SOH: {res_df['SOH'].iloc[-1]:.2%}")