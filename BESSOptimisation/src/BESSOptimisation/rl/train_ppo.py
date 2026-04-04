import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from BESSOptimisation.environments.dayahead_bess_env import BESSEnv

# --- Custom Callback for Timing ---
class TimeTrackingCallback(BaseCallback):
    """
    Custom callback for reporting the total training time.
    """
    def __init__(self, verbose=0):
        super(TimeTrackingCallback, self).__init__(verbose)
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("--- Training Started ---")

    def _on_training_end(self) -> None:
        total_duration = time.time() - self.start_time
        print(f"--- Training Finished ---")
        print(f"Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

    def _on_step(self) -> bool:
        return True

# --- Configuration & Data ---
MODEL_NAME = "../models/bess_ppo_lookahead_normalized"
VEC_NORM_PATH = "../models/vec_normalize_ppo.pkl"
TRAIN_STEPS = 500000
DATA_PATH = '../../data/GBCentralAllComplete_Prices.csv'

battery_params = {
    'time_interval': 0.5, 
    'max_power': 10, 
    'capacity': 20, 
    'rte': 0.9, 
    'soc_min_factor': 0.1, 
    'soc_max_factor': 0.9,
    'deg_per_mwh': 0.00001
}

prices_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
prices_df = prices_df[['DayAhead']]
train_df = prices_df.iloc[:8760] 
eval_df = prices_df.iloc[8760:11640] 

# --- Environment Setup ---
env = DummyVecEnv([lambda: BESSEnv(train_df, battery_params)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

policy_kwargs = dict(net_arch=[256, 256])

# --- Training with Callback ---
if os.path.exists(f"{MODEL_NAME}.zip"):
    print(f"--- Loading model: {MODEL_NAME} ---")
    model = PPO.load(MODEL_NAME, env=env)
    env = VecNormalize.load(VEC_NORM_PATH, env)
else:
    print(f"--- Training model for {TRAIN_STEPS} steps ---")
    model = PPO("MlpPolicy", env, 
                gamma=0.999,
                ent_coef=0.01,
                verbose=1, 
                policy_kwargs=policy_kwargs)
    
    # Initialize the callback
    timer_callback = TimeTrackingCallback()
    
    # Pass the callback to the learn method
    model.learn(total_timesteps=TRAIN_STEPS, callback=timer_callback)
    
    model.save(MODEL_NAME)
    env.save(VEC_NORM_PATH)

# --- Evaluation (Remains the same) ---
eval_env = DummyVecEnv([lambda: BESSEnv(eval_df, battery_params)])
eval_env = VecNormalize.load(VEC_NORM_PATH, eval_env)
eval_env.training = False 
eval_env.norm_reward = False 

obs = eval_env.reset()

for _ in range(len(eval_df) - 2):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    if done[0]: break

# Plotting and Results
res_df = pd.DataFrame(eval_env.envs[0].history)
res_df.index = eval_df.index[:len(res_df)]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
ax1.plot(res_df.index, res_df['SOC'], color='tab:blue')
ax1.set_title("BESS SOC (Lookahead PPO Agent - Normalized)")
ax2.plot(res_df.index, eval_df['DayAhead'].iloc[:len(res_df)], color='black', alpha=0.3)
ax3.plot(res_df.index, res_df['Revenue'].cumsum(), color='orange')
ax3.set_ylabel("Cumulative Profit (£)")

plt.tight_layout()
plt.show()

print(f"Total Net Profit: £{res_df['Revenue'].sum() - res_df['Penalty'].sum():,.2f}")
print(f"Final SOH: {res_df['SOH'].iloc[-1]:.2%}")