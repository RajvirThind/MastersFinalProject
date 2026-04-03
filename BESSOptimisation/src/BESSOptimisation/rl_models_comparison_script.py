import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from dayahead_agent_v2 import BESSEnv

# --- Timing Callback ---
class ComparisonCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ComparisonCallback, self).__init__(verbose)
        self.start_time = None
        self.total_time = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_training_end(self) -> None:
        self.total_time = time.time() - self.start_time

    def _on_step(self) -> bool:
        return True

# --- Environment Wrapper for DQN ---
class DiscreteBESSWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins=21):
        super().__init__(env)
        self.action_space = spaces.Discrete(num_bins)
        self.action_map = np.linspace(-1.0, 1.0, num_bins)
    def action(self, act):
        return np.array([self.action_map[act]], dtype=np.float32)

# --- Configuration ---
TRAIN_STEPS = 500000
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
battery_params = {
    'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 
    'rte': 0.9, 'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'deg_per_mwh': 0.00001
}

# --- Load Data ---
prices_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)[['DayAhead']]
train_df = prices_df.iloc[:8760]
eval_df = prices_df.iloc[8760:11640]

results = {}

def run_experiment(algo_name, model_class, is_discrete=False):
    print(f"\n>>> Starting Experiment: {algo_name}")
    
    # Setup Env
    def make_env():
        base_env = BESSEnv(train_df, battery_params)
        return DiscreteBESSWrapper(base_env) if is_discrete else base_env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Initialize Model
    if algo_name == "PPO":
        model = model_class("MlpPolicy", env, verbose=0, gamma=0.999, ent_coef=0.01)
    elif algo_name == "SAC":
        model = model_class("MlpPolicy", env, verbose=0, gamma=0.999, ent_coef='auto')
    else: # DQN
        model = model_class("MlpPolicy", env, verbose=0, gamma=0.999, exploration_fraction=0.5)
    
    # Train and Measure Training Speed
    callback = ComparisonCallback()
    model.learn(total_timesteps=TRAIN_STEPS, callback=callback)
    train_time = callback.total_time
    
    # Evaluate and Measure Execution Speed
    eval_env_base = BESSEnv(eval_df, battery_params)
    eval_env = DummyVecEnv([lambda: DiscreteBESSWrapper(eval_env_base) if is_discrete else eval_env_base])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True) # Stats shouldn't update
    eval_env.training = False
    eval_env.norm_reward = False
    
    obs = eval_env.reset()
    start_exec = time.time()
    for _ in range(len(eval_df) - 2):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env.step(action)
        if done[0]: break
    exec_time = time.time() - start_exec
    
    # Metrics Extraction
    history = pd.DataFrame(eval_env.envs[0].env.history if is_discrete else eval_env.envs[0].history)
    total_revenue = history['Revenue'].sum() - history['Penalty'].sum()
    # Cycles approx: (Total Throughput / (2 * usable capacity))
    usable_cap = battery_params['capacity'] * (battery_params['soc_max_factor'] - battery_params['soc_min_factor'])
    cycles = history['Throughput'].sum() / (2 * usable_cap)
    
    results[algo_name] = {
        "Train Time (s)": train_time,
        "Inference Time (s)": exec_time,
        "Net Profit (£)": total_revenue,
        "Cycles": cycles
    }

# --- Run All ---
run_experiment("PPO", PPO, is_discrete=False)
run_experiment("SAC", SAC, is_discrete=False)
run_experiment("DQN", DQN, is_discrete=True)

# --- Comparison Table ---
comparison_df = pd.DataFrame(results).T
print("\n--- Final Comparison ---")
print(comparison_df)

# --- Visualization ---
comparison_df.plot(kind='bar', subplots=True, layout=(2,2), figsize=(12,10), title="BESS Algorithm Comparison")
plt.tight_layout()
plt.show()