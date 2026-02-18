import pandas as pd
from stable_baselines3 import PPO
from RLBESSOptimisation import BESSEnv


prices_df = pd.read_csv('data/GBCentralAllComplete_Prices.csv', index_col='Date', parse_dates=True)
prices_df = prices_df.head(2880)

battery_params = {
        'time_interval': 0.5, 
        'max_power': 10, 
        'capacity': 20, 
        'rte': 0.9, 
        'soc_min_factor': 0.1, 
        'soc_max_factor': 0.9,
        'soc_initial_factor': 0.5,
        'cycle_limit': 1.1, 
        'deg_per_mwh': 0.00001, 
        'utilisation_factor': 0.02
    }

env = BESSEnv(prices_df, battery_params)

# 2. Initialize the RL Agent
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 3. Train the model (this replaces the pulp.solve step)
model.learn(total_timesteps=100000)

# 4. Use the model to predict
obs, _ = env.reset()
for _ in range(len(prices_df)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break