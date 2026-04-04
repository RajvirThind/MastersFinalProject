import gymnasium as gym
from gymnasium import spaces
import numpy as np

# class BESSEnv(gym.Env):
#     """Custom Environment for BESS Day Ahead Arbitrage with Lookahead."""
#     def __init__(self, df, battery_params):
#         super(BESSEnv, self).__init__()
#         self.df = df
#         self.lookahead_steps = 48  

#         # battery physical parameters
#         self.dt = battery_params.get('time_interval', 0.5)
#         self.capacity = battery_params.get('capacity', 20)
#         self.p_max = battery_params.get('max_power', 10)
#         self.rte = battery_params.get('rte', 0.9)
#         self.deg_per_mwh = battery_params.get('deg_per_mwh', 0.00001)
        
#         # SOC Limits
#         self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
#         self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        
#         self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
#         # Observation Space: SOC (1) + Time (1) + 48 Price points = 50
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(2 + self.lookahead_steps,), dtype=np.float32
#         )
        
#         self.reset()

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_step = 0 
#         self.soc = self.capacity * 0.5 
#         self.soh = 1.0  
#         self.history = []
#         return self._get_obs(), {}

#     def _get_obs(self):
#         # 1. Normalize SOC between 0 and 1
#         normalized_soc = (self.soc - self.soc_min) / (self.soc_max - self.soc_min)
#         soc_obs = np.array([normalized_soc], dtype=np.float32)
        
#         # 2. Add Temporal Context (Normalized Hour of the Day)
#         # Assumes self.df.index is a DatetimeIndex
#         current_time = self.df.index[self.current_step]
#         hour_obs = np.array([current_time.hour / 23.0], dtype=np.float32)
        
#         # 3. Handle Prices
#         prices = self.df['DayAhead'].values
#         end_idx = self.current_step + self.lookahead_steps
        
#         if end_idx <= len(prices):
#             price_window = prices[self.current_step : end_idx]
#         else:
#             actual_prices = prices[self.current_step:]
#             padding = np.full(self.lookahead_steps - len(actual_prices), prices[-1])
#             price_window = np.concatenate([actual_prices, padding])
            
#         return np.concatenate([soc_obs, hour_obs, price_window]).astype(np.float32)

#     def step(self, action):
#         price = self.df.iloc[self.current_step]['DayAhead']
#         power = action[0] * self.p_max
        
#         if power < 0: 
#             energy_delta = -power * self.rte * self.dt
#         else: 
#             energy_delta = - (power / self.rte) * self.dt
            
#         new_soc = np.clip(self.soc + energy_delta, self.soc_min, self.soc_max)
#         actual_energy_delta = new_soc - self.soc
        
#         if actual_energy_delta > 0: 
#             actual_power = -(actual_energy_delta / (self.rte * self.dt))
#         else: 
#             actual_power = -(actual_energy_delta * self.rte / self.dt)

#         self.soc = new_soc
        
#         revenue = actual_power * price * self.dt
#         throughput = abs(actual_energy_delta) if actual_energy_delta < 0 else 0
#         penalty = throughput * 6.50
#         reward = revenue - penalty
        
#         self.soh -= throughput * self.deg_per_mwh
        
#         self.history.append({
#             'SOC': self.soc,
#             'Power_DayAhead': actual_power,
#             'Revenue': revenue,
#             'Penalty': penalty,
#             'Throughput': throughput,
#             'SOH': self.soh
#         })
        
#         self.current_step += 1
#         done = self.current_step >= len(self.df) - 1
        
#         return self._get_obs(), reward, done, False, {}
    
class DiscreteBESSWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins=21):
        super().__init__(env)
        self.action_space = spaces.Discrete(num_bins)
        self.action_map = np.linspace(-1.0, 1.0, num_bins)

    def action(self, act):
        return np.array([self.action_map[act]], dtype=np.float32)

class BESSEnv(gym.Env):
    def __init__(self, df, battery_params):
        super(BESSEnv, self).__init__()
        self.df = df
        self.lookahead_steps = 48  
        self.dt = battery_params.get('time_interval', 0.5)
        self.capacity = battery_params.get('capacity', 20)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        self.deg_per_mwh = battery_params.get('deg_per_mwh', 0.00001)
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + self.lookahead_steps,), dtype=np.float32
        )
        self.history = []
        self.eval_history = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = self.capacity * 0.5
        self.soh = 1.0
        if hasattr(self, 'history') and len(self.history) > 0:
            self.eval_history = self.history.copy()
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        normalized_soc = (self.soc - self.soc_min) / (self.soc_max - self.soc_min)
        soc_obs = np.array([normalized_soc], dtype=np.float32)
        current_time = self.df.index[self.current_step]
        hour_obs = np.array([current_time.hour / 23.0], dtype=np.float32)
        prices = self.df['DayAhead'].values
        end_idx = self.current_step + self.lookahead_steps
        if end_idx <= len(prices):
            price_window = prices[self.current_step : end_idx]
        else:
            actual_prices = prices[self.current_step:]
            padding = np.full(self.lookahead_steps - len(actual_prices), prices[-1])
            price_window = np.concatenate([actual_prices, padding])
        return np.concatenate([soc_obs, hour_obs, price_window]).astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.current_step]['DayAhead']
        power = action[0] * self.p_max
        if power < 0:
            energy_delta = -power * self.rte * self.dt
        else: 
            energy_delta = - (power / self.rte) * self.dt
        new_soc = np.clip(self.soc + energy_delta, self.soc_min, self.soc_max)
        actual_energy_delta = new_soc - self.soc
        if actual_energy_delta > 0: 
            actual_power = -(actual_energy_delta / (self.rte * self.dt))
        else: 
            actual_power = -(actual_energy_delta * self.rte / self.dt)
        self.soc = new_soc
        revenue = actual_power * price * self.dt
        throughput = abs(actual_energy_delta) if actual_energy_delta < 0 else 0
        penalty = throughput * 6.50
        reward = revenue - penalty
        self.soh -= throughput * self.deg_per_mwh
        self.history.append({
            'SOC': self.soc,
            'Power_DayAhead': actual_power,
            'Revenue': revenue,
            'Penalty': penalty,
            'Throughput': throughput,
            'SOH': self.soh
        })
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}