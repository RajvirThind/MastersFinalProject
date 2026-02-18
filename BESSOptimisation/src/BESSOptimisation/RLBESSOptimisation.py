import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BESSEnv(gym.Env):
    """A Gymnasium environment for BESS RL training."""
    def __init__(self, prices_df, battery_params):
        super(BESSEnv, self).__init__()
        self.prices_df = prices_df
        self.battery_params = battery_params
        
        # Action space: Continuous value from -1 (max charge) to 1 (max discharge)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: [Current SOC, DayAhead Price, Intraday Price, Hour of Day]
        # You can expand this to include all your markets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = self.battery_params.get('soc_initial_factor', 0.5) * self.battery_params['capacity']
        return self._get_obs(), {}

    def _get_obs(self):
        # Fetch current market data
        prices = self.prices_df.iloc[self.current_step]
        hour = self.prices_df.index[self.current_step].hour / 24.0
        return np.array([self.soc, prices['DayAhead'], prices['Intraday'], hour], dtype=np.float32)

    def step(self, action):
        # 1. Scale action (-1 to 1) to MW
        p_max = self.battery_params['max_power']
        power_request = action[0] * p_max
        
        # 2. Physics & Constraints (The logic from your MILP set_constraints)
        dt = self.battery_params['time_interval']
        rte = self.battery_params['rte']
        
        # Calculate energy change with efficiency
        if power_request >= 0: # Discharging
            energy_change = (power_request / rte) * dt
        else: # Charging
            energy_change = (power_request * rte) * dt
            
        # Update SOC and clip to physical limits
        new_soc = np.clip(self.soc - energy_change, 
                          self.battery_params['soc_min_factor'] * self.battery_params['capacity'], 
                          self.battery_params['soc_max_factor'] * self.battery_params['capacity'])
        
        # Calculate actual power delivered (in case SOC hit a limit)
        actual_energy_delta = self.soc - new_soc
        self.soc = new_soc
        
        # 3. Reward Calculation (The logic from your MILP set_objective)
        # Use DayAhead as a simple example; you'd sum across markets in a complex version
        price = self.prices_df.iloc[self.current_step]['DayAhead']
        revenue = actual_energy_delta * price
        
        # Penalty for degradation (using your standard_deg_cost = 6.50)
        deg_penalty = abs(actual_energy_delta) * 6.50
        reward = revenue - deg_penalty

        # 4. Advance time
        self.current_step += 1
        terminated = self.current_step >= len(self.prices_df) - 1
        
        return self._get_obs(), reward, terminated, False, {}