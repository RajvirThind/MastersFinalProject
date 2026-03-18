import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BESSEnv(gym.Env):
    """Custom Environment for BESS Day Ahead Arbitrage with Lookahead."""
    def __init__(self, df, battery_params):
        super(BESSEnv, self).__init__()
        self.df = df
        self.lookahead_steps = 48  # 24 hours of foresight

        # battery physical parameters
        self.dt = battery_params.get('time_interval', 0.5)
        self.capacity = battery_params.get('capacity', 20)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        self.deg_per_mwh = battery_params.get('deg_per_mwh', 0.00001)
        
        # SOC Limits
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        
        # Action Space: [-1, 1] mapped to [Full Charge, Full Discharge]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: SOC + 48 Price points (Current + 47 Future)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + self.lookahead_steps,), dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0 #set the clock back to the start
        self.soc = self.capacity * 0.5 #put battery back to 50% SOC
        self.soh = 1.0  # Reset Health
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Get the current observation from the environment.
        """
        # current SOC
        soc_obs = np.array([self.soc], dtype=np.float32)
        
        # slicing the price list from now to 48 steps into the future
        prices = self.df['DayAhead'].values
        end_idx = self.current_step + self.lookahead_steps
        
        if end_idx <= len(prices):
            price_window = prices[self.current_step : end_idx]
        else:
            # Pad with the last available price if near the end so we have a full window
            actual_prices = prices[self.current_step:]
            padding = np.full(self.lookahead_steps - len(actual_prices), prices[-1])
            price_window = np.concatenate([actual_prices, padding])
            
        return np.concatenate([soc_obs, price_window]).astype(np.float32)

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        price = self.df.iloc[self.current_step]['DayAhead']
        
        # Map action to power
        power = action[0] * self.p_max
        
        # Calculate Energy Delta with RTE
        if power < 0: # Charging
            energy_delta = -power * self.rte * self.dt
        else: # Discharging
            energy_delta = - (power / self.rte) * self.dt
            
        # Update SOC and clamp
        new_soc = np.clip(self.soc + energy_delta, self.soc_min, self.soc_max)
        actual_energy_delta = new_soc - self.soc
        
        # Back-calculate actual power delivered
        if actual_energy_delta > 0: # Actual Charge
            actual_power = -(actual_energy_delta / (self.rte * self.dt))
        else: # Actual Discharge
            actual_power = -(actual_energy_delta * self.rte / self.dt)

        self.soc = new_soc
        
        # Economics (Matches MILP £6.50/MWh penalty)
        revenue = actual_power * price * self.dt
        throughput = abs(actual_energy_delta) if actual_energy_delta < 0 else 0
        penalty = throughput * 6.50
        reward = revenue - penalty
        
        # Update SOH
        self.soh -= throughput * self.deg_per_mwh
        
        # Log History
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