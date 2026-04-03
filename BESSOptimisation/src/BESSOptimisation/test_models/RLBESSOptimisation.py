import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BESSEnv(gym.Env):
    def __init__(self, prices_df, battery_params):
        super(BESSEnv, self).__init__()
        self.prices_df = prices_df
        self.battery_params = battery_params
        
        # Physical Parameters (Mirroring MILP)
        self.capacity = battery_params.get('capacity', 20)
        self.p_max = battery_params.get('max_power', 10)
        self.dt = battery_params.get('time_interval', 0.5)
        self.rte = battery_params.get('rte', 0.9)
        self.alpha = battery_params.get('utilisation_factor', 0.02)
        
        # SOC Limits
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        self.safe_lower = 0.20 * self.capacity 
        self.safe_upper = 0.80 * self.capacity

        # Market Configuration
        self.market_cols = [
            'DayAhead', 'Intraday', 'BM', 'Imbalance', 
            'DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh'
        ]
        
        # Action Space: 4 Arb (-1 to 1), 4 Ancillary (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1]*4 + [0]*4), high=np.array([1]*8), dtype=np.float32)
        
        # Observation Space: 8 Prices + SOC + Block Progress
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
        self.steps_per_efa_block = 8 # 4 hours / 0.5 step
        self.history = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_block_step = 0
        self.soc = self.battery_params.get('soc_initial_factor', 0.5) * self.capacity
        self.locked_ancillary_action = np.zeros(4)
        self.history = []
        return self._get_obs(), {}

    def _get_obs(self):
        prices = self.prices_df.iloc[self.current_step][self.market_cols].values.astype(np.float32) / 200.0
        soc_norm = (self.soc - self.soc_min) / (self.soc_max - self.soc_min)
        block_norm = self.current_block_step / self.steps_per_efa_block
        return np.concatenate([prices, [soc_norm, block_norm]]).astype(np.float32)

    def step(self, action):
        # 1. EFA Block Lock
        if self.current_block_step == 0:
            self.locked_ancillary_action = action[4:]
        else:
            action[4:] = self.locked_ancillary_action
        self.current_block_step = (self.current_block_step + 1) % 8

        # 2. Market Mapping
        p_arb = action[0:4] * self.p_max
        anc_low_mw = np.floor(action[4:6] * self.p_max)
        anc_high_mw = np.floor(action[6:8] * self.p_max)

        # 3. Hard Physical Clipping (10MW Limit)
        net_p_arb = p_arb.sum()
        net_p_anc = (self.alpha * anc_low_mw.sum()) - (self.alpha * anc_high_mw.sum())
        
        actual_phys_p = np.clip(net_p_arb + net_p_anc, -self.p_max, self.p_max)

        # 4. SOC Update
        old_soc = self.soc
        # Physics: Efficiency only applies to the actual energy moved
        energy_move = (actual_phys_p / self.rte if actual_phys_p >= 0 else actual_phys_p * self.rte) * self.dt
        self.soc = np.clip(self.soc - energy_move, self.soc_min, self.soc_max)

        # 5. Financial Reward
        prices = self.prices_df.iloc[self.current_step]
        
        # Revenue from all 8 markets
        rev_arb = np.sum(p_arb * prices[self.market_cols[:4]]) * self.dt
        rev_anc = np.sum(np.append(anc_low_mw, anc_high_mw) * prices[self.market_cols[4:]]) * self.dt
        
        # 6. Inventory Incentive (The "Cycling" Engine)
        # We value the energy change at the current DayAhead price
        # This rewards charging when prices are low and discharging when high
        inventory_delta = (self.soc - old_soc) * prices['DayAhead']
        
        # 7. Simplified Reward (No Degradation)
        # We also scale it down so the SAC algorithm doesn't get overwhelmed
        reward = (rev_arb + rev_anc + inventory_delta) / 100.0

        # Penalize hard if the agent tries to 'over-bid' (Soft constraint)
        total_req_ds = np.maximum(0, p_arb).sum() + anc_low_mw.sum()
        total_req_ch = np.abs(np.minimum(0, p_arb)).sum() + anc_high_mw.sum()
        if total_req_ds > self.p_max or total_req_ch > self.p_max:
            reward -= 10.0 

        # Log and Advance
        self.history.append({
            'SOC': self.soc, 
            'Power_Net': actual_phys_p, 
            'Revenue': rev_arb + rev_anc,
            'Price': prices['DayAhead']
        })

        self.current_step += 1
        done = self.current_step >= len(self.prices_df) - 1
        return self._get_obs(), float(reward), done, False, {}
    
    #agent looks at _get_obs()
    #agent chooses an action based on the observation
    #environment runs step(), calculates energy, checks if we are in EFA block, calculates the reward
    #agent received the new SOC and the reward and learns that charging when cheap = high reward

    