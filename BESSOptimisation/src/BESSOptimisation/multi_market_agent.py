import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class MultiMarketBESSEnv(gym.Env):
    """
    BESS Environment that forces an RL Agent to obey strict MILP physics 
    and 4-hour EFA block rules for 8 simultaneous markets.
    """
    def __init__(self, df, battery_params):
        super(MultiMarketBESSEnv, self).__init__()
        
        # --- Market Setup ---
        self.df = df
        self.markets = ['DayAhead', 'Intraday', 'BM', 'Imbalance', 'DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh']
        self.num_markets = len(self.markets)
        self.lookahead_steps = 48
        
        # --- Battery Parameters ---
        self.dt = battery_params.get('time_interval', 0.5)
        self.capacity = battery_params.get('capacity', 20)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        self.rte_sqrt = self.rte ** 0.5
        self.alpha = battery_params.get('utilisation_factor', 0.02)
        
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        
        # --- RL Spaces ---
        # Action: 8 continuous dials [-1.0 to 1.0] (4 Arbitrage, 4 Ancillary)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_markets,), dtype=np.float32)
        
        # Observation: SOC(1) + Hour(1) + EFA_Timer(1) + Locked_Ancillary(4) + 8 Market Prices * 48
        obs_size = 1 + 1 + 1 + 4 + (self.num_markets * self.lookahead_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.history = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 
        self.soc = self.capacity * 0.5 
        
        # EFA Tracking State
        self.locked_anc_mw = np.zeros(4, dtype=np.float32) 
        
        # Preserve history bug fix
        if hasattr(self, 'history') and len(self.history) > 0:
            self.eval_history = self.history.copy()
        self.history = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. State of Charge
        norm_soc = np.array([(self.soc - self.soc_min) / (self.soc_max - self.soc_min)], dtype=np.float32)
        
        # 2. Time of Day
        current_time = self.df.index[self.current_step]
        hour_obs = np.array([current_time.hour / 23.0], dtype=np.float32)
        
        # 3. EFA Block Timer (Countdown from 8 to 1)
        steps_into_efa = self.current_step % 8
        efa_timer = np.array([(8 - steps_into_efa) / 8.0], dtype=np.float32)
        
        # 4. Currently Locked Ancillary Contracts (Normalized to p_max)
        locked_obs = self.locked_anc_mw / self.p_max
        
        # 5. Price Lookahead for ALL 8 markets
        end_idx = self.current_step + self.lookahead_steps
        price_arrays = []
        
        for market in self.markets:
            prices = self.df[market].values
            if end_idx <= len(prices):
                price_window = prices[self.current_step : end_idx]
            else:
                actual_prices = prices[self.current_step:]
                padding = np.full(self.lookahead_steps - len(actual_prices), prices[-1])
                price_window = np.concatenate([actual_prices, padding])
            price_arrays.append(price_window)
            
        flat_prices = np.concatenate(price_arrays).astype(np.float32)
        
        # Stitch it all together
        return np.concatenate([norm_soc, hour_obs, efa_timer, locked_obs, flat_prices])

    def step(self, action):
        current_prices = self.df.iloc[self.current_step][self.markets].values
        
        # ==========================================
        # STEP A: The EFA Translation Layer
        # ==========================================
        # Index 4-7 are Ancillary: [DCDMLow, DRLow, DCDMHigh, DRHigh]
        if self.current_step % 8 == 0:
            # We are at the start of a 4-hour block. Agent can lock in new contracts.
            # Clip between 0 and 1 (cannot have negative availability), scale to p_max, and round to whole MW
            raw_ancillary = np.clip(action[4:8], 0.0, 1.0) * self.p_max
            
            # Prevent the sum of ancillary services from exceeding p_max
            total_anc_req = np.sum(raw_ancillary)
            if total_anc_req > self.p_max:
                raw_ancillary = (raw_ancillary / total_anc_req) * self.p_max
                
            self.locked_anc_mw = np.round(raw_ancillary) # Must be whole numbers for MILP equivalence
        
        # If it's NOT step 0, we ignore action[4:8] and keep the locked MW active.
        
        # ==========================================
        # STEP B: The Arbitrage Hierarchy Layer
        # ==========================================
        # Index 0-3 are Arbitrage: [DayAhead, Intraday, BM, Imbalance]
        total_locked = np.sum(self.locked_anc_mw)
        remaining_p_max = max(0.0, self.p_max - total_locked)
        
        raw_arb = action[0:4]
        total_arb_req = np.sum(np.abs(raw_arb))
        
        # Force the agent's arbitrage requests to fit inside the remaining physical limit
        if total_arb_req > 1.0:
            safe_arb_action = raw_arb / total_arb_req
        else:
            safe_arb_action = raw_arb
            
        actual_arb_power = safe_arb_action * remaining_p_max
        
        # ==========================================
        # STEP C: Physics & Energy Balance
        # ==========================================
        # Arbitrage throughput
        arb_charge = np.sum(np.abs(actual_arb_power[actual_arb_power < 0]))
        arb_discharge = np.sum(actual_arb_power[actual_arb_power > 0])
        
        # Ancillary throughput (Availability * Utilization Alpha)
        anc_high_charge = np.sum(self.locked_anc_mw[2:4]) * self.alpha # DCDMHigh, DRHigh
        anc_low_discharge = np.sum(self.locked_anc_mw[0:2]) * self.alpha # DCDMLow, DRLow
        
        energy_in = (arb_charge + anc_high_charge) * self.rte_sqrt * self.dt
        energy_out = (arb_discharge + anc_low_discharge) / self.rte_sqrt * self.dt
        
        new_soc = np.clip(self.soc + energy_in - energy_out, self.soc_min, self.soc_max)
        actual_energy_out = max(0, self.soc + energy_in - new_soc) # How much actually left the battery
        self.soc = new_soc
        
        # ==========================================
        # STEP D: Revenue & MILP Penalty Logic
        # ==========================================
        revenue = 0.0
        
        # 1. Arbitrage Revenue
        for i in range(4):
            # If actual_arb_power is negative, we are buying (spending money). If positive, selling (making money).
            revenue += actual_arb_power[i] * current_prices[i] * self.dt
            
        # Apply BM Skip Rate penalty (Assuming 80% skip rate for BM)
        # Note: If you have a skip_rate dataframe, pass it in. Here we hardcode 0.8 for safety.
        bm_raw_rev = actual_arb_power[2] * current_prices[2] * self.dt
        revenue -= bm_raw_rev * 0.8 
        
        # 2. Ancillary Revenue (Paid for availability, regardless of throughput)
        for i in range(4):
            revenue += self.locked_anc_mw[i] * current_prices[i+4] * self.dt
            
        # 3. Standard Degradation Cost (£6.50/MWh)
        deg_cost = actual_energy_out * 6.50
        
        # 4. Stress Penalty (£12.00/MWh equivalent)
        stress_cost = 0.0
        if self.soc < (0.20 * self.capacity) or self.soc > (0.80 * self.capacity):
            stress_cost = 12.00 * self.dt
            
        # 5. Intensity Penalty (£15.00/MWh for > 0.5C)
        intensity_cost = 0.0
        total_discharge_power = arb_discharge + anc_low_discharge
        if total_discharge_power > (0.5 * self.capacity):
            intensity_cost = 15.00 * self.dt
            
        reward = revenue - deg_cost - stress_cost - intensity_cost
        
        # Logging
        self.history.append({
            'SOC': self.soc,
            'Total_Revenue': revenue,
            'Total_Penalties': deg_cost + stress_cost + intensity_cost,
            'Net_Profit': reward,
            'Locked_Ancillary_MW': total_locked,
            'Arb_Power_Used': np.sum(actual_arb_power),
            
            # --- NEW: Individual Market Tracking for Visualization ---
            'Power_DayAhead': actual_arb_power[0],
            'Power_Intraday': actual_arb_power[1],
            'Power_BM': actual_arb_power[2],
            'Power_Imbalance': actual_arb_power[3],
            'Power_DCDMLow': self.locked_anc_mw[0],   # Discharge
            'Power_DRLow': self.locked_anc_mw[1],     # Discharge
            'Power_DCDMHigh': -self.locked_anc_mw[2], # Negative because it requires charging
            'Power_DRHigh': -self.locked_anc_mw[3]    # Negative because it requires charging
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, done, False, {}

# ==========================================
# EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    
    MODEL_NAME = "bess_multimarket_ppo"
    VEC_NORM_PATH = "vec_normalize_multimarket.pkl"
    TRAIN_STEPS = 1000000 # Increased because of the massive observation space
    DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
    
    battery_params = {
        'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 
        'rte': 0.9, 'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'utilisation_factor': 0.02
    }
    
    # --- Data Loading ---
    # Ensure your CSV actually has all 8 of these exact column headers!
    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    required_cols = ['DayAhead', 'Intraday', 'BM', 'Imbalance', 'DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh']
    
    # If you are missing columns in your real data, this will throw an error. 
    # Make sure your dataset is perfectly aligned.
    df = df[required_cols] 
    
    train_df = df.iloc[:8760] 
    eval_df = df.iloc[8760:11640]
    
    # --- Setup ---
    env = DummyVecEnv([lambda: MultiMarketBESSEnv(train_df, battery_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # WIDENED NEURAL NETWORK to handle 391 inputs
    policy_kwargs = dict(net_arch=[512, 512])
    
    # --- Training ---
    if os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"--- Loading multi-market model ---")
        model = PPO.load(MODEL_NAME, env=env)
        env = VecNormalize.load(VEC_NORM_PATH, env)
    else:
        print(f"--- Training multi-market model for {TRAIN_STEPS} steps ---")
        model = PPO("MlpPolicy", env, gamma=0.999, verbose=1, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=TRAIN_STEPS)
        model.save(MODEL_NAME)
        env.save(VEC_NORM_PATH) 

    # --- Evaluation ---
    eval_env = DummyVecEnv([lambda: MultiMarketBESSEnv(eval_df, battery_params)])
    eval_env = VecNormalize.load(VEC_NORM_PATH, eval_env)
    eval_env.training = False 
    eval_env.norm_reward = False 

    obs = eval_env.reset()
    for _ in range(len(eval_df) - 2): # Preventing the wipe bug
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done[0]: break
        
    # --- Quick Plot ---
    res_df = pd.DataFrame(eval_env.envs[0].history)
    res_df.index = eval_df.index[:len(res_df)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(res_df.index, res_df['SOC'], color='tab:blue')
    ax1.set_title("Multi-Market Co-Optimization: State of Charge")
    
    ax2.plot(res_df.index, res_df['Net_Profit'].cumsum(), color='green')
    ax2.set_title("Cumulative Net Profit across 8 Markets (£)")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total Multi-Market Net Profit: £{res_df['Net_Profit'].sum():,.2f}")