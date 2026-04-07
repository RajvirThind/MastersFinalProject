import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from typing import Callable

# --- LEARNING RATE DECAY SCHEDULE ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 and goes to 0.0
        return progress_remaining * initial_value
    return func


class MultiMarketBESSEnv(gym.Env):
    """
    Custom Reinforcement Learning Environment for Multi-Market BESS Arbitrage
    Features Physical Integrity Guardrails, Anti-Barcoding, and Spike Detection.
    """
    def __init__(self, df, battery_params):
        super(MultiMarketBESSEnv, self).__init__()

        self.df = df
        self.markets = ['DayAhead', 'Intraday', 'BM', 'Imbalance', 'DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh']
        self.num_markets = len(self.markets)
        self.lookahead_steps = 48
        
        self.dt = battery_params.get('time_interval', 0.5)
        self.capacity = battery_params.get('capacity', 20)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        self.rte_sqrt = self.rte ** 0.5
        self.alpha = battery_params.get('utilisation_factor', 0.02)
        
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_markets,), dtype=np.float32)
        
        # INCREASED OBS SIZE: Added 4 slots for Wholesale Volatility/Spike Detectors
        obs_size = 1 + 1 + 1 + 4 + 4 + (self.num_markets * self.lookahead_steps)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.history = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 
        self.soc = self.capacity * 0.5 
        self.locked_anc_mw = np.zeros(4, dtype=np.float32) 
        
        self.prev_direction = 0.0 
        self.prev_arb_power = 0.0 
        
        if hasattr(self, 'history') and len(self.history) > 0:
            self.eval_history = self.history.copy()
        self.history = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        norm_soc = np.array([(self.soc - self.soc_min) / (self.soc_max - self.soc_min)], dtype=np.float32)
        current_time = self.df.index[self.current_step]
        hour_obs = np.array([current_time.hour / 23.0], dtype=np.float32)
        steps_into_efa = self.current_step % 8
        efa_timer = np.array([(8 - steps_into_efa) / 8.0], dtype=np.float32)
        locked_obs = self.locked_anc_mw / self.p_max
        
        end_idx = self.current_step + self.lookahead_steps
        price_arrays = []
        volatilities = [] # The new "Spike Detector" array
        
        for i, market in enumerate(self.markets):
            prices = self.df[market].values
            price_window = prices[self.current_step : end_idx] if end_idx <= len(prices) else np.concatenate([prices[self.current_step:], np.full(self.lookahead_steps - len(prices[self.current_step:]), prices[-1])])
            price_arrays.append(price_window)
            
            # Calculate volatility for the 4 Wholesale markets only
            if i < 4:
                mean_p = np.mean(price_window) + 1e-5 # prevent division by zero
                max_p = np.max(price_window)
                # Ratio of Max to Mean (Clips to 10 to prevent neural net exploding)
                volatilities.append(min(10.0, max(0.0, max_p / mean_p)))
                
        vol_obs = np.array(volatilities, dtype=np.float32)
        flat_prices = np.concatenate(price_arrays).astype(np.float32)
        
        return np.concatenate([norm_soc, hour_obs, efa_timer, locked_obs, vol_obs, flat_prices])

    def step(self, action):
        current_prices = self.df.iloc[self.current_step][self.markets].values

        # EFA Layer
        if self.current_step % 8 == 0:
            raw_ancillary = np.clip(action[4:8], 0.0, 0.5) * self.p_max
            total_anc_req = np.sum(raw_ancillary)
            if total_anc_req > (self.p_max * 0.5):
                raw_ancillary = (raw_ancillary / total_anc_req) * (self.p_max * 0.5)
            self.locked_anc_mw = np.round(raw_ancillary)

        # Arbitrage Layer
        total_locked = np.sum(self.locked_anc_mw)
        remaining_p_max = max(0.0, self.p_max - total_locked) 
        raw_arb = action[0:4]
        
        best_market_idx = np.argmax(np.abs(raw_arb))
        focused_arb = np.zeros(4)

        if np.abs(raw_arb[best_market_idx]) > 0.50:
            focused_arb[best_market_idx] = raw_arb[best_market_idx]
            
        actual_arb_power = focused_arb * remaining_p_max

        # Physics Layer
        net_arb_power = np.sum(actual_arb_power)
        arb_charge = np.abs(net_arb_power) if net_arb_power < 0 else 0.0
        arb_discharge = net_arb_power if net_arb_power > 0 else 0.0
            
        anc_high_charge = np.sum(self.locked_anc_mw[2:4]) * self.alpha 
        anc_low_discharge = np.sum(self.locked_anc_mw[0:2]) * self.alpha 
        
        req_in = (arb_charge + anc_high_charge) * self.rte_sqrt * self.dt
        req_out = (arb_discharge + anc_low_discharge) / self.rte_sqrt * self.dt
        
        avail_en = self.soc - self.soc_min
        if req_out > avail_en:
            scale = avail_en / req_out if req_out > 0 else 0.0
            actual_arb_power *= scale
            req_out = avail_en
            
        avail_sp = self.soc_max - (self.soc - req_out)
        if req_in > avail_sp:
            scale = avail_sp / req_in if req_in > 0 else 0.0
            actual_arb_power *= scale
            req_in = avail_sp
            
        net_arb_power = np.sum(actual_arb_power)
        self.soc = np.clip(self.soc + req_in - req_out, self.soc_min, self.soc_max)
        
        # Revenue & Penalties
        revenue = np.sum(actual_arb_power * current_prices[0:4]) * self.dt
        revenue -= (actual_arb_power[2] * current_prices[2] * self.dt) * 0.8 
        revenue += np.sum(self.locked_anc_mw * current_prices[4:8]) * self.dt
        
        deg_cost = req_out * 6.50
        stress_cost = 12.0 * self.dt if (self.soc < 0.2*self.capacity or self.soc > 0.8*self.capacity) else 0.0

        current_dir = np.sign(net_arb_power)
        reversal_penalty = 0.0
        
        if current_dir != 0: 
            if self.prev_direction != 0 and self.prev_direction != current_dir:
                reversal_penalty = 30.0 
            self.prev_direction = current_dir 

        delta_power = np.abs(net_arb_power - self.prev_arb_power)
        ramp_penalty = delta_power * 1.50 
        self.prev_arb_power = net_arb_power

        reward = revenue - deg_cost - stress_cost - reversal_penalty - ramp_penalty
        
        self.history.append({
            'SOC': self.soc, 'Net_Profit': reward, 'Arb_Power_Used': net_arb_power,
            'Power_DayAhead': actual_arb_power[0], 'Power_Intraday': actual_arb_power[1],
            'Power_BM': actual_arb_power[2], 'Power_Imbalance': actual_arb_power[3],
            'Power_DCDMLow': self.locked_anc_mw[0], 'Power_DRLow': self.locked_anc_mw[1],
            'Power_DCDMHigh': -self.locked_anc_mw[2], 'Power_DRHigh': -self.locked_anc_mw[3]
        })
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}


if __name__ == "__main__":
    MODEL_NAME = "models/bess_multimarket_ppo_v3"
    VEC_NORM_PATH = "models/vec_normalize_multimarket_v3.pkl"
    DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
    
    battery_params = {
        'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 
        'rte': 0.9, 'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 
        'utilisation_factor': 0.02
    }
    
    os.makedirs("models", exist_ok=True)
    
    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    required_cols = ['DayAhead', 'Intraday', 'BM', 'Imbalance', 'DCDMLow', 'DRLow', 'DCDMHigh', 'DRHigh']
    df = df[required_cols] 
    
    # Train on ~1 year of data
    train_df = df.iloc[:17520] 
    
    env = DummyVecEnv([lambda: MultiMarketBESSEnv(train_df, battery_params)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    policy_kwargs = dict(net_arch=[512, 512])
    
    model = PPO("MlpPolicy", env, 
                learning_rate=linear_schedule(1e-4), # Starts high, decays to 0
                ent_coef=0.025,                      # Increased to encourage risk/exploration
                batch_size=256, 
                gamma=0.999, 
                verbose=1, 
                policy_kwargs=policy_kwargs)
    
    print("--- Training Multi-Market Agent (V3) for 1,000,000 steps ---")
    model.learn(total_timesteps=1000000)
    
    model.save(MODEL_NAME)
    env.save(VEC_NORM_PATH)
    print("--- Training Complete & Model Saved ---")