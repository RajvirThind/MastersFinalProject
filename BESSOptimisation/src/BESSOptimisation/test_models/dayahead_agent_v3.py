import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

class BESSEnv(gym.Env):
    """Custom Environment for BESS Day-Ahead Arbitrage with 24-hour lookahead."""
    metadata = {"render_modes": ["console"]}

    def __init__(self, df_prices, battery_params):
        super(BESSEnv, self).__init__()
        
        self.df_prices = df_prices.reset_index(drop=True)
        self.max_steps = len(self.df_prices) - 1
        
        # Battery Specs
        self.initial_capacity = battery_params.get('capacity', 20.0) 
        self.p_max = battery_params.get('max_power', 10.0)           
        self.rte = battery_params.get('rte', 0.9)
        self.dt = battery_params.get('time_interval', 0.5)           
        self.standard_deg_cost = 6.50  
        
        # Action Space: [-1.0, 1.0] representing % of max power
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- THE 48-STEP LOOKAHEAD UPDATE ---
        self.forecast_window = 48 # 24 hours at 30-min intervals
        
        # Observation Space: [SOC, SOH, HourOfDay, CurrentPrice] + [48 Future Prices]
        obs_size = 4 + self.forecast_window 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.current_step = 0
        self.soc = 0.0
        self.soh = 1.0
        self.current_capacity = self.initial_capacity * self.soh

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soh = 1.0 
        self.current_capacity = self.initial_capacity * self.soh
        self.soc = 0.5 * self.current_capacity # Start at 50%
        
        return self._get_obs(), {}

    def step(self, action):
        requested_power = action[0] * self.p_max
        rte_sqrt = np.sqrt(self.rte)
        
        # 1. Physics Clipping (Action Masking)
        if requested_power > 0: # Discharging
            max_possible_discharge = (self.soc - (0.1 * self.current_capacity)) * rte_sqrt / self.dt
            actual_power = min(requested_power, max_possible_discharge, self.p_max)
        elif requested_power < 0: # Charging
            max_possible_charge = ((0.9 * self.current_capacity) - self.soc) / (rte_sqrt * self.dt)
            actual_power = max(requested_power, -max_possible_charge, -self.p_max)
        else:
            actual_power = 0.0

        # 2. Update SOC
        if actual_power > 0: 
            energy_out = (actual_power / rte_sqrt) * self.dt
            self.soc -= energy_out
            throughput = energy_out
        else: 
            energy_in = abs(actual_power) * rte_sqrt * self.dt
            self.soc += energy_in
            throughput = 0.0 

        # 3. Calculate Reward (Revenue - Degradation)
        current_price = self.df_prices.loc[self.current_step, 'DayAhead']
        revenue = actual_power * current_price * self.dt
        
        deg_penalty = throughput * self.standard_deg_cost
        
        soc_percentage = self.soc / self.current_capacity
        stress_penalty = 12.00 * self.dt if (soc_percentage < 0.20 or soc_percentage > 0.80) else 0.0
            
        # Reward function scaling: RL agents learn better when rewards are roughly between -1 and 1.
        # Dividing by 100 normalizes the scale of the financial reward slightly.
        reward = (revenue - deg_penalty - stress_penalty) / 100.0

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False 
        
        info = {"actual_power": actual_power, "revenue": revenue, "soc": self.soc}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Extract Hour of Day
        if isinstance(self.df_prices.index, pd.DatetimeIndex):
            hour_of_day = self.df_prices.index[self.current_step].hour
        else:
            hour_of_day = (self.current_step * self.dt) % 24
            
        current_price = self.df_prices.loc[self.current_step, 'DayAhead']
        
        # Build the 48-step forecast window safely
        future_prices = []
        for i in range(1, self.forecast_window + 1):
            if self.current_step + i <= self.max_steps:
                future_prices.append(self.df_prices.loc[self.current_step + i, 'DayAhead'])
            else:
                # If we hit the end of the dataframe, pad with the last known price
                future_prices.append(current_price)
                
        obs = [
            self.soc / self.current_capacity, 
            self.soh,
            hour_of_day / 24.0, # Normalized hour (0 to 1)
            current_price / 100.0 # Heuristically scaled price
        ] + [p / 100.0 for p in future_prices] # Scale forecast prices too
        
        return np.array(obs, dtype=np.float32)
    

def evaluate_and_plot(model, env, df_prices):
    """Runs one full episode using the trained model and plots the agent's strategy."""
    print("Gathering data for visualization...")
    
    obs, info = env.reset()
    
    # Lists to track the agent's performance
    soc_history = [env.soc]
    power_history = []
    price_history = []
    revenue_history = []
    
    # Run the inference loop
    for i in range(len(df_prices) - 1):
        # deterministic=True ensures we see the agent's optimal learned policy, not its exploration
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        soc_history.append(info['soc'])
        power_history.append(info['actual_power'])
        price_history.append(df_prices.loc[i, 'DayAhead'])
        revenue_history.append(info['revenue'])
        
        if terminated:
            break

    # Calculate cumulative profit
    cumulative_profit = np.cumsum(revenue_history)

    # ==========================================
    # Generate the Graphs
    # ==========================================
    # We will look at a 7-day slice (336 half-hourly steps) for clarity, 
    # otherwise a full month looks too compressed.
    plot_steps = min(336, len(power_history))
    time_axis = np.arange(plot_steps)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # --- Subplot 1: Market Price vs. Agent Action (Power) ---
    ax1.set_title('RL Agent Strategy: Price vs. Dispatch', fontsize=14)
    color_price = 'tab:gray'
    ax1.set_ylabel('Market Price (£/MWh)', color=color_price)
    ax1.plot(time_axis, price_history[:plot_steps], color=color_price, alpha=0.6, label='DA Price')
    ax1.tick_params(axis='y', labelcolor=color_price)
    
    ax1_twin = ax1.twinx()
    color_power = 'tab:blue'
    ax1_twin.set_ylabel('Power (MW)', color=color_power)
    # Positive is discharge, Negative is charge
    ax1_twin.bar(time_axis, power_history[:plot_steps], color=color_power, alpha=0.8, width=1.0, label='Battery Power')
    ax1_twin.axhline(0, color='black', linewidth=1)
    ax1_twin.tick_params(axis='y', labelcolor=color_power)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 2: State of Charge (SOC) Management ---
    ax2.set_title('Battery State of Charge (SOC)')
    ax2.set_ylabel('SOC (MWh)')
    ax2.plot(time_axis, soc_history[:plot_steps], color='tab:green', linewidth=2, label='Current SOC')
    
    # Draw the "Stress Zone" boundaries (20% and 80%)
    stress_low = 0.20 * env.current_capacity
    stress_high = 0.80 * env.current_capacity
    ax2.axhline(stress_low, color='red', linestyle='--', alpha=0.5, label='20% Stress Boundary')
    ax2.axhline(stress_high, color='orange', linestyle='--', alpha=0.5, label='80% Stress Boundary')
    
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --- Subplot 3: Cumulative Financial Performance ---
    ax3.set_title('Cumulative Revenue Generated')
    ax3.set_xlabel('Time Step (Half-Hourly)')
    ax3.set_ylabel('Cumulative Profit (£)')
    ax3.plot(time_axis, cumulative_profit[:plot_steps], color='tab:purple', linewidth=2)
    ax3.fill_between(time_axis, 0, cumulative_profit[:plot_steps], color='tab:purple', alpha=0.2)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

# ==========================================
# Training and Execution Script
# ==========================================
if __name__ == "__main__":
    # --- 1. Load Custom Data ---
    DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
    print(f"Loading custom dataset from {DATA_PATH}...")
    
    prices_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    prices_df = prices_df[['DayAhead']]

    # --- 2. Train / Eval Split ---
    # 8760 steps = 1 year of hourly data (or 6 months of half-hourly data)
    train_df = prices_df.iloc[:8760] 
    eval_df = prices_df.iloc[8760:11640] 
    
    print(f"Training steps: {len(train_df)} | Evaluation steps: {len(eval_df)}")

    battery_params = {'capacity': 20.0, 'max_power': 10.0, 'rte': 0.9, 'time_interval': 0.5}

    # --- 3. Instantiate Environments ---
    # We create two separate environments to prevent data leakage.
    # The agent learns on env_train, and proves its worth on env_eval.
    env_train = BESSEnv(train_df, battery_params)
    env_eval = BESSEnv(eval_df, battery_params)
    
    print("Checking training environment compatibility...")
    check_env(env_train)

    # --- 4. Initialize and Train the PPO Model ---
    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", env_train, verbose=1, learning_rate=0.0003)

    print("Starting Training on train_df...")
    # You may need to increase this to 500,000+ for production performance
    model.learn(total_timesteps=100000) 

    model.save("ppo_bess_day_ahead_custom")
    print("Model saved.")

    # --- 5. Evaluate and Plot on Unseen Data ---
    print("Running Inference on eval_df (Unseen Data)...")
    evaluate_and_plot(model, env_eval, eval_df)