import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import your custom environment
from BESSOptimisation.src.BESSOptimisation.environments.dayahead_bess_env import BESS_RLEnv

# --- CONFIGURATION ---
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
MODEL_NAME = "bess_ppo_agent"
LOG_DIR = "./ppo_bess_logs/"

# Toggle this to True to run a 10-second test to check for crashes
DEBUG_MODE = False 

# Use a safe number of cores (leave 1-2 for your OS)
NUM_CORES = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

params = {
    'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 'rte': 0.9,
    'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'soc_initial_factor': 0.5,
    'cycle_limit': 1.1 
}

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # --- Load Data ---
    full_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    full_df = full_df[['DayAhead']]
    
    if DEBUG_MODE:
        print("!!! RUNNING IN DEBUG MODE !!!")
        train_df = full_df.iloc[:48 * 7]  # Train on 7 days
        eval_df = full_df.iloc[48 * 7 : 48 * 14] # Eval on next 7 days
        total_timesteps = 10_000
    else:
        # Full Production Run
        train_df = full_df.iloc[:48 * 180] # Train on 180 days
        eval_df = full_df.iloc[48 * 180 : 48 * 210] # Eval on unseen next 30 days
        total_timesteps = 1_000_000 # PPO needs more steps, but executes them much faster

    # --- Setup Vectorized Environments ---
    # We use a factory function for SubprocVecEnv to avoid multiprocessing Pickling errors
    def make_train_env():
        def _init():
            # is_training=True disables the history list for speed
            env = BESS_RLEnv(train_df, params, is_training=True)
            return Monitor(env)
        return _init

    print(f"Setting up {NUM_CORES} environments in parallel...")
    train_env = SubprocVecEnv([make_train_env() for _ in range(NUM_CORES)])
    
    # Eval environment stays single-core and tracks history (is_training=False)
    eval_env = DummyVecEnv([lambda: Monitor(BESS_RLEnv(eval_df, params, is_training=False))])

    # --- Initialize PPO ---
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=dict(net_arch=[256, 256]), # Wide network for complex pricing logic
        tensorboard_log=LOG_DIR
    )

    print(f"Training on device: {model.device}")

    # --- Callbacks ---
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR, 
        eval_freq=max(10_000 // NUM_CORES, 1), # Adjust frequency for parallel workers
        deterministic=True, 
        render=False
    )

    # --- Train ---
    print(f"Starting Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # --- Save ---
    model.save(MODEL_NAME)
    print(f"\nModel saved to {MODEL_NAME}.zip")
    
    # Clean up parallel workers
    train_env.close()

if __name__ == "__main__":
    # Required for safe multiprocessing on Windows and macOS
    main()