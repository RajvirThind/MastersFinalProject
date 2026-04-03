import os
import torch as th
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import SAC
from DayAheadMILP_optimiser import BESS_Optimiser
from BESSOptimisation.src.BESSOptimisation.test_models.dayahead_agent_v3 import BESSEnv


#this code implements behavioural cloning, which is a form of imitation learning. the goal is to take a slow, mathemtical solver and use its logic into a fast neural network 
#the MILP solver provides a set of expert demonstrations, which the RL agent will learn to imitate


# --- 1. CONFIGURATION ---
DATA_PATH = 'data/GBCentralAllComplete_Prices.csv'
MODEL_NAME = "bess_hybrid_imitation"
LOOKAHEAD = 48
TRAIN_DAYS = 180  # Number of days to let the MILP "teach" the RL
BATCH_SIZE = 512
EPOCHS = 100

params = {
    'time_interval': 0.5, 'max_power': 10, 'capacity': 20, 'rte': 0.9,
    'soc_min_factor': 0.1, 'soc_max_factor': 0.9, 'soc_initial_factor': 0.5,
    'deg_per_mwh': 0.00001
}

def generate_expert_dataset(df, battery_params):
    """Phase 1: Let the MILP solve and record the 'Perfect' actions."""
    print(f"--- Phase 1: Generating Expert Data from MILP ({TRAIN_DAYS} days) ---")
    obs_list = []
    act_list = []
    
    current_soh = 1.0
    last_soc = battery_params['soc_initial_factor'] * battery_params['capacity']
    
    # pre-convert prices to a numpy array for faster slicing
    prices_array = df['DayAhead'].values
    
    for i in range(TRAIN_DAYS): 
        start = i * 48
        day_df = df.iloc[start : start + 48] #for each day it is pulling 48 half hour slots

        opt = BESS_Optimiser(day_df, battery_params, current_soh, last_soc) #running the MILP optimiser
        opt.define_variables()
        opt.set_objective()
        opt.set_constraints()
        results = opt.solve_and_collect()
        
        if results is not None:
            for t in results.index:
                soc_val = results.at[t, 'SOC']
                
                # Robust Index Lookup
                raw_idx = df.index.get_loc(t)
                if isinstance(raw_idx, (slice, np.ndarray)):
                    # Handle cases where get_loc returns a slice or boolean mask
                    t_idx = raw_idx.start if isinstance(raw_idx, slice) else np.where(raw_idx)[0][0]
                else:
                    t_idx = raw_idx
                
                # slice the price window
                end_idx = t_idx + LOOKAHEAD
                window = prices_array[t_idx : end_idx]
                
                # pad if near the end of the dataframe
                if len(window) < LOOKAHEAD:
                    window = np.pad(window, (0, LOOKAHEAD - len(window)), mode='edge')

                obs = np.concatenate([[soc_val], window])
                
                #normalising action: MW to [-1, 1]
                action = results.at[t, 'Power'] / battery_params['max_power']
                
                obs_list.append(obs)
                act_list.append([action])
                
            # update state for next day simulation
            current_soh -= (results['Throughput'].sum() * battery_params['deg_per_mwh'])
            last_soc = results['SOC'].iloc[-1]
            print(f"MILP solved Day {i+1}")

    return np.array(obs_list), np.array(act_list)

def train_imitation_model(env, expert_obs, expert_acts):
    """Phase 2: Supervised Learning (Behavioral Cloning)."""
    print(f"--- Phase 2: Training RL Agent to Mimic MILP ---")
    
    # initialise SAC
    model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[256, 256])) #creating two hidden layers of 256 neurons each

    #manual step, trigger internal setup
    model._setup_model()
    
    # Prepare PyTorch Tensors
    obs_tensor = th.tensor(expert_obs, dtype=th.float32)
    act_tensor = th.tensor(expert_acts, dtype=th.float32)
    loader = DataLoader(TensorDataset(obs_tensor, act_tensor), batch_size=BATCH_SIZE, shuffle=True)
    
    # Access the actor (policy) optimizer specifically for Behavioral Cloning
    # In SAC, there are separate optimizers for actor and critic.
    # Since we are mimicking actions, we only need to update the actor.
    optimizer = model.actor.optimizer
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for b_obs, b_act in loader:
            pred_actions = model.actor(b_obs)
            
            # 1. Standard Imitation Loss
            imitation_loss = th.nn.functional.mse_loss(pred_actions, b_act)
            
            # 2. NEW: Sparsity Penalty
            # Punishes the agent for non-zero actions when the MILP target is exactly 0
            zero_mask = (b_act == 0).float()
            sparsity_loss = th.mean(th.square(pred_actions) * zero_mask)
            
            # Combine: Higher weight on sparsity (e.g., 0.2) reduces jitter
            total_loss = imitation_loss + (0.2 * sparsity_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {epoch_loss/len(loader):.6f}")
            
    model.save(MODEL_NAME)
    print(f"Hybrid Model Saved as {MODEL_NAME}")
    return model

if __name__ == "__main__":
    #load Data
    full_df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    full_df = full_df[['DayAhead']]
    
    #collect Expert Data
    x_train, y_train = generate_expert_dataset(full_df, params)
    
    #create Environment and Train
    train_env = BESSEnv(full_df.iloc[:48*TRAIN_DAYS], params)
    hybrid_model = train_imitation_model(train_env, x_train, y_train)