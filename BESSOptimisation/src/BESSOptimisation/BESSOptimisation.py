"""Main module."""
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BESS_Optimiser:
    """Mixed Ineteger Linear Programming (MILP) Optimiser for BESS Operation across multiple markets."""
    def __init__(self, prices_df, battery_params):
        """initialising the model parameters and input data"""
        self.arbitrage_markets = ['DayAhead', 'Intraday', 'BM', 'Imbalance']
        self.ancillary_low = ['DCDMLow', 'DRLow'] #needs discharge headroom
        self.ancillary_high = ['DCDMHigh', 'DRHigh'] #needs charge headroom

        self.prices_df = prices_df.sort_index()
        self.time_steps = prices_df.index
        self.markets = self.prices_df.columns.tolist()
        self.all_prices = self.prices_df.to_dict(orient='series')

        #battery parameters
        self.T = len(self.prices_df)
        self.dt = battery_params.get('time_interval', 0.5)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        self.capacity = battery_params.get('capacity', 20)
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.capacity
        self.soc_initial = battery_params.get('soc_initial_factor', 0.5) * self.capacity

        self.cycle_limit = battery_params.get('cycle_limit', 2.0) #1 cycle per day
        self.charge_c_rate = battery_params.get('charge_c_rate', 1.0) #1C charge rate
        self.discharge_c_rate = battery_params.get('discharge_c_rate', 1.0) #1C discharge rate
        self.big_M = self.p_max # Placeholder for big M formulation

        
        self.model = pulp.LpProblem("BESS_Optimisation", pulp.LpMaximize)

    def define_variables(self):
        """setting up linear programming decision variables"""
        charge_discharge_indices = [(t, m) for t in self.time_steps for m in self.markets]
        self.charge = pulp.LpVariable.dicts("charge", charge_discharge_indices, 
                                            lowBound=0, upBound=self.p_max, cat='Continuous') 
        self.discharge = pulp.LpVariable.dicts("discharge", charge_discharge_indices, 
                                               lowBound=0, upBound=self.p_max, cat='Continuous')
        self.soc = pulp.LpVariable.dicts("soc", self.time_steps, 
                                         lowBound=self.soc_min, upBound=self.soc_max, cat='Continuous')
        self.is_charging = pulp.LpVariable.dicts("is_charging", self.time_steps, cat='Binary')
        self.is_discharging = pulp.LpVariable.dicts("is_discharging", self.time_steps, cat='Binary')
        self.daily_cycles = pulp.LpVariable("daily_cycles", lowBound=0, upBound=self.cycle_limit, cat='Continuous') #creating a variable for daily throughput cycles
        self.discharge_energy = pulp.LpVariable.dicts("discharge_energy", self.time_steps, lowBound=0, cat='Continuous') #variable to track energy discharged each time step
        self.high_intensity_discharge = pulp.LpVariable.dicts("high_intensity_discharge", self.time_steps, lowBound=0, cat='Continuous') #variable to track high intensity discharge


    def set_objective(self):
        """
        Defining objective function: 
        1. Maximize Energy Profit (Arbitrage)
        2. Maximize Availability Revenue (Ancillary)
        3. Subtract Degradation Costs (Standard + High C-Rate Penalty)
        """
        # Define Market Groups
        arbitrage_markets = ['DayAhead', 'Intraday', 'BM', 'Imbalance']
        ancillary_markets = ['DCDMLow', 'DCDMHigh', 'DRLow', 'DRHigh']
        
        # 1. Arbitrage Profit (Energy Traded)
        # Revenue = (Discharge - Charge) * Price * Time
        arbitrage_profit = pulp.lpSum(
            (self.discharge[(t, m)] - self.charge[(t, m)]) * self.all_prices[m][t] * self.dt
            for t in self.time_steps for m in arbitrage_markets
        )

        # 2. Ancillary Revenue (Capacity Reserved)
        # Revenue = Reserved Power * Price * Time (No 'Charge' cost subtracted)
        ancillary_revenue = pulp.lpSum(
            (self.discharge[(t, m)] + self.charge[(t, m)]) * self.all_prices[m][t] * self.dt
            for t in self.time_steps for m in ancillary_markets
        )

        # 3. Degradation & Penalty Costs
        standard_deg_cost = 5.0   # £/MWh of throughput
        high_c_penalty = 15.0     # Extra £/MWh for high intensity use
        
        # Standard throughput wear (on total arbitrage discharge)
        deg_cost = pulp.lpSum(
            pulp.lpSum(self.discharge[(t, m)] for m in arbitrage_markets) * self.dt * standard_deg_cost
            for t in self.time_steps
        )
        
        # C-Rate penalty (calculated in set_constraints)
        penalty_cost = pulp.lpSum(
            self.high_intensity_discharge[t] * self.dt * high_c_penalty
            for t in self.time_steps
        )

        # Final Objective: Maximize Net Profit
        self.model += (arbitrage_profit + ancillary_revenue) - (deg_cost + penalty_cost), "Total_Net_Profit"


    def set_constraints(self):
        """applies all physical and operational constraints to the model"""

        total_discharge_energy = []
        safe_c_rate = 0.5 
        safe_power_limit = safe_c_rate * self.capacity

        arb_mkts = ['DayAhead', 'Intraday', 'BM', 'Imbalance']
        anc_low_mkts = ['DCDMLow', 'DRLow']   # Export/Discharge services
        anc_high_mkts = ['DCDMHigh', 'DRHigh'] # Import/Charge services

        for t in self.time_steps:

            # Total energy actually traded (arbitrage)
            step_arb_charge = pulp.lpSum(self.charge[(t, m)] for m in arb_mkts)
            step_arb_discharge = pulp.lpSum(self.discharge[(t, m)] for m in arb_mkts)

            # Total reserve commited (ancillary)
            step_anc_high = pulp.lpSum(self.charge[(t, m)] for m in anc_high_mkts)
            step_anc_low = pulp.lpSum(self.discharge[(t, m)] for m in anc_low_mkts)

            Total_Charge_t = step_arb_charge + step_anc_high
            Total_Discharge_t = step_arb_discharge + step_anc_low

            #Total combined power must not exceed p_max -> you cannot sell your max power to DayAhead and to DR at the same time    
            self.model += Total_Charge_t <= self.p_max, f"Max_Total_Charge_Power_{t}"
            self.model += Total_Discharge_t <= self.p_max, f"Max_Total_Discharge_Power_{t}"
            
            # --- Mutual Exclusivity and Binary Links ---
            self.model += Total_Charge_t <= self.is_charging[t] * self.big_M, f"Charge_Exclusive_{t}"
            self.model += Total_Discharge_t <= self.is_discharging[t] * self.big_M, f"Discharge_Exclusive_{t}"
            self.model += self.is_charging[t] + self.is_discharging[t] <= 1, f"Charging_Status_{t}"

            # === Daily Cycle Limit ===
            self.model += self.discharge_energy[t] == (Total_Discharge_t / self.rte) * self.dt, f"Discharge_Energy_Calc_{t}"
            total_discharge_energy.append(self.discharge_energy[t])

            # ===C-Rate Constraints ===
            self.model += Total_Charge_t <= self.charge_c_rate * self.capacity, f"Charge_CRate_{t}"
            self.model += Total_Discharge_t <= self.discharge_c_rate * self.capacity, f"Discharge_CRate_{t}"
            # --- High Intensity Discharge Tracking ---
            self.model += self.high_intensity_discharge[t] >= Total_Discharge_t - safe_power_limit, f"High_Intensity_Tracking_{t}"

            # --- Energy Balance (SOC Dynamics) ---
            t_loc = self.prices_df.index.get_loc(t)
            soc_prev = self.soc_initial if t_loc == 0 else self.soc[self.time_steps[t_loc - 1]]
                
            self.model += self.soc[t] == soc_prev + (Total_Charge_t * self.rte * self.dt) - \
                                        (Total_Discharge_t / self.rte * self.dt), f"Energy_Balance_{t}"
        
        usable_capacity = self.soc_max - self.soc_min
        self.model += pulp.lpSum(total_discharge_energy) <= self.daily_cycles * usable_capacity, f"Daily_Cycle_Limit_{t}"
            

    def solve_and_collect(self):
        """solves the LP model and extracts the results into a DataFrame."""
        self.model.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0.05)) # Reduced gapRel for faster testing

        if pulp.LpStatus[self.model.status] == 'Optimal':
            print(f"Optimal Solution Found. Total Profit: {pulp.value(self.model.objective):,.2f}")
            results_df = pd.DataFrame(index=self.time_steps) 

            # Extract Total Power and SOC
            results_df['Total Charge'] = [
                sum(self.charge[(t, m)].varValue or 0 for m in self.markets) for t in self.time_steps
            ]
            results_df['Total Discharge'] = [
                sum(self.discharge[(t, m)].varValue or 0 for m in self.markets) for t in self.time_steps
            ]
            results_df['SOC'] = [self.soc[t].varValue or 0 for t in self.time_steps]
            
            # Extract Market-Specific Power and Profit
            total_profit_values = np.zeros(len(self.time_steps))

            for m in self.markets:
                results_df[f'Charge_{m}'] = [self.charge[(t, m)].varValue for t in self.time_steps]
                results_df[f'Discharge_{m}'] = [self.discharge[(t, m)].varValue for t in self.time_steps]
                
                market_profit = [
                    (self.discharge[(t, m)].varValue * self.all_prices[m][t] * self.dt) - 
                    (self.charge[(t, m)].varValue * self.all_prices[m][t] * self.dt)
                    for t in self.time_steps
                ]
                results_df[f'Profit_{m}'] = market_profit
                total_profit_values += np.array(market_profit)

            results_df['Total Hourly Profit'] = total_profit_values
            return results_df
            
        else:
            print(f"No optimal solution found. Status: {pulp.LpStatus[self.model.status]}")
            return None
        
    def plot_operation(self, results_df):
        """generates a figure with stacked subplots for SOC and Hourly Profit."""
        if results_df is None:
            print("Cannot plot; no optimal solution found.")
            return

        # Create a figure and three subplots (3 rows, 1 column)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # --- Subplot 1: State of Charge (SOC) ---
        ax1.plot(results_df.index, results_df['SOC'], color='tab:blue', label='State of Charge', linewidth=2)
        
        # Add min/max SOC limits
        ax1.axhline(self.soc_min, color='gray', linestyle='--', linewidth=1, label='Min SOC')
        ax1.axhline(self.soc_max, color='gray', linestyle='--', linewidth=1, label='Max SOC')
        
        ax1.set_ylabel('SOC (MWh)')
        ax1.set_title('BESS Optimal Operation Schedule', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', linestyle=':')
        ax1.set_ylim(0, self.capacity * 1.05)


        # --- Subplot 3: Hourly Profit ---
        ax3.plot(results_df.index, results_df['Total Hourly Profit'], 
                color='tab:orange', label='Hourly Profit', linestyle='-', marker='.')
        
        # Shade area based on profit/loss
        ax3.fill_between(results_df.index, 0, results_df['Total Hourly Profit'], 
                        where=(results_df['Total Hourly Profit'] > 0), 
                        facecolor='tab:orange', alpha=0.3, interpolate=True)
        ax3.fill_between(results_df.index, 0, results_df['Total Hourly Profit'], 
                        where=(results_df['Total Hourly Profit'] < 0), 
                        facecolor='tab:red', alpha=0.3, interpolate=True)
        
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.set_ylabel('Profit (Currency Unit)')
        ax3.set_xlabel('Time Step')
        ax3.grid(axis='y', linestyle=':')
        ax3.legend(loc='upper left')

        fig.tight_layout()
        plt.show()
        



prices_df = pd.read_csv("Data/GBCentralAllComplete_Prices.csv")
prices_df = prices_df.drop(columns=['Unnamed: 0', 'WeatherYear', 'Year'], errors='ignore') #getting rid of unneccesary columns
prices_df['Date'] = pd.to_datetime(prices_df['Date']) #making sure Date is correct format
prices_df = prices_df.set_index('Date')
prices_df = prices_df.head(96)  # Using a smaller dataset for quicker testing
print(prices_df)
battery_params = {
    'time_interval': 0.5,  # hours
    'max_power': 10,       # MW
    'rte': 0.9,            # round-trip efficiency
    'capacity': 20,        # MWh
    'soc_min_factor': 0.1, # 10% of capacity
    'soc_max_factor': 0.9, # 90% of capacity
    'soc_initial_factor': 0.5, # 50% of capacity
    'cycle_limit': 2.0     # 2 cycles per day
}

optimiser = BESS_Optimiser(prices_df, battery_params)
optimiser.define_variables()
optimiser.set_objective()
optimiser.set_constraints()
results_df = optimiser.solve_and_collect()
print(results_df.head())


if results_df is not None:
    # Now call the new plotting method
    optimiser.plot_operation(results_df)





#improvements to be made:
#need to ensure that SOC only moves for arbitrage, ancillary restricts how full or empty the battery can get
#adding minimum delivery constraints
#market block constraints (EFA blocks)
#utilisation factor for ancillary services to address the fact the ancillary services get called occasionally 
#include contract constraints to ensure battery meets contractual obligations


#optional 
#soc dependent power limits - charging rate changes as batery approaches 100%
#piecewise linear efficiency curve where efficiency peaks at 0.5C and drops at 1.0C
#parasitic losses 
#depth of discharge constraints to protect battery life


