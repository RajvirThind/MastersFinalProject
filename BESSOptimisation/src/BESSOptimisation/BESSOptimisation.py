"""Main module."""
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BESS_Optimiser:
    """Mixed Ineteger Linear Programming (MILP) Optimiser for BESS Operation across multiple markets."""
    def __init__(self, prices_df, battery_params, current_soh = 1.0, last_soc = None):
        """initialising the model parameters and input data"""

        if prices_df is None or prices_df.empty:
            raise ValueError("The prices_df passed to the optimiser is None or empty. "
                             "Check your data slicing in the simulation loop.")

        self.arbitrage_markets = ['DayAhead', 'Intraday', 'BM', 'Imbalance']
        self.ancillary_low = ['DCDMLow', 'DRLow'] #needs discharge headroom
        self.ancillary_high = ['DCDMHigh', 'DRHigh'] #needs charge headroom

        self.prices_df = prices_df.sort_index()
        self.time_steps = prices_df.index
        self.markets = self.prices_df.columns.tolist()
        self.all_prices = self.prices_df.to_dict(orient='series')
        self.T = len(self.prices_df)

        #battery parameters
        self.initial_capacity = battery_params.get('initial_capacity', 20)
        self.soh = current_soh

        #calculating available capacity based on SOH 
        self.current_capacity = self.initial_capacity * self.soh


        self.dt = battery_params.get('time_interval', 0.5)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        #self.capacity = battery_params.get('capacity', 20)
    
        #dynamic SOC limits, must shrink as battery degrades
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.current_capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.current_capacity

        #is this is day 2, use the SOC from the previous day
        if last_soc is not None:
            self.soc_initial = last_soc
        else:
            self.soc_initial = battery_params.get('soc_initial_factor', 0.5) * self.current_capacity

        self.cycle_limit = battery_params.get('cycle_limit', 2.0) #1 cycle per day
        self.deg_per_mwh = battery_params.get('deg_per_mwh', 0.00001) #degradation per MWh discharged
        self.charge_c_rate = battery_params.get('charge_c_rate', 1.0) #1C charge rate
        self.discharge_c_rate = battery_params.get('discharge_c_rate', 1.0) #1C discharge rate
        self.utilisation_factor = battery_params.get('utilisation_factor', 0.02)
        self.big_M = self.p_max * 1.1 # Big M value for constraints

        #skip rates for risk adjusted revenue
        self.skip_rates = battery_params.get('skip_rates', pd.DataFrame(0, index=self.time_steps, columns=self.markets))

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

        # 2. Ancillary Revenue with Skip Rate Penalty (Availability)
        # Revenue = Reserved Power * (Price - Penalty * SkipRate)
        penalty_weight = 10.0 # Adjust based on risk tolerance
        ancillary_revenue = pulp.lpSum(
            (self.discharge[(t, m)] + self.charge[(t, m)]) * (self.all_prices[m][t] - penalty_weight * self.skip_rates.loc[t, m]) * self.dt
            for t in self.time_steps for m in (self.ancillary_low + self.ancillary_high)
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
        safe_power_limit = safe_c_rate * self.current_capacity
        alpha = self.utilisation_factor

        arb_mkts = ['DayAhead', 'Intraday', 'BM', 'Imbalance']
        anc_low_mkts = ['DCDMLow', 'DRLow']   # Export/Discharge services
        anc_high_mkts = ['DCDMHigh', 'DRHigh'] # Import/Charge services

        # EFA block constraints (4 hour locks for ancillary services)
        # 4 hours = 8 steps (since dt = 0.5 hours)
        steps_per_block = int(4 / self.dt)

        for block_start in range(0, len(self.time_steps), steps_per_block):
            block_indices = self.time_steps[block_start: block_start + steps_per_block]
            first_t = block_indices[0]

            for m in (anc_low_mkts + anc_high_mkts):
                for next_t in block_indices[1:]:
                    # Force all steps in the block to have the same power for market m
                    self.model += self.charge[(next_t, m)] == self.charge[(first_t, m)], f"EFA_Block_Ch_{m}_{next_t}"
                    self.model += self.discharge[(next_t, m)] == self.discharge[(first_t, m)], f"EFA_Block_Ds_{m}_{next_t}"


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

            # ===C-Rate Constraints ===
            self.model += Total_Charge_t <= self.charge_c_rate * self.current_capacity, f"Charge_CRate_{t}"
            self.model += Total_Discharge_t <= self.discharge_c_rate * self.current_capacity, f"Discharge_CRate_{t}"
            # --- High Intensity Discharge Tracking ---
            self.model += self.high_intensity_discharge[t] >= Total_Discharge_t - safe_power_limit, f"High_Intensity_Tracking_{t}"

            # --- Energy Balance (SOC Dynamics) ---
            t_loc = self.prices_df.index.get_loc(t)
            soc_prev = self.soc_initial if t_loc == 0 else self.soc[self.time_steps[t_loc - 1]]

            # SOC moves firm for Arbitrage, and slightly (alpha) for Ancillary
            energy_in = (step_arb_charge + (alpha * step_anc_high)) * self.rte * self.dt
            energy_out = (step_arb_discharge + (alpha * step_anc_low)) / self.rte * self.dt
            self.model += self.soc[t] == soc_prev + energy_in - energy_out, f"Energy_Balance_{t}"

            self.model += self.soc[t] >= self.soc_min + (step_anc_low * 0.5), f"Ancillary_Footroom_{t}" #ensure we have enough energy to discharge if an ancillary low service is called 
            self.model += self.soc[t] <= self.soc_max - (step_anc_high * 0.5), f"Ancillary_Headroom_{t}" #ensure we have enough energy to charge if an ancillary high service is called
            #0.5 is used which represents a 30 minute full power delivery requirement

            # === Daily Cycle Limit ===
            self.model += self.discharge_energy[t] == energy_out, f"Discharge_Energy_Calc_{t}"
            total_discharge_energy.append(self.discharge_energy[t])


        usable_capacity = self.soc_max - self.soc_min
        self.model += pulp.lpSum(total_discharge_energy) <= self.daily_cycles * usable_capacity, f"Global_Daily_Cycle_Limit_{t}"
            

    def solve_and_collect(self):
        """solves the LP model and extracts the results into a DataFrame safely."""
        # Use a slightly more relaxed gap for speed in 10-year runs if needed
        self.model.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0.05))

        # Check for Optimal or Integer Feasible status
        if pulp.LpStatus[self.model.status] in ['Optimal', 'Feasible']:
            print(f"Profit: {pulp.value(self.model.objective):,.2f} | Cycles: {pulp.value(self.daily_cycles):.2f}")
            
            results_df = pd.DataFrame(index=self.time_steps) 
            
            # Helper to handle None values (occurs if solver fails on a specific step)
            get_v = lambda x: x.varValue if x.varValue is not None else 0.0

            results_df['SOC'] = [get_v(self.soc[t]) for t in self.time_steps]
            
            total_profit_values = np.zeros(len(self.time_steps))

            for m in self.markets:
                # Extracting power values safely
                p_ch = np.array([get_v(self.charge[(t, m)]) for t in self.time_steps])
                p_ds = np.array([get_v(self.discharge[(t, m)]) for t in self.time_steps])
                
                results_df[f'Power_{m}'] = p_ds - p_ch
                
                if m in self.arbitrage_markets:
                    m_profit = (p_ds - p_ch) * self.all_prices[m] * self.dt
                else:
                    m_profit = (p_ds + p_ch) * self.all_prices[m] * self.dt # Availability
                
                results_df[f'Profit_{m}'] = m_profit
                total_profit_values += m_profit

            results_df['Total Hourly Profit'] = total_profit_values
            
            # Crucial for SOH tracking in the 10-year loop
            results_df['Throughput_MWh'] = [get_v(self.discharge_energy[t]) for t in self.time_steps]
            
            return results_df
        
        # If the solver is Infeasible, return None so the outer loop can perform an SOC reset
        print(f"Solver Status: {pulp.LpStatus[self.model.status]} - Skipping current period.")
        return None
        
    def plot_operation(self, results_df):
        """Generates a figure with stacked subplots for SOC, Market Power, and Hourly Profit."""
        if results_df is None:
            print("Cannot plot; no optimal solution found.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # --- Subplot 1: State of Charge (SOC) ---
        ax1.plot(results_df.index, results_df['SOC'], color='tab:blue', label='SOC', linewidth=2)
        ax1.axhline(self.soc_min, color='red', linestyle='--', alpha=0.5, label='Min SOC')
        ax1.axhline(self.soc_max, color='green', linestyle='--', alpha=0.5, label='Max SOC')
        ax1.set_ylabel('SOC (MWh)')
        ax1.set_title('BESS Optimal Operation Schedule', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- Subplot 2: Market Power Allocation (Stacked) ---
        # Separating the columns for stacking
        power_cols = [c for c in results_df.columns if 'Power_' in c]
        
        # We plot positive (discharge/low) and negative (charge/high) separately for clarity
        for col in power_cols:
            market_name = col.replace('Power_', '')
            ax2.bar(results_df.index, results_df[col], width=self.dt/24, label=market_name, alpha=0.8)
            
        ax2.set_ylabel('Power (MW)')
        ax2.set_title('Market Power Allocation (Stacking)')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        ax2.grid(True, linestyle=':', alpha=0.6)

        # --- Subplot 3: Hourly Profit Breakdown ---
        # Plotting the total profit as a line and shading the area
        ax3.plot(results_df.index, results_df['Total Hourly Profit'], 
                color='tab:orange', label='Total Profit', linewidth=1.5)
        ax3.fill_between(results_df.index, 0, results_df['Total Hourly Profit'], 
                        color='tab:orange', alpha=0.2)
        
        ax3.set_ylabel('Profit (Currency)')
        ax3.set_xlabel('Time Step')
        ax3.set_title('Total Operational Profit')
        ax3.grid(True, linestyle=':', alpha=0.6)

        fig.tight_layout()
        plt.show()
        



# prices_df = pd.read_csv("Data/GBCentralAllComplete_Prices.csv")
# prices_df = prices_df.drop(columns=['Unnamed: 0', 'WeatherYear', 'Year'], errors='ignore') #getting rid of unneccesary columns
# prices_df['Date'] = pd.to_datetime(prices_df['Date']) #making sure Date is correct format
# prices_df = prices_df.set_index('Date')
# prices_df = prices_df.head(96)  # Using a smaller dataset for quicker testing
# print(prices_df)
# battery_params = {
#     'time_interval': 0.5,  # hours
#     'max_power': 10,       # MW
#     'rte': 0.9,            # round-trip efficiency
#     'capacity': 20,        # MWh
#     'soc_min_factor': 0.1, # 10% of capacity
#     'soc_max_factor': 0.9, # 90% of capacity
#     'soc_initial_factor': 0.5, # 50% of capacity
#     'cycle_limit': 2.0     # 2 cycles per day
# }

# optimiser = BESS_Optimiser(prices_df, battery_params)
# optimiser.define_variables()
# optimiser.set_objective()
# optimiser.set_constraints()
# results_df = optimiser.solve_and_collect()
# print(results_df.head())


# if results_df is not None:
#     # Now call the new plotting method
#     optimiser.plot_operation(results_df)





#improvements to be made:
# incorporate skip rates


