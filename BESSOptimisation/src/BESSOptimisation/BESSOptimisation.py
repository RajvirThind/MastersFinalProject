"""Main module."""
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy_financial as npf

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
        self.initial_capacity = battery_params.get('capacity', 20)
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

        self.cycle_limit = battery_params.get('cycle_limit', 1.1) #cycles per day
        self.deg_per_mwh = battery_params.get('deg_per_mwh', 0.00001) #degradation per MWh discharged
        self.charge_c_rate = battery_params.get('charge_c_rate', 1.0) #1C charge rate
        self.discharge_c_rate = battery_params.get('discharge_c_rate', 1.0) #1C discharge rate
        self.utilisation_factor = battery_params.get('utilisation_factor', 0.02)
        self.big_M = self.p_max * 1.1 # Big M value for constraints

        #skip rates for risk adjusted revenue
        self.skip_rates = battery_params.get('skip_rates', pd.DataFrame(0, index=self.time_steps, columns=self.markets))

        self.model = pulp.LpProblem("BESS_Optimisation", pulp.LpMaximize)

    def define_variables(self):
        """Setting up linear programming decision variables with Integer and Binary types."""
        charge_discharge_indices = [(t, m) for t in self.time_steps for m in self.markets]
        
        # 1. Existing Power and Status Variables
        self.charge = pulp.LpVariable.dicts("charge", charge_discharge_indices, 
                                            lowBound=0, upBound=self.p_max, cat='Continuous') 
        self.discharge = pulp.LpVariable.dicts("discharge", charge_discharge_indices, 
                                            lowBound=0, upBound=self.p_max, cat='Continuous')
        self.is_charging = pulp.LpVariable.dicts("is_charging", self.time_steps, cat='Binary')
        self.is_discharging = pulp.LpVariable.dicts("is_discharging", self.time_steps, cat='Binary')
        
        # 2. Existing SOC and Throughput Variables
        self.soc = pulp.LpVariable.dicts("soc", self.time_steps, 
                                        lowBound=self.soc_min, upBound=self.soc_max, cat='Continuous')
        self.discharge_energy = pulp.LpVariable.dicts("discharge_energy", self.time_steps, lowBound=0)
        self.daily_cycles = pulp.LpVariable("daily_cycles", lowBound=0)
        self.high_intensity_discharge = pulp.LpVariable.dicts("high_intensity_discharge", self.time_steps, lowBound=0)

        # 3. NEW: Ancillary Logic Variables (Integer MW and Min Capacity)
        anc_indices = [(t, m) for t in self.time_steps for m in (self.ancillary_low + self.ancillary_high)]
        self.anc_mw = pulp.LpVariable.dicts("anc_mw", anc_indices, lowBound=0, upBound=self.p_max, cat='Integer')
        self.anc_active = pulp.LpVariable.dicts("anc_active", anc_indices, cat='Binary')

        # 4. NEW: Piecewise Linear SOC (Safe vs. Stress zones)
        self.soc_safe = pulp.LpVariable.dicts("soc_safe", self.time_steps, lowBound=0)
        self.soc_stress = pulp.LpVariable.dicts("soc_stress", self.time_steps, lowBound=0)        

    def set_objective(self):
        """
        Defining the objective function to maximize net profit while internalizing:
        1. Piecewise Linear Degradation (Stress vs. Safe SOC zones)
        2. High C-Rate (Intensity) wear and tear
        3. Balancing Mechanism (BM) skip-rate risk weighting
        """
        # --- 1. Market Revenue Streams ---
        
        # Arbitrage (DayAhead, Intraday, Imbalance)
        # Excludes BM here to apply specific risk weighting below
        arb_basic = ['DayAhead', 'Intraday', 'Imbalance']
        arbitrage_profit = pulp.lpSum(
            (self.discharge[(t, m)] - self.charge[(t, m)]) * self.all_prices[m][t] * self.dt
            for t in self.time_steps for m in arb_basic
        )

        # Balancing Mechanism (BM) with 80% Skip-Rate Weighted Revenue
        # We penalize the 'expected' revenue to reflect that most bids are ignored
        bm_profit = pulp.lpSum(
            (self.discharge[(t, 'BM')] - self.charge[(t, 'BM')]) * self.all_prices['BM'][t] * (1 - self.skip_rates.loc[t, 'BM']) * self.dt
            for t in self.time_steps
        )

        # Ancillary Revenue (Availability)
        # Revenue = MW * (£/MW/h) * Time
        ancillary_revenue = pulp.lpSum(
            (self.discharge[(t, m)] + self.charge[(t, m)]) * self.all_prices[m][t] * self.dt
            for t in self.time_steps for m in (self.ancillary_low + self.ancillary_high)
        )

        # --- 2. Cost & Penalty Terms (Internalizing Degradation) ---
        
        # A. Standard Throughput Cost (£/MWh)
        # Reflects the 'Levelized Cost of Storage' (LCOS) wear-and-tear
        # In 2026, standard LFP wear is roughly £5.00 - £8.00 per MWh discharged
        standard_deg_cost = 6.50   
        deg_cost = pulp.lpSum(
            self.discharge_energy[t] * standard_deg_cost for t in self.time_steps
        )

        # B. Piecewise Linear Stress Penalty
        # Penalizes the battery for sitting in or moving through high/low SOC zones (<20% or >80%)
        # This mimics accelerated chemical degradation at SOC extremes.
        stress_penalty_rate = 12.00 # Extra £ per MWh equivalent sitting in stress zone
        stress_cost = pulp.lpSum(
            self.soc_stress[t] * stress_penalty_rate * self.dt for t in self.time_steps
        )

        # C. C-Rate / High Intensity Penalty
        # Extra wear-and-tear for discharging above 0.5C (e.g., fast 1C bursts)
        high_c_penalty_rate = 15.00 # £ per MWh for high-power usage
        intensity_cost = pulp.lpSum(
            self.high_intensity_discharge[t] * self.dt * high_c_penalty_rate
            for t in self.time_steps
        )

        # --- 3. Final Objective Function ---
        # Maximize: (Total Revenue) - (Total Internalized Costs)
        total_revenue = arbitrage_profit + bm_profit + ancillary_revenue
        total_costs = deg_cost + stress_cost + intensity_cost
        
        self.model += total_revenue - total_costs, "Total_Net_Profit"

    def set_constraints(self):
        """
        Applies all physical, operational, and contractual constraints to the model.
        Ensures EFA blocks, 1MW minimums, whole MW bids, and SOC duration safety.
        """
        total_discharge_energy = []
        safe_c_rate = 0.5 
        safe_power_limit = safe_c_rate * self.current_capacity
        alpha = self.utilisation_factor
        
        # Piecewise SOC Thresholds (Stress Zone < 20% or > 80%)
        safe_lower_threshold = 0.20 * self.current_capacity
        safe_upper_threshold = 0.80 * self.current_capacity
        
        # 2026 Delivery Durations (Hours)
        durations = {
            'DCDMLow': 0.25, 'DCDMHigh': 0.25, # 15 mins for DC
            'DRLow': 1.0, 'DRHigh': 1.0        # 60 mins for DR
        }

        # === 1. EFA Block Constraints (4-hour locks) ===
        steps_per_block = int(4 / self.dt)
        for block_start in range(0, len(self.time_steps), steps_per_block):
            block_indices = self.time_steps[block_start : block_start + steps_per_block]
            first_t = block_indices[0]
            for m in (self.ancillary_low + self.ancillary_high):
                for next_t in block_indices[1:]:
                    self.model += self.charge[(next_t, m)] == self.charge[(first_t, m)], f"EFA_Block_Ch_{m}_{next_t}"
                    self.model += self.discharge[(next_t, m)] == self.discharge[(first_t, m)], f"EFA_Block_Ds_{m}_{next_t}"

        for t in self.time_steps:
            # Market Summation
            step_arb_charge = pulp.lpSum(self.charge[(t, m)] for m in self.arbitrage_markets)
            step_arb_discharge = pulp.lpSum(self.discharge[(t, m)] for m in self.arbitrage_markets)
            step_anc_high = pulp.lpSum(self.charge[(t, m)] for m in self.ancillary_high)
            step_anc_low = pulp.lpSum(self.discharge[(t, m)] for m in self.ancillary_low)

            Total_Charge_t = step_arb_charge + step_anc_high
            Total_Discharge_t = step_arb_discharge + step_anc_low

            # --- Power Limits & Binary Exclusivity ---
            self.model += Total_Charge_t <= self.p_max, f"Max_Total_Charge_Power_{t}"
            self.model += Total_Discharge_t <= self.p_max, f"Max_Total_Discharge_Power_{t}"
            self.model += Total_Charge_t <= self.is_charging[t] * self.big_M, f"Charge_Exclusive_{t}"
            self.model += Total_Discharge_t <= self.is_discharging[t] * self.big_M, f"Discharge_Exclusive_{t}"
            self.model += self.is_charging[t] + self.is_discharging[t] <= 1, f"Charging_Status_{t}"

            # --- Ancillary Specific: Whole MW & 1 MW Minimum ---
            for m in (self.ancillary_low + self.ancillary_high):
                self.model += self.charge[(t, m)] == self.anc_mw[(t, m)], f"Whole_MW_Ch_{t}_{m}"
                self.model += self.discharge[(t, m)] == self.anc_mw[(t, m)], f"Whole_MW_Ds_{t}_{m}"
                # Min 1MW logic: Power >= 1 * Binary
                self.model += self.anc_mw[(t, m)] >= 1 * self.anc_active[(t, m)], f"Min_Cap_Low_{t}_{m}"
                self.model += self.anc_mw[(t, m)] <= self.p_max * self.anc_active[(t, m)], f"Min_Cap_High_{t}_{m}"

            # --- Piecewise SOC Degradation Logic ---
            self.model += self.soc[t] == self.soc_safe[t] + self.soc_stress[t], f"SOC_Composition_{t}"
            self.model += self.soc_safe[t] <= (safe_upper_threshold - safe_lower_threshold), f"Safe_SOC_Limit_{t}"

            # --- Energy Balance (Utilization-Adjusted) ---
            t_loc = self.prices_df.index.get_loc(t)
            soc_prev = self.soc_initial if t_loc == 0 else self.soc[self.time_steps[t_loc - 1]]
            
            energy_in = (step_arb_charge + (alpha * step_anc_high)) * self.rte * self.dt
            energy_out = (step_arb_discharge + (alpha * step_anc_low)) / self.rte * self.dt
            self.model += self.soc[t] == soc_prev + energy_in - energy_out, f"Energy_Balance_{t}"

            # --- Advanced Delivery Duration (Safety Buffer) ---
            # Summing (Power * Required_Duration_Hours) for each active service
            anc_low_energy_req = pulp.lpSum(self.discharge[(t, m)] * durations.get(m, 0.5) for m in self.ancillary_low)
            anc_high_energy_req = pulp.lpSum(self.charge[(t, m)] * durations.get(m, 0.5) for m in self.ancillary_high)

            self.model += self.soc[t] >= self.soc_min + (anc_low_energy_req / self.rte), f"Ancillary_Footroom_{t}"
            self.model += self.soc[t] <= self.soc_max - (anc_high_energy_req * self.rte), f"Ancillary_Headroom_{t}"

            # High Intensity Tracking & Cycle Budget
            self.model += self.high_intensity_discharge[t] >= Total_Discharge_t - safe_power_limit, f"High_Intensity_Tracking_{t}"
            self.model += self.discharge_energy[t] == energy_out, f"Discharge_Energy_Calc_{t}"
            total_discharge_energy.append(self.discharge_energy[t])

        # === Global Cycle Budget ===
        usable_capacity = self.soc_max - self.soc_min
        num_days_in_window = (len(self.time_steps) * self.dt) / 24
        total_cycle_budget = self.cycle_limit * num_days_in_window
        self.model += pulp.lpSum(total_discharge_energy) <= total_cycle_budget * usable_capacity, "Global_Cycle_Limit_Constraint"
        self.model += self.daily_cycles == pulp.lpSum(total_discharge_energy) / usable_capacity / num_days_in_window
            

    def solve_and_collect(self):
        """solves the LP model and extracts the results into a DataFrame safely."""
        # Use a slightly more relaxed gap for speed in 10-year runs if needed
        self.model.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0.2))

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

    def plot_long_term_appraisal(summary_df):
        """Generates a multi-year view of revenue streams and battery health."""
        # Group by month to make the chart readable over 10 years
        summary_df['Date'] = pd.to_datetime(summary_df['Date'])
        monthly = summary_df.resample('M', on='Date').sum()
        monthly_soh = summary_df.resample('M', on='Date').mean()['SOH'] # Use average SOH for the line

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # --- Subplot 1: Stacked Revenue Streams ---
        # We stack the different profit sources
        revenue_cols = ['Arbitrage_Profit', 'Ancillary_Low_Profit', 'Ancillary_High_Profit']
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e'] # Blue, Green, Orange
        
        ax1.stackplot(monthly.index, 
                    [monthly[col] for col in revenue_cols],
                    labels=['Arbitrage', 'Ancillary Low', 'Ancillary High'],
                    colors=colors, alpha=0.8)
        
        ax1.set_ylabel('Monthly Profit (£)')
        ax1.set_title('10-Year Forecasted Revenue Breakdown', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- Subplot 2: SOH vs Cumulative Profit ---
        ax2_twin = ax2.twinx()
        
        # Plot SOH (State of Health)
        lns1 = ax2.plot(monthly.index, monthly_soh * 100, color='red', linewidth=2, label='SOH %')
        ax2.set_ylabel('State of Health (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(70, 105)
        ax2.axhline(80, color='black', linestyle='--', alpha=0.5, label='End of Life (80%)')

        # Plot Cumulative Profit
        cum_profit = monthly['Total_Profit'].cumsum()
        lns2 = ax2_twin.plot(monthly.index, cum_profit, color='blue', linestyle=':', label='Cum. Profit')
        ax2_twin.set_ylabel('Cumulative Profit (£)', color='blue')
        ax2_twin.tick_params(axis='y', labelcolor='blue')

        ax2.set_title('Battery Degradation and Cumulative Wealth')
        ax2.set_xlabel('Year')
        
        # Merging legends for the twin axis
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_financial_appraisal(summary_df, battery_params):
        # --- 1. Parameters & Assumptions (2026 Benchmarks) ---
        p_max_mw = battery_params.get('max_power', 10)
        cap_mwh = battery_params.get('capacity', 20)
        
        capex_per_mwh = 135000       # £135k per MWh for LFP
        opex_per_mw_year = 6500      # £6.5k per MW per year (Fixed)
        discount_rate = 0.09         # 9% WACC
        
        initial_investment = cap_mwh * capex_per_mwh
        
        # --- 2. Aggregate Data to Annual ---
        # Ensure Date is datetime
        summary_df['Date'] = pd.to_datetime(summary_df['Date'])
        yearly = summary_df.resample('YE', on='Date').agg({
            'Total_Profit': 'sum',
            'SOH': 'min'
        })
        
        # Calculate Net Cash Flow (Profit - Fixed OPEX)
        yearly['Net_Cash_Flow'] = yearly['Total_Profit'] - (p_max_mw * opex_per_mw_year)
        
        # Create the Cash Flow Series (Year 0 is CAPEX)
        cash_flows = [-initial_investment] + yearly['Net_Cash_Flow'].tolist()
        cum_cash_flow = np.cumsum(cash_flows)
        years = np.arange(0, len(cash_flows))

        # --- 3. Calculate Metrics ---
        npv_val = npf.npv(discount_rate, cash_flows)
        irr_val = npf.irr(cash_flows)
        
        # --- 4. Plotting the J-Curve ---
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Bar chart for annual net cash flow
        bars = ax1.bar(years[1:], yearly['Net_Cash_Flow'], color='skyblue', alpha=0.7, label='Annual Net Cash Flow')
        
        # Line chart for cumulative cash flow
        ax1.plot(years, cum_cash_flow, color='navy', marker='o', linewidth=2, label='Cumulative Cash Flow')
        
        # Formatting
        ax1.axhline(0, color='black', linewidth=1) # Breakeven line
        ax1.set_xlabel('Project Year')
        ax1.set_ylabel('Currency (£)')
        ax1.set_title(f'BESS Financial Appraisal: NPV £{npv_val:,.0f} | IRR {irr_val*100:.2f}%', fontsize=14)
        
        # Annotate Payback Period
        payback_year = next((i for i, v in enumerate(cum_cash_flow) if v > 0), None)
        if payback_year:
            ax1.annotate(f'Payback: Year {payback_year}', 
                        xy=(payback_year, cum_cash_flow[payback_year]),
                        xytext=(payback_year-1, cum_cash_flow[payback_year]+500000),
                        arrowprops=dict(facecolor='black', shrink=0.05))

        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def run_financial_analysis(summary_df, battery_params):
        """Post-processing static method for NPV/IRR."""
        import numpy_financial as npf
        
        # Financial Benchmarks
        p_max = battery_params.get('max_power', 10)
        capacity = battery_params.get('capacity', 20)
        capex_total = capacity * 135000 
        opex_annual = p_max * 6500
        discount_rate = 0.09
        
        summary_df['Date'] = pd.to_datetime(summary_df['Date'])
        yearly = summary_df.resample('YE', on='Date').agg({'Total_Profit': 'sum'})
        
        # Cash Flow: [Year 0: -CAPEX, Year 1: (Profit - OPEX), ...]
        annual_net_flows = (yearly['Total_Profit'] - opex_annual).tolist()
        cash_flows = [-capex_total] + annual_net_flows
        
        return {
            "npv": npf.npv(discount_rate, cash_flows),
            "irr": npf.irr(cash_flows),
            "cum_cf": np.cumsum(cash_flows),
            "years": np.arange(len(cash_flows))
        }



#improvements to be made:
# consider non linear battery degradation
# add payback period and IRR graphs, need to estimate price/MW for battery installation
# daily profit seems high considering modo energy has provided figures of £250/MW per day: https://www.solarpowerportal.co.uk/battery-storage/british-bess-earns-second-highest-daily-total-price-for-2024
# look at this link for reference: https://www.macquarie.com/hk/en/about/company/macquarie-asset-management/institutional/insights/battery-storage-strategies-for-revenue-stacking-and-investment-success.html

#research prices
# look into current battery installation costs 
