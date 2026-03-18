import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BESS_Optimiser:
    def __init__(self, prices_df, battery_params, current_soh=1.0, last_soc=None):
        self.prices_df = prices_df.sort_index()
        self.time_steps = prices_df.index
        self.dt = battery_params.get('time_interval', 0.5)
        self.p_max = battery_params.get('max_power', 10)
        self.rte = battery_params.get('rte', 0.9)
        
        self.current_capacity = battery_params.get('capacity', 20) * current_soh
        self.soc_min = battery_params.get('soc_min_factor', 0.1) * self.current_capacity
        self.soc_max = battery_params.get('soc_max_factor', 0.9) * self.current_capacity

        self.soc_initial = last_soc if last_soc is not None else (battery_params.get('soc_initial_factor', 0.5) * self.current_capacity)
        self.cycle_limit = battery_params.get('cycle_limit', 1.1)
        self.big_M = self.p_max * 1.1 
        self.all_prices = self.prices_df['DayAhead'].to_dict()

        self.model = pulp.LpProblem("BESS_DayAhead_Optimisation", pulp.LpMaximize)

    def define_variables(self):
        self.charge = pulp.LpVariable.dicts("charge", self.time_steps, lowBound=0, upBound=self.p_max)
        self.discharge = pulp.LpVariable.dicts("discharge", self.time_steps, lowBound=0, upBound=self.p_max)
        self.is_charging = pulp.LpVariable.dicts("is_charging", self.time_steps, cat='Binary')
        self.is_discharging = pulp.LpVariable.dicts("is_discharging", self.time_steps, cat='Binary')
        self.soc = pulp.LpVariable.dicts("soc", self.time_steps, lowBound=self.soc_min, upBound=self.soc_max)
        self.discharge_energy = pulp.LpVariable.dicts("discharge_energy", self.time_steps, lowBound=0)
        self.soc_stress = pulp.LpVariable.dicts("soc_stress", self.time_steps, lowBound=0)

    def set_objective(self):
        arbitrage_profit = pulp.lpSum((self.discharge[t] - self.charge[t]) * self.all_prices[t] * self.dt for t in self.time_steps)
        # Internal degradation costs
        costs = pulp.lpSum((self.discharge_energy[t] * 6.50) + (self.soc_stress[t] * 12.00 * self.dt) for t in self.time_steps)
        self.model += arbitrage_profit - costs

    def set_constraints(self):
        total_discharge_energy = []
        for t in self.time_steps:
            self.model += self.charge[t] <= self.is_charging[t] * self.big_M
            self.model += self.discharge[t] <= self.is_discharging[t] * self.big_M
            self.model += self.is_charging[t] + self.is_discharging[t] <= 1
            
            t_loc = self.prices_df.index.get_loc(t)
            soc_prev = self.soc_initial if t_loc == 0 else self.soc[self.time_steps[t_loc - 1]]
            self.model += self.soc[t] == soc_prev + (self.charge[t] * self.rte * self.dt) - (self.discharge[t] / self.rte * self.dt)
            self.model += self.discharge_energy[t] == (self.discharge[t] / self.rte * self.dt)
            total_discharge_energy.append(self.discharge_energy[t])

        num_days = (len(self.time_steps) * self.dt) / 24
        self.model += pulp.lpSum(total_discharge_energy) <= (self.cycle_limit * num_days * (self.soc_max - self.soc_min))

    def solve_and_collect(self):
        self.model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[self.model.status] in ['Optimal', 'Feasible']:
            res = pd.DataFrame(index=self.time_steps)
            get_v = lambda x: x.varValue if x.varValue is not None else 0.0
            res['SOC'] = [get_v(self.soc[t]) for t in self.time_steps]
            res['Power'] = [get_v(self.discharge[t]) - get_v(self.charge[t]) for t in self.time_steps]
            res['Price'] = self.prices_df['DayAhead']
            res['Hourly_Profit'] = res['Power'] * res['Price'] * self.dt
            res['Throughput'] = [get_v(self.discharge_energy[t]) for t in self.time_steps]
            return res
        return None

    @staticmethod
    def plot_full_period(full_results_df):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        ax1.plot(full_results_df.index, full_results_df['SOC'], color='blue', label='SOC')
        ax1.set_ylabel('SOC (MWh)')
        ax1.set_title('BESS State of Charge')
        ax1.grid(True, alpha=0.3)

        ax2_twin = ax2.twinx()
        ax2.plot(full_results_df.index, full_results_df['Price'], color='black', alpha=0.3, label='Price')
        ax2_twin.fill_between(full_results_df.index, 0, full_results_df['Power'], step='post', color='green', alpha=0.5, label='Power')
        ax2.set_ylabel('Price (£/MWh)')
        ax2_twin.set_ylabel('Power (MW)')
        ax2.set_title('Market Price vs Battery Dispatch')

        ax3.plot(full_results_df.index, full_results_df['Hourly_Profit'].cumsum(), color='orange', linewidth=2)
        ax3.set_ylabel('Cumulative Profit (£)')
        ax3.set_title('Total Profit Over Time')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()