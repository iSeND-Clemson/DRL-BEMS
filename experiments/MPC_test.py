# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:39:58 2023

@author: hao9
"""

import numpy as np
import cvxpy as cp

# Define system parameters
n_zones = 6  # Number of zones
n_steps_per_hour = 12  # Number of time steps in one hour (5-minute intervals)
n_days = 1  # Number of days
n_hours_per_day = 24  # Number of hours per day
n_total_steps = n_days * n_hours_per_day * n_steps_per_hour  # Total number of steps
n_variables = 14  # Number of state variables

# Generate random system dynamics (simplified for demonstration)
A = np.random.rand(n_variables, n_variables)  # Random state transition matrix
B = np.random.rand(n_variables, n_zones)   # Random input matrix

# Define optimization variables
u = cp.Variable((n_zones, n_total_steps))  # Control inputs (VAV on/off status)
x = cp.Variable((n_variables, n_total_steps + 1))  # State variables

# Generate random initial system states for demonstration
initial_states = np.random.rand(n_variables)

# Energy balance equation terms
h = np.random.rand(n_zones, n_zones)  # Heat transfer coefficients between zones
A_zone = np.random.rand(n_zones, n_zones)  # Area of contact/interface between zones
m_dot = np.random.rand(n_zones)  # Air mass flow rates for each zone
Cp = np.random.rand(n_zones)  # Specific heat capacity of air for each zone
Thvac = np.random.rand(n_zones)  # Temperature of HVAC supply air for each zone

# MPC control loop
for t in range(10):
    # Get the current system states (temperature, humidity, etc.) as input
    current_states = x[:, t]
        

    # Define a simple objective function (minimize the sum of squared control inputs)
    objective = cp.Minimize(cp.sum_squares(u[:, t]))



    # Define constraints (e.g., temperature limits, VAV constraints, etc.) - Simplified for demonstration
    constraints = [
        current_states[2:8] >= 20,  # Temperature lower bound
        current_states[2:8] <= 24,  # Temperature upper bound
        u[:, t] >= 0,  # VAV lower bound
        u[:, t] <= 1,  # VAV upper bound
    ]



    # Energy balance equation
    energy_balance_constraints = []
    for i in range(n_zones):
        delta_Q = (
            cp.sum(cp.multiply(h[i, :], cp.multiply(A_zone[i, :], current_states[2:8] - current_states[2:8][i]))) +
            m_dot[i] * Cp[i] * (Thvac[i] - current_states[2:8][i])
        )
        energy_balance_constraints.append(x[2:8, t + 1][i] == x[2:8, t][i] + delta_Q)  # Energy balance equation



    # Add energy balance constraints to the list of constraints
    constraints.extend(energy_balance_constraints)



    # Define and solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()



    # Extract and output the optimal control input (VAV status) for the current time step
    optimal_u = u[:, t].value
    
    print(f"Time Step {t + 1}: VAV Control Signal - {optimal_u}")

# The optimal_u array represents the control input (VAV status) for each 5-minute interval










import numpy as np
import cvxpy as cp

# Define system parameters
n_zones = 6  # Number of zones
n_steps_per_hour = 12  # Number of time steps in one hour (5-minute intervals)
n_days = 1  # Number of days
n_hours_per_day = 24  # Number of hours per day
n_total_steps = n_days * n_hours_per_day * n_steps_per_hour  # Total number of steps
n_variables = 14  # Number of state variables

# Generate random system dynamics (simplified for demonstration)
A = np.random.rand(n_variables, n_variables)  # Random state transition matrix
B = np.random.rand(n_variables, n_zones)   # Random input matrix

# Define optimization variables
u = cp.Variable((n_zones, 1), boolean=True)  # Control inputs (VAV on/off status)
x = cp.Variable((n_variables, 1))  # State variables
T = cp.Variable((6,1))  # Temperatures for 6 zones

# Generate random initial system states for demonstration
initial_states = np.random.rand(n_variables)

# Energy balance equation terms
h = np.random.rand(n_zones, n_zones)  # Heat transfer coefficients between zones
A_zone = np.random.rand(n_zones, n_zones)  # Area of contact/interface between zones
m_dot = np.random.rand(n_zones)  # Air mass flow rates for each zone
Cp = np.random.rand(n_zones)  # Specific heat capacity of air for each zone
Thvac = np.random.rand(n_zones)  # Temperature of HVAC supply air for each zone

airflow_rate = 0.5  # m³/s
specific_heat_capacity_air = 1005  # J/kg°C

    
# MPC control loop
for t in range(10):
    # Get the current system states (temperature, humidity, etc.) as input
    current_states = x
    
    T = np.random.rand(6)


    E_HVAC_expressions = []
    
    for i in range(T.size):

        # Define piecewise linear functions for cooling and heating energy
        temperature_difference_cooling = u[i] * cp.maximum(68 - T[i], 0)
        cooling_energy = airflow_rate * specific_heat_capacity_air * temperature_difference_cooling
        
        temperature_difference_heating = u[i] * cp.maximum(T[i] - 77, 0)
        heating_energy = 3 * airflow_rate * specific_heat_capacity_air * temperature_difference_heating
        
        E_HVAC_expressions.append(cooling_energy + heating_energy)
    
    E_total = cp.sum(E_HVAC_expressions)
    
    print(E_total.value)


    # Define a simple objective function (minimize the sum of squared control inputs)
    objective = cp.Minimize(E_total)


    # Define constraints (e.g., temperature limits, VAV constraints, etc.) - Simplified for demonstration
    constraints = [
        current_states[2:8] >= 20,  # Temperature lower bound
        current_states[2:8] <= 24,  # Temperature upper bound
        u >= 0,  # VAV lower bound
        u <= 1,  # VAV upper bound
        E_total >= 100000
    ]


    # Define and solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()



    # Extract and output the optimal control input (VAV status) for the current time step
    optimal_u = u.value
    
    print(f"Time Step {t + 1}: VAV Control Signal - {optimal_u}")

# The optimal_u array represents the control input (VAV status) for each 5-minute interval



















import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Constants
control_horizon = 10  # Number of control steps
prediction_horizon = 10  # Number of prediction steps
initial_temperature = 65.0  # Initial room temperature
comfort_range = [68, 77]  # Comfort range for room temperature

# Simulated system dynamics (simplified)
def simulate_temperature(power_profile):
    temperature = [initial_temperature]
    for t in range(1, len(power_profile)):
        temperature.append(temperature[t-1] + 0.05 * power_profile[t-1])
    return temperature

# MPC Controller
class RoomTemperatureMPC:
    def __init__(self):
        self.u = cp.Variable(control_horizon)  # Control signal (heater power)
        self.T = cp.Variable(prediction_horizon + 1)  # Room temperature over prediction horizon

        # Constraints
        self.constraints = [
            self.T >= comfort_range[0],  # Lower temperature bound constraint
            self.T <= comfort_range[1],  # Upper temperature bound constraint
            cp.diff(self.T) == 0.05 * self.u,  # Temperature dynamics constraint
        ]

        # Initialize the energy cost variable and problem, but do not set the cost function yet
        self.energy_cost = cp.Variable(1)
        self.cost = cp.Minimize(self.energy_cost)  # Define the cost function here
        self.problem = cp.Problem(self.cost, self.constraints)

    def optimize(self, current_temperature):
        # Define the energy cost objective based on power consumption
        energy_cost_expression = (cp.sum(self.T) - current_temperature) *350

        # Set the energy cost objective in the optimization problem
        self.cost = cp.Minimize(energy_cost_expression)
        self.problem = cp.Problem(self.cost, self.constraints)

        # Solve the optimization problem
        self.problem.solve()

        # Apply the first control input to the system
        u_optimal = self.u.value[0]

        return u_optimal

# MPC Simulation
mpc_controller = RoomTemperatureMPC()
room_temperature_history = [initial_temperature]
control_profile = []

room_temperature = initial_temperature

for _ in range(prediction_horizon):
    # Obtain the optimal control signal from the MPC controller
    u_optimal = mpc_controller.optimize(room_temperature)
    control_profile.append(u_optimal)

    # Simulate the system using the control signal
    room_temperature = simulate_temperature([u_optimal])
    room_temperature_history.append(room_temperature[-1])


# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(room_temperature_history)
plt.axhline(y=comfort_range[0], color='r', linestyle='--', label='Comfort Range (Lower)')
plt.axhline(y=comfort_range[1], color='r', linestyle='--', label='Comfort Range (Upper)')
plt.title('Room Temperature Control')
plt.xlabel('Time Step')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(control_profile)
plt.title('Heater Power Control')
plt.xlabel('Time Step')
plt.ylabel('Heater Power')
plt.tight_layout()

plt.show()













































