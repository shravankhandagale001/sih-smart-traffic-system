import os
import sys
import traci
import numpy as np

# --- SUMO CONFIGURATION ---
# Check if the SUMO_HOME environment variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# SUMO settings
sumo_binary = "sumo-gui"  # Use "sumo" for a faster run without GUI
config_file = "intersection.sumocfg"

# --- Q-LEARNING AGENT CONFIGURATION ---
# The agent will learn a policy to choose between two actions
# Action 0: Set North-South phase to green
# Action 1: Set East-West phase to green
q_table = np.zeros((10, 10, 2))  # State space (10x10 bins) x Action space (2)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# --- SIMULATION HELPER FUNCTIONS ---
def get_state():
    """
    Retrieves the number of waiting cars on N-S and E-W approaches and maps it to a state index.
    """
    # Detector IDs for each approach direction
    ns_detectors = ["det_N_0", "det_N_1", "det_S_0", "det_S_1"]
    ew_detectors = ["det_E_0", "det_E_1", "det_W_0", "det_W_1"]

    # Get the number of halting cars for each direction
    ns_waiting_cars = sum(traci.lanearea.getLastStepHaltingNumber(det) for det in ns_detectors)
    ew_waiting_cars = sum(traci.lanearea.getLastStepHaltingNumber(det) for det in ew_detectors)

    # Discretize the number of cars into 10 bins (0-9)
    # This simplifies the state space for our Q-table
    ns_state = min(ns_waiting_cars // 5, 9) # Divide by 5 to create bins of size 5
    ew_state = min(ew_waiting_cars // 5, 9)

    return ns_state, ew_state

def choose_action(ns_state, ew_state):
    """
    Chooses an action based on the Q-table (exploit) or randomly (explore).
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])  # Explore: choose a random action
    else:
        return np.argmax(q_table[ns_state, ew_state, :])  # Exploit: choose the best known action

def apply_action(action, current_phase):
    """
    Applies the chosen action to the traffic light.
    Handles the yellow phase transition.
    """
    # Action 0 corresponds to Green Phase 0 (N-S)
    # Action 1 corresponds to Green Phase 2 (E-W)
    target_green_phase = action * 2

    # If the light is not already on the target green phase, transition through yellow
    if current_phase != target_green_phase:
        # Set the corresponding yellow phase (phase + 1)
        traci.trafficlight.setPhase("J1", current_phase + 1)
        # Wait for the yellow phase duration (4 seconds)
        for _ in range(4):
            traci.simulationStep()

    # Set the target green phase
    traci.trafficlight.setPhase("J1", target_green_phase)
    # Wait for a minimum green time (10 seconds)
    for _ in range(10):
        traci.simulationStep()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Start the SUMO simulation
    sumo_cmd = [sumo_binary, "-c", config_file]
    traci.start(sumo_cmd)

    # Main learning loop
    step = 0
    total_reward = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        # Get the current state
        ns_state, ew_state = get_state()
        
        # Calculate the reward (negative of total waiting cars)
        reward = -(ns_state + ew_state)
        total_reward += reward

        # Choose an action
        action = choose_action(ns_state, ew_state)

        # Apply the action and run simulation for a step
        current_phase = traci.trafficlight.getPhase("J1")
        apply_action(action, current_phase)

        # Get the new state after the action
        next_ns_state, next_ew_state = get_state()

        # Update the Q-table using the Bellman equation
        old_value = q_table[ns_state, ew_state, action]
        next_max = np.max(q_table[next_ns_state, next_ew_state, :])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[ns_state, ew_state, action] = new_value

        step += 1
    
    # End of simulation
    print(f"Simulation finished after {step} steps.")
    print(f"Total reward: {total_reward}")
    traci.close()