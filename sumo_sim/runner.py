import os
import sys
import traci
import numpy as np
from collections import deque

# --- SUMO CONFIGURATION ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

sumo_binary = "sumo-gui"
config_file = "intersection.sumocfg"

# --- Q-LEARNING AGENT CONFIGURATION ---
# State space: [queue_NS, queue_EW] -> Discretized into 10 bins each
# Action space: 2 actions -> [NS_Green, EW_Green]
q_table = np.zeros((10, 10, 2))

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.95     # Discount factor
epsilon = 0.05   # Exploration rate
decay = 0.999    # Epsilon decay rate

# --- SIMULATION PARAMETERS ---
GREEN_PHASE_DURATION = 10
YELLOW_PHASE_DURATION = 4

# --- HELPER FUNCTIONS ---
def get_state():
    """Retrieves discretized queue length for NS and EW directions."""
    ns_detectors = ["det_N_0", "det_N_1", "det_S_0", "det_S_1"]
    ew_detectors = ["det_E_0", "det_E_1", "det_W_0", "det_W_1"]
    
    ns_queue = sum(traci.lanearea.getLastStepHaltingNumber(det) for det in ns_detectors)
    ew_queue = sum(traci.lanearea.getLastStepHaltingNumber(det) for det in ew_detectors)
    
    # Discretize queue length into 10 bins
    ns_state = min(ns_queue // 5, 9)
    ew_state = min(ew_queue // 5, 9)
    
    return ns_state, ew_state, ns_queue + ew_queue

def choose_action(ns_state, ew_state):
    """Chooses an action using epsilon-greedy policy."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice([0, 1])  # Explore
    else:
        return np.argmax(q_table[ns_state, ew_state, :])  # Exploit

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    sumo_cmd = [sumo_binary, "-c", config_file]
    traci.start(sumo_cmd)
    
    current_step = 0
    last_action_step = 0
    total_reward = 0
    
    # Get initial state
    ns_state, ew_state, _ = get_state()
    
    # The agent will make a decision every (GREEN + YELLOW) duration
    decision_interval = GREEN_PHASE_DURATION + YELLOW_PHASE_DURATION
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            current_step += 1
            
            # Agent makes a decision at each interval
            if current_step - last_action_step >= decision_interval:
                # 1. Get current state and calculate reward from the last action period
                new_ns_state, new_ew_state, total_queue = get_state()
                reward = -total_queue  # Reward is the negative of the total queue
                total_reward += reward
                
                # 2. Choose a new action
                action = choose_action(new_ns_state, new_ew_state)

                # 3. Update Q-Table
                old_value = q_table[ns_state, ew_state, action]
                next_max = np.max(q_table[new_ns_state, new_ew_state, :])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[ns_state, ew_state, action] = new_value

                # 4. Apply the new action (set the green phase)
                # Phase 0 is NS_Green, Phase 2 is EW_Green
                traci.trafficlight.setPhase("J1", action * 2)

                # 5. Update state and timers
                ns_state, ew_state = new_ns_state, new_ew_state
                last_action_step = current_step
                
                # Decay epsilon to reduce exploration over time
                epsilon = max(0.01, epsilon * decay)
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        
    finally:
        print(f"Simulation finished after {current_step} steps.")
        print(f"Total reward: {total_reward}")
        print("Final Q-Table:")
        np.set_printoptions(precision=2)
        print(q_table)
        traci.close()