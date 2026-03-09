import traci
import time
import numpy as np
from agent import DQNAgent

# Configuration
EPISODES = 50
SIMULATION_STEPS = 1000 # How long each episode lasts
ACTION_DELAY = 10 # Let the light stay on for 10 seconds before checking again

def get_state(tls_id):
    """Reads the simulation and returns [Queue_N, Queue_S, Queue_E, Queue_W, Phase]"""
    lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    state = []
    
    # We want a consistent order of lanes, so we sort them
    for lane in sorted(lanes):
        queue = traci.lane.getLastStepHaltingNumber(lane)
        state.append(queue)
        
    current_phase = traci.trafficlight.getPhase(tls_id)
    state.append(current_phase)
    
    # Return as a numpy array for the neural network
    return np.array(state, dtype=np.float32)

def get_reward(tls_id):
    """Calculates the reward based on the total queue (Negative is bad)"""
    lanes = list(set(traci.trafficlight.getControlledLanes(tls_id)))
    total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
    
    # The AI gets punished for every car waiting
    return -total_queue

def run_simulation():
    # We have 4 lanes + 1 phase = 5 state inputs. 2 actions (Keep or Switch).
    agent = DQNAgent(state_size=5, action_size=2)
    
    sumo_cmd = ["sumo", "-c", "single.sumocfg", "--no-warnings"] # Using 'sumo' instead of 'sumo-gui' to train faster
    
    for episode in range(EPISODES):
        # Start or reset the simulation
        if episode == 0:
            traci.start(sumo_cmd)
        else:
            traci.load(["-c", "single.sumocfg"])
            
        tls_id = traci.trafficlight.getIDList()[0]
        step = 0
        total_reward = 0
        
        # Get the initial state
        state = get_state(tls_id)
        
        while step < SIMULATION_STEPS:
            # 1. Decide action
            action = agent.act(state)
            
            # 2. Execute action
            current_phase = traci.trafficlight.getPhase(tls_id)
            
            if action == 1: # AI chose to switch the light
                # A. Trigger the Yellow Light (Current Phase + 1)
                yellow_phase = current_phase + 1
                traci.trafficlight.setPhase(tls_id, yellow_phase)
                
                # B. Hold the Yellow Light for 3 seconds
                for _ in range(3):
                    traci.simulationStep()
                    step += 1
                
                # C. Switch to the new Green Light
                next_green = 2 if current_phase == 0 else 0
                traci.trafficlight.setPhase(tls_id, next_green)
                
            # Advance simulation to let the cars move on the Green light
            for _ in range(ACTION_DELAY):
                traci.simulationStep()
                step += 1 
            
            # 3. Observe the new state and reward
            next_state = get_state(tls_id)
            reward = get_reward(tls_id)
            total_reward += reward
            
            # Done if we reached the end of the simulation
            done = step >= SIMULATION_STEPS
            
            # 4. Remember and Learn
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            
        print(f"Episode: {episode + 1}/{EPISODES} | Total Reward (Score): {total_reward} | Epsilon: {agent.epsilon:.2f}")

    traci.close()
    
    # Save the trained brain!
    import torch
    torch.save(agent.model.state_dict(), "traffic_brain.pth")
    print("Training complete. Model saved as traffic_brain.pth!")

if __name__ == "__main__":
    run_simulation()
