import traci
import time
import torch
import numpy as np
from model import DQN

# 1. Initialize the Brain and load your trained weights
state_size = 5
action_size = 2
model = DQN(state_size, 64, action_size)
model.load_state_dict(torch.load("traffic_brain.pth"))
model.eval() # Put the network in evaluation mode (no training)

# 2. Start SUMO with the visual GUI
sumo_cmd = ["sumo-gui", "-c", "single.sumocfg"]
traci.start(sumo_cmd)
tls_id = traci.trafficlight.getIDList()[0]

print("\n[INFO] AI Model Loaded. The Agent is now in full control!")

step = 0
while step < 3600: # Run for 1 hour of simulation time
    # --- GET STATE ---
    lanes = sorted(list(set(traci.trafficlight.getControlledLanes(tls_id))))
    state = [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]
    state.append(traci.trafficlight.getPhase(tls_id))
    
    # --- AI MAKES DECISION ---
    state_tensor = torch.tensor(np.array(state, dtype=np.float32))
    with torch.no_grad():
        # Pure exploitation: Just pick the action with the highest Q-value
        action = torch.argmax(model(state_tensor)).item()

    # --- EXECUTE ACTION ---
    current_phase = traci.trafficlight.getPhase(tls_id)
    if action == 1: # AI chose to switch the light
        # Yellow Light
        yellow_phase = current_phase + 1
        traci.trafficlight.setPhase(tls_id, yellow_phase)
        for _ in range(3):
            traci.simulationStep()
            step += 1
            time.sleep(0.02)
            
        # New Green Light
        next_green = 2 if current_phase == 0 else 0
        traci.trafficlight.setPhase(tls_id, next_green)

    # --- LET TRAFFIC MOVE ---
    for _ in range(10): 
        traci.simulationStep()
        step += 1
        time.sleep(0.02)

traci.close()
