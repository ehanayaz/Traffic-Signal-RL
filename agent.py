import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import DQN # Importing the Brain we just built

# Hyperparameters
MAX_MEMORY = 10000
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.95 # Discount rate: How much it cares about future rewards vs immediate rewards

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # The "Diary": Stores past experiences (State, Action, Reward, Next_State, Done)
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Exploration vs. Exploitation
        self.epsilon = 1.0        # Start with 100% random exploration
        self.epsilon_min = 0.01   # Always explore at least 1% of the time
        self.epsilon_decay = 0.995 # Decay rate per training step
        
        # Initialize the Neural Network (The Brain)
        self.model = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss() # Mean Squared Error loss

    def remember(self, state, action, reward, next_state, done):
        """Saves a memory into the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Decides whether to explore (random) or exploit (use model)."""
        if random.uniform(0, 1) <= self.epsilon:
            # EXPLORE: Pick a random action (0 or 1)
            return random.randrange(self.action_size)
        
        # EXPLOIT: Ask the Neural Network
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad(): # Don't calculate gradients just for acting
            q_values = self.model(state_tensor)
            
        # Return the action with the highest Q-value
        return torch.argmax(q_values).item()

    def replay(self):
        """Trains the neural network using past memories."""
        # Don't start training until we have enough memories to form a batch
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Grab a random batch of past experiences
        minibatch = random.sample(self.memory, BATCH_SIZE)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            
            # The target is the immediate reward
            target = reward
            
            # If the episode didn't end, add the discounted future reward
            if not done:
                target = reward + GAMMA * torch.max(self.model(next_state_tensor)).item()
            
            # Get the current Q-values the model predicts
            current_q = self.model(state_tensor)
            
            # Update the specific action's Q-value to match our new target
            target_q = current_q.clone()
            target_q[action] = target
            
            # Backpropagation (Train the network)
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
            
        # Slowly reduce exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    agent = DQNAgent(state_size=5, action_size=2)
    dummy_state = [10, 2, 0, 5, 0] # Example queue lengths and phase
    
    action = agent.act(dummy_state)
    print(f"Agent initialized! Chose action: {action} (Epsilon is {agent.epsilon:.2f}, so it's random)")
