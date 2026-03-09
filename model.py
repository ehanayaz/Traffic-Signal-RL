import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Second hidden layer to output layer (Q-values)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the state through the network with ReLU activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # The output layer doesn't use ReLU because Q-values can be negative
        return self.fc3(x)

# Quick test to ensure PyTorch is working
if __name__ == "__main__":
    # State: 4 lanes + 1 phase = 5. Actions: Keep or Switch = 2.
    net = DQN(input_size=5, hidden_size=64, output_size=2)
    dummy_state = torch.tensor([10.0, 2.0, 0.0, 5.0, 0.0]) # Example state
    q_values = net(dummy_state)
    print(f"Network built successfully! Output Q-Values: {q_values.detach().numpy()}")
