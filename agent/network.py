import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hyperparams import HIDDEN_NEURONS, K_PATHS, OBS_SIZE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, HIDDEN_NEURONS)
        self.out = nn.Linear(HIDDEN_NEURONS, K_PATHS)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

    def best_action(self, state, device):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = self.forward(state_tensor)
        return q_values.argmin().item()

if __name__ == "__main__":
    net = QNetwork()
    # Mock batch of 2 observations of OBS_SIZE
    mock_obs = torch.rand((2, OBS_SIZE), dtype=torch.float32)
    out = net(mock_obs)
    assert out.shape == (2, K_PATHS), f"Wrong output shape: {out.shape}"
    a = net.best_action(mock_obs[0].numpy(), 'cpu')
    assert 0 <= a < K_PATHS, f"Invalid action: {a}"
    print(f"QNetwork test PASSED. Output shape: {out.shape}, best_action(obs)={a}")
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")