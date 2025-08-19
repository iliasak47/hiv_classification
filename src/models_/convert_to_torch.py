import torch, joblib

import torch.nn as nn
class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(7,32), nn.Linear(32,16), nn.Linear(16,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)

# Charger weights (state_dict)
m = Net(); m.load_state_dict(torch.load("models/pytorch_mlp.pt", map_location="cpu")); m.eval()
# Export TorchScript
ts = torch.jit.script(m)
ts.save("models/pytorch_mlp_ts.pt")
print("Saved models/pytorch_mlp_ts.pt")
