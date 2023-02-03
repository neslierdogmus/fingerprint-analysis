import torch
from torch import nn

class FOEMLP(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 32),
            nn.LeakyReLU(True),
            nn.Linear(32, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 2),
            nn.Hardtanh(min_val=-1,max_val=1)
        )
        
    def forward(self, x):
        x = self.encoder_lin(x)
        return x