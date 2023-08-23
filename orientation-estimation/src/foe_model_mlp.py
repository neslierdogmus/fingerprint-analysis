# import numpy as np
import torch


class hardtanh_reflected(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        return torch.nn.functional.hardtanh(torch.sign(input) * input,
                                            self.min_val,
                                            self.max_val)


class FOE_MLP(torch.nn.Module):

    def __init__(self, path, inp_dim, out_dim, device):
        super().__init__()

        self.path = path
        self.device = device
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(inp_dim * 32, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(True),
            # torch.nn.Linear(128, 16),
            # torch.nn.BatchNorm1d(16),
            # torch.nn.ReLU(True),
            # torch.nn.Linear(16, 8),
            # torch.nn.BatchNorm1d(8),
            # torch.nn.ReLU(True),
            torch.nn.Linear(128, out_dim),
            # hardtanh_reflected(0, np.pi)
            torch.nn.ReLU(True)
        )

        self = self.to(device)
        self.init_weights()

    def __str__(self):
        return "MLP"

    def forward(self, x):
        x = self.layers(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if type(m) == torch.nn.Linear:
                # torch.nn.init.normal_(m.weight, mean=0.0, std=0.062)
                torch.nn.init.uniform_(m.weight, a=-0.06, b=0.06)
