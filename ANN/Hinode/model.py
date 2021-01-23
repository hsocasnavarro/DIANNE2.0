import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np

class Neural(nn.Module):
    def __init__(self, n_stokes, n_latent, n_hidden):
        super(Neural, self).__init__()

        self.n_stokes = n_stokes
        self.n_latent = n_latent
        self.n_hidden = n_hidden

        self.C1 = nn.Linear(self.n_stokes, self.n_hidden)
        self.C2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.C3 = nn.Linear(self.n_hidden, self.n_hidden)
        self.C4 = nn.Linear(self.n_hidden, self.n_hidden)
        self.C5 = nn.Linear(self.n_hidden, self.n_hidden)
        self.C6 = nn.Linear(self.n_hidden, self.n_hidden)
        self.C7 = nn.Linear(self.n_hidden, self.n_latent)

        self.relu = nn.LeakyReLU(inplace=True)
        self.elu = nn.ELU()

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def forward(self, x):        
        out = self.C1(x)
        out = self.relu(out)
        out = self.C2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.relu(out)
        out = self.C4(out)
        out = self.relu(out)
        out = self.C5(out)
        out = self.relu(out)
        out = self.C6(out)
        out = self.relu(out)
        out = self.C7(out)
        out = self.relu(out)

        return out

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class NeuralCNN(nn.Module):
    def __init__(self, n_latent, n_hidden):
        super(NeuralCNN, self).__init__()
        
        self.n_latent = n_latent
        self.n_hidden = n_hidden

        self.P = nn.AvgPool1d(2)
    
        self.C1 = nn.Conv1d(4, n_hidden, kernel_size=11, padding=5)
        
        self.C2 = nn.Conv1d(n_hidden, n_hidden, kernel_size=7, padding=3)
        
        self.C3 = nn.Conv1d(n_hidden, n_hidden, kernel_size=5, padding=2)
        
        
        self.C4 = nn.Linear(n_hidden*21, self.n_latent)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def forward(self, x):

        # 175
        out = self.C1(x)
        out = self.relu(out)
        out = self.P(out)

        # 87
        out = self.C2(out)
        out = self.relu(out)
        out = self.P(out)

        # 43
        out = self.C3(out)
        out = self.relu(out)
        out = self.P(out)

        # 21

        # Flatten
        out = out.view(out.size(0), -1)
        out = self.C4(out)

        # Use Box-Muller transform to generate outputs that like on a Gaussian ball as assumed in the embedding
        out = torch.clamp(out, -16, 8)
        out = self.sigmoid(out)
        tmp = out.view(out.size(0), 2, out.size(1)//2)

        
        tmp1 = torch.sqrt(-2.0 * torch.log(tmp[:,0])) * torch.cos(2.0 * np.pi * tmp[:,1])
        tmp2 = torch.sqrt(-2.0 * torch.log(tmp[:,0])) * torch.sin(2.0 * np.pi * tmp[:,1])

        out = torch.cat([tmp1, tmp2], dim=1)

        return out

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
