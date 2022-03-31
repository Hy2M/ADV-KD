import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys

class Discriminator(nn.Module):
    def __init__(self, n_class, n_dense=10, n_domain=2):
        super(Discriminator, self).__init__()

        # self.linear_1 = nn.Linear(n_class, n_dense, bias=True)
        # self.linear_2 = nn.Linear(n_dense, n_domain, bias=True)

        self.FC = nn.Sequential(
            nn.Linear(n_class, n_dense),
            nn.BatchNorm1d(n_dense),
            nn.LeakyReLU(0.1),
            nn.Linear(n_dense, n_domain)
        )

    def forward(self, x):

        out = self.FC(x)

        # out = self.linear_1(x)
        # out = F.relu(out)
        # out = self.linear_2(out)

        return out