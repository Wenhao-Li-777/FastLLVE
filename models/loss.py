import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2 = epsilon * epsilon

    def forward(self, x, y):
        diff = x - y
        value = torch.sqrt(torch.pow(diff, 2) + self.epsilon2)
        return torch.mean(value)