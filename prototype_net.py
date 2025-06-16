import torch
import torch.nn as nn

class PrototypeNet(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(PrototypeNet, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x):
        logits = -((x.unsqueeze(1) - self.prototypes.unsqueeze(0))**2).sum(dim=2)
        return logits, self.prototypes