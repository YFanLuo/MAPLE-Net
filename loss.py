import torch
import torch.nn as nn
import torch.nn.functional as F


class PMRLoss(nn.Module):
    def __init__(self, lambda_ce=1.0, lambda_proto=1e-3):
        super(PMRLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_ce = lambda_ce
        self.lambda_proto = lambda_proto

    def forward(self, logits, prototypes, features, targets):
        def check_and_fix(x, name):
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: {name} contains NaN or Inf values")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            return x

        ce_loss = self.ce_loss(logits, targets)
        ce_loss = check_and_fix(ce_loss, "CE loss")

        feature_dim = features.size(-1)
        proto_dim = prototypes.size(-1)

        if feature_dim > proto_dim:
            features = features[:, :proto_dim]
        elif feature_dim < proto_dim:
            pad_size = proto_dim - feature_dim
            features = F.pad(features, (0, pad_size))

        proto_loss = -torch.log(torch.exp(-((features.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).sum(dim=2)).sum(dim=1)).mean()
        proto_loss = check_and_fix(proto_loss, "Proto loss")

        total_loss = self.lambda_ce * ce_loss + self.lambda_proto * proto_loss
        total_loss = check_and_fix(total_loss, "Total loss")

        return total_loss, ce_loss, proto_loss
