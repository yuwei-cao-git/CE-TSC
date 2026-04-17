import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        loss = torch.mean(squared_errors)
        return loss


# MSE loss
def calc_mse_loss(valid_outputs, valid_targets):
    mse = MSELoss()
    loss = mse(valid_outputs, valid_targets)

    return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss


def calc_wmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return loss

def calc_entropy_loss(valid_outputs, valid_targets):
    plot_entropy = -(valid_targets * torch.log(valid_targets + 1e-8)).sum(dim=1)
    weights = 1 + plot_entropy
    loss = (weights[:, None] * (valid_outputs - valid_targets) ** 2).mean()
    return loss

def get_class_grw_weight(class_weight, exp_scale=0.2):
    """
    Caculate the Generalized Re-weight for Loss Computation
    """

    ratio = 1 / class_weight

    class_weight = 1 / (ratio**exp_scale)
    class_weight = class_weight / torch.sum(class_weight) * len(class_weight)
    return class_weight

def get_loss(loss_func_name, outputs, targets, weights=None):
    if loss_func_name == "wmse":
        return calc_wmse_loss(outputs, targets, weights)
    elif loss_func_name == "ewmse":
        eweights = get_class_grw_weight(weights)
        return calc_wmse_loss(outputs, targets, eweights)
    elif loss_func_name == "mse":
        return calc_mse_loss(outputs, targets)
    elif loss_func_name == "smooth_l1":
        return F.smooth_l1_loss(outputs, targets)
    elif loss_func_name == "entropy":
        return calc_entropy_loss(outputs, targets)


def initialize_weights(m):
    # A recursive function to apply initialization to all relevant layers
    if isinstance(m, nn.Linear):
        # Kaiming/He initialization for linear layers followed by ReLU
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
