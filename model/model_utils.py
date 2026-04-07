import torch
import torch.nn as nn

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


def get_class_grw_weight(class_weight, exp_scale=0.2):
    """
    Caculate the Generalized Re-weight for Loss Computation
    """

    ratio = 1 / class_weight

    class_weight = 1 / (ratio**exp_scale)
    class_weight = class_weight / torch.sum(class_weight) * len(class_weight)
    return class_weight

def get_loss(loss_func_name, outputs, targets, weights=None):
    weights = weights.to(outputs.device) if weights is not None else None
    if loss_func_name == "wmse":
        return calc_wmse_loss(outputs, targets, weights)
    elif loss_func_name == "ewmse":
        eweights = get_class_grw_weight(weights)
        return calc_wmse_loss(outputs, targets, eweights)
    elif loss_func_name == "mse":
        return calc_mse_loss(outputs, targets)
