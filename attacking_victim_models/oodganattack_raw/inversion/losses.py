import torch.nn as nn


def get_loss(loss_name, args):
    
    if loss_name == 'l1':
        return nn.L1Loss(reduction='mean')
    elif loss_name == 'l2':
        return nn.MSELoss(reduction='mean')
    elif loss_name == 'lc':
        return nn.CrossEntropyLoss(reduction='mean')
