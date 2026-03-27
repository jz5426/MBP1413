
import torch.nn as nn
from cxrclip import util
import torch


class BaseLoss(nn.Module):
    def __init__(self, loss_ratio=1.0):
        super(BaseLoss, self).__init__()
        self.loss_ratio = loss_ratio


all_gather_func = util.DistAutogradAllGatherFunction(partial=False)
def all_gather(tensor):
    world_size = util.GlobalEnv.get().world_size
    if world_size > 1:
        tensor_list = all_gather_func.apply(tensor) # from each rank.
        all_tensor = torch.cat(tensor_list, 0) # concatenate the tensor into a single one.
    else:
        all_tensor = tensor
    return all_tensor