from typing import List

import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, loss_list: List[nn.Module]):
        super(CombinedLoss, self).__init__()
        self.loss_list = loss_list

    def forward(self, **kwargs):
        loss_dict = dict()
        total_loss = 0.0
        for loss in self.loss_list:
            cur_loss = loss(**kwargs)
            # a specific loss
            loss_dict[loss.name] = cur_loss
            total_loss += cur_loss * loss.loss_ratio

        # manually added the total loss.
        loss_dict["total"] = total_loss
        return loss_dict
