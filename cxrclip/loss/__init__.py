from typing import Dict

from .combined_loss import CombinedLoss
from .cxrclip_plus_dqn import CXRClipPlusWithDQN
from .cxrclip_plus import CXRClipPlus

def build_loss(all_loss_config: Dict) -> CombinedLoss:
    loss_list = []

    for loss_config in all_loss_config:
        cfg = all_loss_config[loss_config]
        if cfg["loss_ratio"] == 0.0:
            continue
        elif loss_config == 'cxrclip_plus_dqn':
            loss = CXRClipPlusWithDQN(**cfg)
        else:
            raise KeyError(f"Unknown loss: {loss_config}")

        loss_list.append(loss)

    total_loss = CombinedLoss(loss_list)
    return total_loss
