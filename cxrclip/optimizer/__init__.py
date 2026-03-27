import logging
from typing import Dict
import torch
from torch import nn
from cxrclip import util

log = logging.getLogger(__name__)

def build_optimizer(model: nn.Module, optim_config: Dict):

    no_decay = [name.lower() for name in optim_config.get("no_decay", [])]
    if no_decay:
        wd = optim_config["config"]["weight_decay"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd.lower() in n.lower() and 'projection' not in n.lower() and 'dqn' not in n.lower() for nd in no_decay)], "weight_decay": wd}, # dqn and projection has decay
            {"params": [p for n, p in model.named_parameters() if any(nd.lower() in n.lower() and 'projection' not in n.lower() and 'dqn' not in n.lower() for nd in no_decay)], "weight_decay": 0.0}, # the rest of the layers has not decay
        ]
        # make sure the parameters are exhaustive.
        assert len(params[0]['params'] + params[1]['params']) == len(list(model.named_parameters()))
        if util.GlobalEnv.get().local_rank < 1:
            log.info("seperated no decay params (#params:%d no-decay #params:%d)", len(params[0]["params"]), len(params[1]["params"]))
            no_decay_list = [n for n, _ in model.named_parameters() if any(nd.lower() in n.lower() and 'projection' not in n.lower() and 'dqn' not in n.lower() for nd in no_decay)]
            log.info('List of parameters with no decay:')
            for n in no_decay_list:
                log.info(f'  {n}')
    else:
        params = model.parameters()

    optim_name = optim_config["name"].lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(params, **optim_config["config"])
    else:
        raise NotImplementedError(f"Not implemented optimizer : {optim_name}")
    return optimizer

