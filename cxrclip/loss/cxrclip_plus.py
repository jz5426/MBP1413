"""
simpilified version of cxrclip_plusplus.py
"""

import torch
from torch.nn import functional as F
from cxrclip import util
from cxrclip.loss.baseloss import BaseLoss, all_gather
import logging
log = logging.getLogger(__name__)

class CXRClipPlus(BaseLoss):
    """
    cxrclip plus plus without additional image and text augmention
    only one text, one prompt ,and one image
    """
    def __init__(self, 
                 label_smoothing=0.0, 
                 i2i_weight=0.0, 
                 t2t_weight=0.0, 
                 loss_ratio=1.0,
                 attention_variant=None,
                 is_decoupled_loss=False,
                 enable_t2i_attention=False,
                 enable_i2t_attention=False,
                 keep_prompt_queried_image_embeddings=False,
                 asl_gamma_neg=1,
                 asl_gamma_pos=0,
                 # PLACEHOLDER
                 softmax_lambda=None,
                 multi_pos_version=None,
                 bce_lambda=None,
                 gm_pred_lambda=None,
                 multi_pos_softmax=False,
                 init_tau_aux=None,
                 min_tau_allowed_in_clamp=None,
                 max_tau_allowed_in_clamp=None,
                 is_learnable_tau_aux=None,
                 use_lambda_aux_schedule=None,
                 lambda_warmup_steps=None,
                 lambda_max=None,
                 mp_InfoNCE_function_type=None,
                 enable_weakly_dense_supervision=None,
                 weakly_dense_supervision_topk=None
                ):
        super(CXRClipPlus, self).__init__(loss_ratio)
        self.name = "contrastive"
        self.label_smoothing = label_smoothing
        self.i2i_weight = i2i_weight
        self.t2t_weight = t2t_weight

        self.is_decoupled_loss = is_decoupled_loss
        self.enable_t2i_attention = enable_t2i_attention
        self.enable_i2t_attention = enable_i2t_attention
        self.keep_prompt_queried_image_embeddings = keep_prompt_queried_image_embeddings
        self.attention_variant = attention_variant
        assert self.attention_variant in ['flair', 'gloria', None]

        self.asl_gamma_neg = asl_gamma_neg
        self.asl_gamma_pos = asl_gamma_pos