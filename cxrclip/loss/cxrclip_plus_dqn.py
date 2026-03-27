
from cxrclip import util
from cxrclip.loss.cxrclip_plus import CXRClipPlus
from cxrclip.loss.baseloss import all_gather
from torch.nn import BCEWithLogitsLoss
import torch
import logging
import torch.nn.functional as F
import math

log = logging.getLogger(__name__)

class CXRClipPlusWithDQN(CXRClipPlus):
    """
    cxrclip_plus.py with dqn
    """
    def __init__(
        self, 
        loss_type=None, 
        include_contrastive_loss_term=True, 
        **kwargs
    ):
        super(CXRClipPlusWithDQN, self).__init__(**kwargs)
        self.softmax_lambda = 1. # default, backward compatible
    def stochastic_decoupled_softmax(
            self,
            multihot_labels, 
            logit_scale_aux,
            predictions_by_label, 
            predictions_by_image, 
            predictions_by_text,
        ):
        """
        multihot_labels: shape [batch size, labels of 0/1]
        predictions_by_label: [batch size, labels, 1]
        predictions_by_image: [batch size, labels, 1]
        predictions_by_text: [batch size, labels, 1]
        predictions_by_grounding_maps: [batch size, labels, 1]
        """
        multihot_labels = all_gather(multihot_labels.contiguous()).bool()
        if predictions_by_label is not None:
            predictions_by_label = all_gather(predictions_by_label.contiguous())
        if predictions_by_image is not None:
            predictions_by_image = all_gather(predictions_by_image.contiguous())
        if predictions_by_text is not None:
            predictions_by_text = all_gather(predictions_by_text)

        def _compute_mp_aux_ce_vectorized(preds):
            """
            Vectorized computation of the MP Aux Cross Entropy loss.
            
            Args:
                preds: Tensor of shape [B, C, 1] or [B, C]
                multihot_labels: Tensor of shape [B, C] (0 or 1)
                logit_scale_aux: Scalar or 1D Tensor scaling factor
            """
            logits = preds.squeeze(-1) # Shape: [B, C]
            device = logits.device
            
            # 1. Scale logits
            scaled_logits = logits * logit_scale_aux
            
            # 2. Create Masks
            # pos_mask identifies p (target), neg_mask identifies n (candidates)
            pos_mask = multihot_labels.bool()
            neg_mask = ~pos_mask
            
            # 3. Compute LogSumExp of Negatives per image
            # We want log(sum(exp(negative_logits))). 
            # To do this safely with generic logsumexp, we set positive logits to -inf
            # so they contribute 0 to the sum of exponentials.
            neg_logits_safe = scaled_logits.clone()
            neg_logits_safe = neg_logits_safe.masked_fill(pos_mask, float('-inf'))
            
            # Shape: [B, 1] - This represents log(sum(e^n)) for each image
            log_sum_exp_neg = torch.logsumexp(neg_logits_safe, dim=1, keepdim=True)
            
            # 4. Compute Loss for Positive indices
            # Formula: Loss_p = -z_p + log(e^z_p + sum(e^z_n))
            # We use logaddexp for numerically stable log(e^a + e^b)
            # log_denominator shape: [B, C] (broadcasted)
            log_denominator = torch.logaddexp(scaled_logits, log_sum_exp_neg)
            
            # Calculate raw losses for ALL entries
            all_losses = -scaled_logits + log_denominator
            
            # 5. Filter and Average
            # We only care about losses at the positive indices
            masked_losses = all_losses * pos_mask.float()
            
            # Sum losses per image
            loss_sum_per_img = masked_losses.sum(dim=1)
            
            # Count positives per image
            num_pos_per_img = pos_mask.sum(dim=1).float()
            num_neg_per_img = neg_mask.sum(dim=1).float()
            
            # Determine valid images (Original code skips if no pos OR no neg)
            valid_img_mask = (num_pos_per_img > 0) & (num_neg_per_img > 0)
            
            if valid_img_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Average within each valid image
            # Clamp denominator to 1.0 to avoid nan on invalid images (which are filtered out next anyway)
            img_losses = loss_sum_per_img / num_pos_per_img.clamp(min=1.0)
            
            # Average across valid images
            final_loss = img_losses[valid_img_mask].mean()
            
            return final_loss

        aux_ce_func = _compute_mp_aux_ce_vectorized

        total_loss = torch.tensor(0.0, device=multihot_labels.device)
        num_loss_terms = 0

        ce_loss_by_label = torch.tensor(0.0, device=multihot_labels.device)  
        if predictions_by_image is not None:      
            ce_loss_by_label = aux_ce_func(predictions_by_image)
            total_loss += ce_loss_by_label
            num_loss_terms += 1

        # toggle either random positive masked softmax or all positives masked softmax
        ce_loss_by_image = torch.tensor(0.0, device=multihot_labels.device)
        if predictions_by_label is not None:
            ce_loss_by_image = aux_ce_func(predictions_by_label)
            total_loss += ce_loss_by_image
            num_loss_terms += 1

        ce_loss_by_report = torch.tensor(0.0, device=multihot_labels.device)
        if predictions_by_text is not None:
            ce_loss_by_report = aux_ce_func(predictions_by_text)
            total_loss += ce_loss_by_report
            num_loss_terms += 1

        return total_loss / max(num_loss_terms, 1), ce_loss_by_image, ce_loss_by_label, ce_loss_by_report


    def forward(
        self, 
        multihot_labels,
        current_step,
        steps_per_epoch,
        predictions_by_label = None,
        predictions_by_image = None,
        predictions_by_text = None,
        attention_maps=None,
        **kwargs # the rest of embeddings are here.
        ):  
        """
        infoNCE + CE loss on labels using DQN

        multihot_labels: shape [batch size, labels]
        predictions_by_label: [batch size, labels, 1]
        predictions_by_image: [batch size, labels, 1]
        predictions_by_text: [batch size, labels, 1]
        """

        auxiliary_softmax_loss = 0.
        if self.softmax_lambda > 0:
            logit_scale_aux = kwargs['logit_scale_aux']
            auxiliary_softmax_loss, il_loss, lil_loss, tl_loss = self.stochastic_decoupled_softmax(
                multihot_labels=multihot_labels, 
                logit_scale_aux=logit_scale_aux,
                predictions_by_label=predictions_by_label, 
                predictions_by_image=predictions_by_image, 
                predictions_by_text=predictions_by_text,
            )
            if 'is_train' in kwargs and kwargs['is_train']:
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "dqn/overall_auxiliary_softmax_loss", auxiliary_softmax_loss, util.GlobalEnv.get().summary_writer.global_step
                )
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "dqn/image2label_infonce", il_loss, util.GlobalEnv.get().summary_writer.global_step
                )
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "dqn/labelQbyImage2label_infonce", lil_loss, util.GlobalEnv.get().summary_writer.global_step
                )
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "dqn/text2label_infonce", tl_loss, util.GlobalEnv.get().summary_writer.global_step
                )
                util.GlobalEnv.get().summary_writer.train.add_scalar(
                    "hyperparam/logit_scale_aux", logit_scale_aux.detach().cpu().item(), util.GlobalEnv.get().summary_writer.global_step
                )

        return (auxiliary_softmax_loss).mean()