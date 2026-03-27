import logging
from typing import Dict
import torch
from torch import nn
from cxrclip.model.baseclip import AttentionBasedClip
import logging
import numpy as np
import torch.nn.functional as F
from .modules import load_projection_head
from .kad_dqn import TQN_Model
from cxrclip import util

log = logging.getLogger(__name__)

class CXRClipPlusWithDQN(AttentionBasedClip):
    """Custom forward function for cxrclip plus with attention AND DQN module. """

    def __init__(self, model_config: Dict, all_loss_config: Dict, config: Dict={}, tokenizer = None):
        super(CXRClipPlusWithDQN, self).__init__(model_config, all_loss_config, tokenizer)
        self.temperature = model_config["temperature"] if "temperature" in model_config else None
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            log.warning("[CXRCLIP] missing temperature scaling factor")

        self.logit_scale_aux = None
        if 'cxrclip_plus_dqn' in all_loss_config:
            self.init_tau_aux = all_loss_config['cxrclip_plus_dqn']['init_tau_aux'] if 'init_tau_aux' in all_loss_config['cxrclip_plus_dqn'] else 1.
            self.min_tau_allowed_in_clamp = all_loss_config['cxrclip_plus_dqn']['min_tau_allowed_in_clamp'] if 'min_tau_allowed_in_clamp' in all_loss_config['cxrclip_plus_dqn'] else None        
            self.max_tau_allowed_in_clamp = all_loss_config['cxrclip_plus_dqn']['max_tau_allowed_in_clamp'] if 'max_tau_allowed_in_clamp' in all_loss_config['cxrclip_plus_dqn'] else None 
            self.logit_scale_aux = nn.Parameter(
                torch.ones([]) * np.log(1.0 / self.init_tau_aux), 
                requires_grad=all_loss_config['cxrclip_plus_dqn']['is_learnable_tau_aux'] if 'is_learnable_tau_aux' in all_loss_config['cxrclip_plus_dqn'] else False
            )
            self.is_learnable_tau_aux = all_loss_config['cxrclip_plus_dqn']

        # local embedding projections., from model/filip.py
        if self.projection:

            # vision
            try:
                if self.vision_dual_cls: # use prompt token classifier if defined
                    self.image_patch_projection = load_projection_head(
                        embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                    ) if model_config['projection_head']['seperate_local_projection_head'] else self.image_cls2_projection
                else: # use text report classifier.
                    self.image_patch_projection = load_projection_head(
                        embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                    ) if model_config['projection_head']['seperate_local_projection_head'] else self.image_cls_projection
            except:
                # backward compatible when the key seperate_local_projection_head does not exists in the ckpt checkpoint.
                self.image_patch_projection = load_projection_head(
                    embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
                )

            # text
            try:
                self.word_projection = load_projection_head(
                    embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
                ) if model_config['projection_head']['seperate_local_projection_head'] else self.text_cls_projection
            except:
                # backward compatible when the key seperate_local_projection_head does not exists in the ckpt checkpoint.
                self.word_projection = load_projection_head(
                    embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
                )

        self.enable_t2i_attention = self.loss_config['cxrclip_plus_dqn']['enable_t2i_attention']
        self.enable_i2t_attention = self.loss_config['cxrclip_plus_dqn']['enable_i2t_attention']
        self.keep_prompt_queried_image_embeddings = self.loss_config['cxrclip_plus_dqn']['keep_prompt_queried_image_embeddings']
        self.attention_variant = self.loss_config['cxrclip_plus_dqn']['attention_variant']

        # image related tokens for cross attention.
        self.use_all_image_tokens = self.model_config['dqn_image_tokens']['use_all_image_tokens'] if 'dqn_image_tokens' in self.model_config else True

        if self.attention_variant is None:
            assert self.enable_t2i_attention == False and self.enable_i2t_attention == False, "integrity constraints problems."

        self.enable_4d_descriptive_dqn_forwarding = (
            model_config['enable_4d_descriptive_dqn_forwarding'] if 'enable_4d_descriptive_dqn_forwarding' in model_config 
            else False
        )
        self.enable_enriched_embeddings_as_input_to_dqn = (
            model_config['enable_enriched_embeddings_as_input_to_dqn'] if 'enable_enriched_embeddings_as_input_to_dqn' in model_config 
            else False
        )
        assert self.enable_enriched_embeddings_as_input_to_dqn == False, 'This should be false to match the inference distribution.'

        try:
            self.classification_head = model_config['classification_head']
        except:
            self.classification_head = 'dqn'
        assert self.classification_head in ['dqn']

        self.enable_dqn_forward_pass = False
        if self.classification_head == 'both' or self.classification_head == 'dqn':
            self.enable_dqn_forward_pass = True    

        self.dqn_fusion_type = model_config['dqn_fusion_type'] if 'dqn_fusion_type' in model_config else None
        assert self.dqn_fusion_type is not None, "DQN should not be None is this class."
        if self.dqn_fusion_type == 'dqn_kad' and self.enable_dqn_forward_pass:
            self.dqn = TQN_Model(
                logit_scale=self.logit_scale,
                nhead=model_config['dqn_nheads'] if 'dqn_nheads' in model_config else 4,
                nlayers=model_config['dqn_nlayers'] if 'dqn_nlayers' in model_config else 4,
                embed_dim=model_config['projection_head']['proj_dim'],
                text_query_noise_std=model_config['text_query_noise_std'] if 'text_query_noise_std' in model_config else 0.,
                # normalize_before=not self.enable_4d_descriptive_dqn_forwarding,
                allow_self_attention=model_config['dqn_allow_self_attention'] if 'dqn_allow_self_attention' in model_config else True, # default is True
                use_mlp=model_config['use_dqn_mlp_head'] if 'use_dqn_mlp_head' in model_config else False,
                allow_disease_information_leakage_in_dqn=model_config['allow_disease_information_leakage_in_dqn'] if 'allow_disease_information_leakage_in_dqn' in model_config else False,
                enable_dqn_fewshot=model_config.get('enable_dqn_fewshot', False),
                unlock_delta=model_config.get('unlock_delta', False),
                unlock_multi_mlp_heads=model_config.get('unlock_multi_mlp_heads', False),
                delta_input_query_num=config['data_train']['base']['n_class'] if len(config) > 0 and 'base' in config['data_train'] else -1 # NOTE: -1 should crash the training (expected)
            )
        self.enable_dqn_prediction_by_image_query = model_config['enable_dqn_prediction_by_image_query'] if 'enable_dqn_prediction_by_image_query' in model_config else False
        self.enable_dqn_prediction_by_text_report = model_config['enable_dqn_prediction_by_text_report'] if 'enable_dqn_prediction_by_text_report' in model_config else True # NOTE: intentional

        self.use_last_n_layer_features = model_config['use_last_n_layer_features'] if 'use_last_n_layer_features' in model_config else [-1]
        assert len(set(self.use_last_n_layer_features)) == len(self.use_last_n_layer_features), "confusing, duplicate layer features indicated."
        assert len(self.use_last_n_layer_features) == 1, "this option no longer available as deep dqn is implemented. please toggle the deep_dqn_access_layers option"

        self.vision_features_miggle = model_config['operation'] if 'operation' in model_config else 'mean'
        
        # for text, the default miggle operation is sum instead of mean
        self.text_use_last_n_layer_features = list(model_config['text_use_last_n_layer_features']) if 'text_use_last_n_layer_features' in model_config else [-1]
        assert len(set(self.text_use_last_n_layer_features)) == len(self.text_use_last_n_layer_features), "confusing, duplicate layer features indicated."
        self.text_features_miggle = model_config['text_operation'] if 'text_operation' in model_config else 'sum'
        assert self.text_features_miggle == 'sum', 'text features should be sum'
        self.text_cls_type = model_config['cls_type'] if 'cls_type' in model_config else 'cls'
        assert self.text_cls_type in ['patch', 'cls'], 'cls type for text should either be "cls" or "patch".'
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        try:
            self.text_include_cls_token_to_patch_aggregation = model_config['include_cls_token_to_patch_aggregation']
        except:
            self.text_include_cls_token_to_patch_aggregation = False

        # integrity constraints
        if len(self.text_use_last_n_layer_features) > 1:
            assert self.text_cls_type == "patch", "multi-layers for text only suppose 'patch' based features not 'cls'."

    def enrich_embeddings_by_attention(
        self,
        # CLS/CLS2 embeddings
        image_embeddings_for_text, 
        image_embeddings_for_prompt,
        text_embeddings, 
        prompt_embeddings,
        # Patch-level embeddings
        image_patch_embeddings_for_text,
        text_patch_embeddings,
        # Others
        text_attention_mask,
        **kwarg
        ):
    
        enhanced_image_embeddings_for_text = image_embeddings_for_text 
        enhanced_image_embeddings_for_prompt = image_embeddings_for_prompt
        enhanced_text_embeddings_for_image = text_embeddings

        # enrich the image embeddings based on text embeddings
        if self.attention_variant and self.enable_t2i_attention:

            # text-guided vision embeddings for the original image
            enhanced_image_embeddings_for_text = self.flair_attention(
                text_embeddings.unsqueeze(1), 
                torch.cat([image_embeddings_for_text.unsqueeze(1), image_patch_embeddings_for_text], dim=1),
                None
            )

            if self.keep_prompt_queried_image_embeddings and prompt_embeddings is not None:
                # log.warning("[CXRCLIP_PLUSPLUS] USING PROMPTED-QUERIED IMAGE EMBEDDINGS.")
                # prompt-guided for the original image
                enhanced_image_embeddings_for_prompt = self.flair_attention(
                    prompt_embeddings.unsqueeze(1),
                    torch.cat([image_embeddings_for_prompt.unsqueeze(1), image_patch_embeddings_for_text], dim=1),
                    None
                )

        # enrich the text embedding based on the images
        if self.attention_variant and self.enable_i2t_attention:

            # original image-guided text embeddings
            enhanced_text_embeddings_for_image = self.flair_attention(
                image_embeddings_for_text.unsqueeze(1), 
                torch.cat([text_embeddings.unsqueeze(1), text_patch_embeddings], dim=1),
                text_attention_mask.unsqueeze(-1)
            )

        # assert enhanced_image_embeddings_for_prompt is None and enhanced_text_embeddings_for_image is None
        return {
            "enhanced_image_embeddings_for_text": enhanced_image_embeddings_for_text, # query is text, and create weighted sum of the image embeddings
            "enhanced_image_embeddings_for_prompt": enhanced_image_embeddings_for_prompt,
            "enhanced_text_embeddings_for_image": enhanced_text_embeddings_for_image,
        }

    def custom_project_vision_cls_features(self, image_last_cls_features, hidden_states):
        assert len(hidden_states) <= 4, 'only last 4 layers of hidden states is stored. otherwise out of gpu memory'
        if self.use_last_n_layer_features == [-1]:
            image_cls_embeddings_raw, image_cls_embeddings_for_text = self._project_and_normalize(image_last_cls_features, self.image_cls_projection, return_raw_and_normed=True)
            image_cls_embeddings_raw, image_cls_embeddings_for_text = [image_cls_embeddings_raw], [image_cls_embeddings_for_text]
        else:
            image_cls_embeddings_raw, image_cls_embeddings_for_text= [], []
            for i in range(len(hidden_states)):
                layer_cls_feat_raw, layer_cls_feat_normed  = self._project_and_normalize(hidden_states[i][:, :1, :], self.image_cls_projection, return_raw_and_normed=True)
                image_cls_embeddings_raw.append(layer_cls_feat_raw)
                image_cls_embeddings_for_text.append(layer_cls_feat_normed)

        return image_cls_embeddings_raw, image_cls_embeddings_for_text

    def custom_project_vision_patch_features(self, image_last_patch_features, hidden_states):
        assert len(hidden_states) <= 4, 'only last 4 layers of hidden states is stored. otherwise out of gpu memory'
        if self.use_last_n_layer_features == [-1]:
            image_patch_embeddings_raw, image_patch_embeddings_for_text = self._project_and_normalize(image_last_patch_features, self.image_patch_projection, return_raw_and_normed=True)
            image_patch_embeddings_raw, image_patch_embeddings_for_text = [image_patch_embeddings_raw], [image_patch_embeddings_for_text]
        else:
            
            image_patch_embeddings_raw, image_patch_embeddings_for_text= [], []
            for i in range(len(hidden_states)):
                layer_patch_feat_raw, layer_patch_feat_normed  = self._project_and_normalize(hidden_states[i][:, 1:, :], self.image_patch_projection, return_raw_and_normed=True)
                image_patch_embeddings_raw.append(layer_patch_feat_raw)
                image_patch_embeddings_for_text.append(layer_patch_feat_normed)
        return image_patch_embeddings_raw, image_patch_embeddings_for_text
    
    def custom_project_text_cls_token(self, last_cls_token, hidden_states):
        assert len(hidden_states) <= 4, 'only last 4 layers of hidden states is stored. otherwise out of gpu memory'
        if self.text_use_last_n_layer_features == [-1]:
            # the combine operation does not matter
            cls_embeddings_raw, cls_embeddings = self._project_and_normalize(last_cls_token, self.text_cls_projection, return_raw_and_normed=True)
        else:  
            # extract per layer vision features
            
            cls_features_stacked = torch.stack([hidden_states[i][:, 0, :] for i in range(len(hidden_states))])
            # cls_features_stacked = torch.stack([hidden_states[i][:, 0, :] for i in self.text_use_last_n_layer_features])
            
            # combine layer-wise word features via text_features_miggle
            if self.text_features_miggle == 'mean':
                # Sum across layers and divide by number of layers (standard mean)
                # Since padding is zeroed, it remains zero.
                cls_features_aggregated = cls_features_stacked.mean(dim=0)
            elif self.text_features_miggle == 'sum':
                cls_features_aggregated = cls_features_stacked.sum(dim=0)
            else:
                assert False, "Non supported patch feature miggle operation."

            # project the features
            cls_embeddings_raw, cls_embeddings = self._project_and_normalize(cls_features_aggregated.unsqueeze(1), self.text_cls_projection, return_raw_and_normed=True)
            assert False, 'Should not be here.'

        return cls_embeddings_raw, cls_embeddings

    def aggregate_tokens(self, embeddings, caption_ids):

        embeddings = embeddings.permute(0, 2, 1, 3)
        sentence_embeddings, sentences = [], []
       
        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]": # end of sentence
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    # add back the [SEP] token
                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                elif word.startswith("##"): # non-beginning partial words
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])
            
            # Handle any remaining token in the bank (e.g., if sentence didn't end with SEP)
            if len(token_bank) > 0:
                new_emb = torch.stack(token_bank).sum(axis=0)
                agg_embs.append(new_emb)
                words.append("".join(word_bank)) # <<<< RESTORED

            # --- AGGREGATION STEP PER SENTENCE WITHOUT PADDINGS TOKENS ---
            if len(agg_embs) > 0:
                agg_embs_tensor = torch.stack(agg_embs)
                # always average word tokens per layer
                sentence_embeddings.append(agg_embs_tensor.mean(dim=0))
                sentences.append(words) # <<<< RESTORED (Add list of words to batch list)

        # Stack the batch
        sentence_embeddings = torch.stack(sentence_embeddings)
        
        return sentence_embeddings, sentences

    def custom_project_word_features(self, word_features, hidden_states, attention_mask, ids):
        """
        word_features: of shape [number of query label, number of tokens per query label exclude cls token, feature dimension per token]
        hidden_states: of shape [number of query label, number of tokens per query label + cls token, feature dimension per token]
        attention_mask: of shape [number of query label, number of tokens per query label + cls token]
        """
        assert len(hidden_states) <= 4, 'only last 4 layers of hidden states is stored. otherwise out of gpu memory'

        if self.text_use_last_n_layer_features == [-1]:
            word_embeddings_raw, word_embeddings = self._project_and_normalize(word_features, self.word_projection, return_raw_and_normed=True)
            # the combine operation does not matter
        else:  
            embeddings = torch.stack(hidden_states)  # layers, batch, sent_len, embedding size
            embeddings = embeddings.permute(1, 0, 2, 3)

            sent_embeddings, sents = self.aggregate_tokens(embeddings, ids)

            if self.text_features_miggle == "sum":
                sent_embeddings = sent_embeddings.sum(dim=1)
            elif self.text_features_miggle == "mean":
                sent_embeddings = sent_embeddings.mean(dim=1)
            else:
                print(self.text_features_miggle)
                raise Exception("Aggregation method not implemented")

            # NOTE: NO PROJECTION FOR THIS
            word_embeddings_raw = sent_embeddings # shape [number of labels, feature dimension]
            word_embeddings = sent_embeddings / torch.norm(sent_embeddings, 2, dim=1, keepdim=True).expand_as(sent_embeddings)

        return word_embeddings_raw, word_embeddings

    def aggregate_valid_word_feature_representation(self, word_features, full_attention_mask):
        """
        aggregate by averaging the word feature representation excluding the padding tokens
        """
        # using word features aggregate as cls token
        if not self.text_include_cls_token_to_patch_aggregation:
            full_attention_mask = full_attention_mask[:, 1:]
            word_features = word_features[:, 1:, :]
            assert full_attention_mask.shape[1] == word_features.shape[1], "shape mismatch"

        valid_mask = full_attention_mask.unsqueeze(-1)
        patch_features_masked = word_features * valid_mask # => of shape [number of query, number of word tokens, feature size]
        # sum masked features over tokens
        sum_features = patch_features_masked.sum(dim=1)     # [Q, D]
        # count valid (non-pad) tokens
        valid_counts = valid_mask.sum(dim=1)                 # [Q, 1]
        # avoid division by zero
        avg_features = sum_features / (valid_counts + 1e-12)  # [Q, D]
        return avg_features

    def forward(self, batch, device=None):
        """content of the batch dictionary refers to imagetext.py"""

        device = batch["images"].device if device is None else device

        # get image and text features from the cls token (NOT CLS2)
        images = batch["images"].to(device)
        image_features = self.encode_image(images, self.use_last_n_layer_features)
        hidden_states = image_features['hidden_states'] if 'hidden_states' in image_features else None
        image_features, image_patch_features = image_features['cls_token'], image_features['patch_tokens']
        image_cls_embeddings_raw, image_cls_embeddings_for_text = self.custom_project_vision_cls_features(image_features[:, :1, :], hidden_states)
        image_patch_embeddings_raw, image_patch_embeddings_for_text = self.custom_project_vision_patch_features(image_patch_features, hidden_states)

        text_cls_embeddings, text_attention_mask, text_embeds_raw, text_patch_embeddings = None, None, None, None
        if 'text_tokens' in batch: # for the report
            text_features = self.encode_text(batch["text_tokens"].to(device)) # NOTE: intentionally always use the last layer always
            text_attention_mask = batch['text_attention_mask'].to(device)
            text_features, text_patch_features = text_features['cls_token'], text_features['patch_tokens']
            text_cls_embeddings_raw, text_cls_embeddings = self._project_and_normalize(text_features[:, :1, :], self.text_cls_projection, return_raw_and_normed=True)
            assert text_features.shape[1] == 1, 'Text encoder should have only one CLS token, shared between text and prompt'
            text_patch_embeddings_raw, text_patch_embeddings = self._project_and_normalize(text_patch_features, self.word_projection, return_raw_and_normed=True)
            text_embeds_raw = torch.cat([text_cls_embeddings_raw.unsqueeze(1), text_patch_embeddings_raw], dim=1)

        if 'prompt' in batch:
            assert image_features.shape[1] == 2, 'Vision encoder should have dual CLS tokens. One for report, one for prompt.'

        # FEATURES FOR AUGMENTED IMAGES AND TEXTS
        prompt_embeddings, image_cls_embeddings_for_prompt = None, None
        if 'prompt_tokens' in batch:
            prompt_features = self.encode_text(batch["prompt_tokens"].to(device)) # NOTE: intentionally always use the last layer
            prompt_features = prompt_features['cls_token']
            prompt_embeddings = self._project_and_normalize(prompt_features[:, :1, :], self.text_cls_projection)
            image_cls_embeddings_for_prompt = self._project_and_normalize(image_features[:, 1:2, :], self.image_cls2_projection)

        # get the query label text features
        labels, label_tokens, multihot_labels = batch['labels'], batch['label_tokens'], batch['multihot_label']
        labels = np.array(labels)
        label_features = self.encode_text(label_tokens.to(device), self.text_use_last_n_layer_features)

        if self.text_cls_type == 'patch':
            # patch features only
            label_cls_features_raw, label_cls_embeddings = self.custom_project_word_features(
                label_features['patch_tokens'], 
                label_features['hidden_states'], 
                label_tokens['attention_mask'], 
                label_tokens['input_ids']
            )
        else:
            # use cls tokens only
            label_cls_features_raw, label_cls_embeddings = self.custom_project_text_cls_token(label_features['cls_token'], label_features['hidden_states'])
        
        vanilla_features = {
            # [CLS]/[CLS2] embeddings for a single image
            "image_embeddings_for_text": image_cls_embeddings_for_text, # image embeddings for text
            "image_embeddings_for_prompt": image_cls_embeddings_for_prompt,
            "text_embeddings": text_cls_embeddings, # report
            "prompt_embeddings": prompt_embeddings, # alignment prompt for open-vocab zero-shot
            "label_cls_embeddings": label_cls_embeddings, # for multi-positive infonce between images and labels
            # Local embeddings
            "image_patch_embeddings_for_text": image_patch_embeddings_for_text,
            "text_patch_embeddings": text_patch_embeddings,
            # attention masks
            "text_attention_mask": text_attention_mask
        }

        # define placeholders
        l2i_predictions, i2l_predictions, l2t_predictions = None, None, None
        image_embeds_raw = [torch.cat([t1.unsqueeze(1), t2], dim=1) for t1, t2 in zip(image_cls_embeddings_raw, image_patch_embeddings_raw)]

        # cross-modality DQN
        attention_weights_for_images = None
        if self.enable_dqn_forward_pass:
            l2i_predictions, attention_weights_for_images = self.dqn(
                image_embeds_raw if self.use_all_image_tokens else image_patch_embeddings_raw, 
                label_cls_features_raw, 
                labels=labels
            )

            # for debug purpose only
            util.GlobalEnv.get().summary_writer.train.add_scalar( # for debug purpose
                "debug/l2i_max_logit", l2i_predictions.max().item(), util.GlobalEnv.get().summary_writer.global_step
            )

        # the following are for contrasive learning with attentioned modality embeddings.
        enriched_features = dict({})
        # NOTE: disable it for now for future work
        if self.attention_variant:
            enriched_features = self.enrich_embeddings_by_attention(**vanilla_features)

        return enriched_features | vanilla_features | {
            "logit_scale": self.logit_scale.exp(), 
            "logit_scale_aux": self.logit_scale_aux.exp(), # exp happens in the corresponding loss file.
            'predictions_by_label': i2l_predictions, # shape [32, number of labels=204, 1]
            'predictions_by_image': l2i_predictions, # shape [32, number of labels=204, 1]
            # 'predictions_by_grounding_maps': predictions_by_grounding_maps, # shape [32, number of labels=204, 1]
            'predictions_by_text': l2t_predictions,
            'attention_maps': attention_weights_for_images,
            "multihot_labels": torch.stack([torch.from_numpy(arr) for arr in multihot_labels], dim=0).long().to(device) # shape [32, number of labels=204]
        }

    def get_grounding_maps(self, image_patch_normed, label_cls_normed):
        """
        image_patch_normed: normalized image patch features (without cls)
        label_cls_normed: normalized label cls feature
        """
        # cosine similarity between each label CLS and each patch token
        # [B, Q, P] = [B, P, D] @ [Q, D]^T
        grounding_logits = torch.einsum("bpd,qd->bqp", image_patch_normed, label_cls_normed)
        # softmax across patches -> attention distribution over spatial tokens
        grounding_map = F.softmax(grounding_logits, dim=-1)  # [B, Q, P]
        return grounding_map, grounding_logits
    
