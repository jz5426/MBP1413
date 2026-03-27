from torch import nn
from .transformer_decoder_kad import *
import logging
log = logging.getLogger(__name__)

class TQN_Model(nn.Module):
    def __init__(
        self, 
        logit_scale: nn.Parameter,
        nhead: int = 4,
        nlayers: int = 4,
        embed_dim: int = 768, 
        class_num: int = 1, 
        text_query_noise_std: float = 0,
        normalize_before: bool = True, # stay as default
        allow_self_attention: bool = True,
        use_mlp: bool = False,
        allow_disease_information_leakage_in_dqn: bool = False,
        # for finetuning
        enable_dqn_fewshot: bool = False,
        unlock_delta: bool = True,
        unlock_multi_mlp_heads: bool = True,
        delta_input_query_num: bool = -1
        ):
        super().__init__()
        self.text_query_noise_std = text_query_noise_std
        if self.text_query_noise_std > 0:
            log.info(f'[TQN_MODEL]: adding noise to the query embeddings of the DQN. {self.text_query_noise_std}')
        self.d_model = embed_dim
        self.logit_scale = logit_scale
        self.normalize_before = normalize_before
        self.allow_self_attention = allow_self_attention
        decoder_layer = TransformerDecoderLayer(
            self.d_model, 
            nhead, 
            1024,
            0.1, 
            'relu', 
            normalize_before=normalize_before,
            allow_self_attention=allow_self_attention
        )
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, 
            nlayers, 
            self.decoder_norm,
            return_intermediate=False
        )
        self.dropout_feas = nn.Dropout(0.1)
        self.class_num = class_num
        # classification head
        if use_mlp:
            self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
                nn.LayerNorm(embed_dim), # TODO: when using pre-norm version of dqn, remove this for post-norm
                nn.Linear(embed_dim, 1024),
                nn.GELU(),
                nn.Dropout(0.1),

                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.1),

                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),

                nn.LayerNorm(256),
                nn.Linear(256, class_num)
            )
        else:
            self.mlp_head = nn.Sequential( # nn.LayerNorm(768),
                nn.Linear(embed_dim, class_num)
            )

        self.apply(self._init_weights)
        # self.allow_disease_information_leakage_in_dqn = allow_disease_information_leakage_in_dqn
        # log.warning("[TQN_MODEL] allow information leakage") if self.allow_disease_information_leakage_in_dqn else log.warning("[TQN_MODEL] NOT allow information leakage")
        log.info('[TQN_MODEL]: Using TQN_model')

        # The Delta (The Learnable Part for few-shot finetuning)
        self.delta, self.mlp_heads = None, None
        self.enable_dqn_fewshot = enable_dqn_fewshot
        if enable_dqn_fewshot:

            if unlock_delta:
                self.delta = nn.Parameter(torch.zeros(delta_input_query_num, 1, embed_dim)) # => [number of labels, 1, embedding size]
                log.warning('[TQN_MODEL] There are delta parameters to learn!! Make sure this is finetuning')

            # 2. Create a ModuleList where each element is a deep copy of source_head.
            # deepcopy ensures they share the same INITIAL values, 
            # but are completely independent objects (gradients won't be shared).
            if unlock_multi_mlp_heads:
                self.mlp_heads = nn.ModuleList([
                    copy.deepcopy(self.mlp_head) for _ in range(delta_input_query_num)
                ])
                log.warning('[TQN_MODEL] Use multi-mlp head (one for each query)!! Make sure this is finetuning')

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def make_block_mask(self, Q: int, T: int, device=None, dtype=torch.float32, neg_inf=None):
        """
        Build block-diagonal attention mask to prevent cross-disease attention.
        Each block (size TxT) allows full attention inside, blocks others.
        Output shape: [Q*T, Q*T]
        """
        if neg_inf is None:
            neg_inf = -1e9 if dtype == torch.float32 else -1e4  # safer for float16

        L = Q * T
        mask = torch.full((L, L), neg_inf, device=device, dtype=dtype)
        for q in range(Q):
            start = q * T
            end = start + T
            mask[start:end, start:end] = 0.0  # allow intra-caption attention
        return mask

    def forward(
            self, 
            targets, 
            query_features, 
            return_atten = False, 
            caption_mask = None,
            labels=None,
            is_train_mode=True,
            is_feature_extraction_mode=False
        ):
        assert isinstance(targets, list), "target should be a list"
        B, N, D = targets[0].shape
        
        normed_targets = []
        for target in targets:
            target = target.transpose(0,1)
            # if not self.normalize_before: # TODO: avoid double norm check this.
            #     targets = self.decoder_norm(targets) # key and values
            target = self.decoder_norm(target) # key and values
            normed_targets.append(target)

        # --------------------------
        # Detect input format
        # --------------------------
        query_padding_mask, query_mask, memory_key_padding_mask = None, None, None
        if query_features.dim() == 2:
            # single set of text, then replicate it, otherwise keep the text shape as it is.
            query_features = query_features.unsqueeze(1).repeat(1, B, 1) # => number of query labels, Batch size, 512
        
        if query_features.dim() == 3:
            query_features_dim = 3

            # Single-token queries [Q, B, D]
            Q, B_, D_ = query_features.shape
            assert B_ == B and D_ == D, "Mismatch in batch or feature dimension"

            # --- ADD GAUSSIAN NOISE HERE ---
            # caption_mask = None => the query is label and key/values are the memory; otherwise the query is the image and the key/value are the text
            if is_train_mode and self.text_query_noise_std > 0 and caption_mask is None:
                # Generate Gaussian noise with the same shape as tgt
                noise = torch.randn_like(query_features) * self.text_query_noise_std
                query_features = query_features + noise

            if self.delta is not None:
                query_features = query_features + self.delta
            
            # if not self.normalize_before: # TODO: avoid double norm check this.
            #     query_features = self.decoder_norm(query_features)
            query_features = self.decoder_norm(query_features)
            query = query_features

            # ultimate decision form the self.allow_self_attention
            if not self.allow_self_attention:
                # no self attention module => no need for the mask
                query_mask = None

            # caption token masks.
            if caption_mask is not None:
                memory_key_padding_mask = ~(caption_mask.bool()) 

        elif query_features.dim() == 4:
            query_features_dim = 4

            # Caption-token queries [Q, T, B, D]
            query_features = query_features.transpose(1,2) # -> Q, T, B_, D_
            Q, T, B_, D_ = query_features.shape
            assert B_ == B and D_ == D, "Mismatch in batch or feature dimension"
            assert caption_mask is not None and caption_mask.shape == (Q, T, B)

            # Flatten tokens across queries
            query_features = query_features.reshape(Q * T, B_, D_) # [Q*T, B, D], comebine disease description into single long description
            if not self.normalize_before: # TODO: avoid double norm check this.
                query_features = self.decoder_norm(query_features)
            query = query_features

            # Construct tgt_key_padding_mask if provided
            if caption_mask is not None:
                assert caption_mask.shape == (Q, T, B), "caption_mask must be [Q, T, B]"
                caption_mask = caption_mask.reshape(Q * T, B)            # [Q*T, B]
                query_padding_mask = ~(caption_mask.T).bool()        # [B, Q*T], padding tokens = 1 for nn.multiheadattention from pytorch.     

            assert False, "Should not be here."
        else:
            raise ValueError(f"Unexpected text_features shape: {query_features.shape}")

        # --------------------------
        # Transformer Decoder Entry Point
        # --------------------------

        features, atten_map = self.decoder(
            tgt=query, # query
            memory=normed_targets, # key and values 
            tgt_mask=query_mask,
            tgt_key_padding_mask=query_padding_mask,
            query_pos=None, # the query tokens are the output from the bert model, already encoded positional information -> no need to add additional
            memory_key_padding_mask=memory_key_padding_mask,
            pos=None # the image patches already encoded positional information -> no need to add additional.
        ) 
        # atten_map in shape [batch size, classes, number of patch tokens=1370]

        # --------------------------
        # Aggregate & Classify
        # --------------------------
        if query_features_dim <= 3:
            # Single-token queries
            features = self.dropout_feas(features).transpose(0,1)       # [B, Q, D]
        else:
            # Caption-token queries
            features = features.view(Q, T, B, D) if features.dim() == 3 else features        # [Q, T, B, D]
            features = features.mean(dim=1)             # [Q, B, D] ← mean pooling over T
            features = features.transpose(0, 1)        # [B, Q, D]

        # only keep the extracted features (miggled with text information) for few-shot learning
        if is_feature_extraction_mode:
            return features

        # for training mode
        # out = self.mlp_head(features)  # => (batch_size, query_num)
        out = self.mlp_classification(features)
        return out, atten_map

    def mlp_classification(self, features):
        if self.mlp_heads is not None: # for finetuning
            outputs = []            
            # Iterate over both the heads and the query dimension of x
            for i, head in enumerate(self.mlp_heads):
                # 1. Slice the input: Take the i-th query token for the whole batch
                # Shape becomes: [batch_size, embed_dim]
                query_input = features[:, i, :] 
                
                # 2. Pass it through the specific head for this query index
                # Shape becomes: [batch_size, class_num]
                query_out = head(query_input)
                
                outputs.append(query_out)
            
            # 3. Stack the results back together
            # Final Shape: [batch_size, num_queries, class_num]
            out = torch.stack(outputs, dim=1)
        else:
            out = self.mlp_head(features)

        return out