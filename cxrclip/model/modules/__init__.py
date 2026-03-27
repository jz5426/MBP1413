from functools import partial
import os
from typing import Dict
import torch.nn as nn
import torch
from .image_classifier import LinearClassifier
from .image_encoder import MRM, HuggingfaceImageEncoder, ResNet50, DINOv3, RadDINO, XrayDinov2_224
from .projection import LinearProjectionHead, MLPProjectionHead
from .text_encoder import HuggingfaceTextEncoder
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers.tokenization_utils import PreTrainedTokenizer

def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "huggingface":
        assert config_image_encoder['dual_cls'] == False, 'Dual CLS tokens are not supported in Swin model'
        cache_dir = config_image_encoder["cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder["gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else None
        _image_encoder = HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )

    elif config_image_encoder["source"].lower() == "torchvision" and config_image_encoder["name"] == "resnet": # make sure backward compatible

        cache_dir = config_image_encoder['cache_dir']
        assert 'dual_cls' not in config_image_encoder, 'Dual CLS tokens are not supported in ResNet model'
        _image_encoder = ResNet50(cache_dir=cache_dir)

    elif config_image_encoder["source"].lower() == "laihaoran" and config_image_encoder["name"] == "VITB-16-M3AE":
        cache_dir = config_image_encoder['cache_dir']
        assert 'VITB-16-M3AE_last.ckpt' in cache_dir

        # preprocess the pretrained checkpoints
        torch.serialization.add_safe_globals([ModelCheckpoint])
        pretrain_dict = torch.load(cache_dir, map_location=torch.device('cpu'), weights_only=False)
        pretrained_dict = {k[len('vision_encoder.'):]: v for k, v in pretrain_dict['state_dict'].items() if 'vision_encoder.' in k}

        _image_encoder = MRM(
            patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            dual_cls=config_image_encoder['dual_cls'],
            custom_pretrain_weights=pretrained_dict,
            load_pretrain=config_image_encoder['pretrained']
        )

    elif config_image_encoder["source"].lower() == "laihaoran" and config_image_encoder["name"] == "CARZero_best": # load the image encoder part form the carzero_best checkpoint
       
        cache_dir = config_image_encoder['cache_dir']
        carzero_best_derived_img_dict = torch.load(cache_dir, map_location=torch.device('cpu'), weights_only=False)

        _image_encoder = MRM(
            patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            dual_cls=config_image_encoder['dual_cls'], 
            custom_pretrain_weights=carzero_best_derived_img_dict,
            load_pretrain=config_image_encoder['pretrained']
        )

    elif config_image_encoder["source"].lower() == "meta" and 'dinov3' in config_image_encoder["name"]:
        _image_encoder = DINOv3(
            dinov3_version_name=config_image_encoder['name'],
            ckpt_path=config_image_encoder['cache_dir'],
            dual_cls=config_image_encoder['dual_cls'],
            load_pretrain=config_image_encoder['pretrained']
            )
    elif config_image_encoder["source"].lower() == "microsoft" and 'raddino' in config_image_encoder["name"]:
        _image_encoder = RadDINO(
            ckpt_dir=config_image_encoder['cache_dir'],
            freeze_backbone=config_image_encoder['freeze_backbone'],
            interpolate_pos_encoding=False if config_image_encoder.get('image_size', 518) == 518 else True # default is 518 unless explicitly specificed
        )
    elif config_image_encoder["source"].lower() == "stanford_aiml" and 'xraydinov2_224' in config_image_encoder["name"]:
        _image_encoder = XrayDinov2_224(
            ckpt_dir=config_image_encoder['cache_dir'],
            freeze_backbone=config_image_encoder['freeze_backbone']
        )
    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder

def load_text_encoder(config_text_encoder: Dict, tokenizer: PreTrainedTokenizer):
    if config_text_encoder["source"].lower() == "huggingface":
        # assert False, 'Not support CXRCLIP pretrained weights for now.'
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            tokenizer=tokenizer,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=config_text_encoder["trust_remote_code"],
            dual_cls=config_text_encoder['dual_cls']
        )

    elif config_text_encoder["source"].lower() == "laihaoran":

        # load the default pretrained weights from Lanhairan/bioClinicalMPBERT
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]

        # load the text encoder weights based on the carzero vlm pretrained version instead of the independently pretrained model.
        revised_dict = None
        if 'carzero_pretrained_dir' in config_text_encoder:
            carzero_best_derived_text_dict = torch.load(config_text_encoder["carzero_pretrained_dir"], map_location="cpu", weights_only=False)
            revised_dict = { key[len('text_encoder.'):] : carzero_best_derived_text_dict[key] for key in carzero_best_derived_text_dict if 'text_encoder.' in key}
            
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            tokenizer=tokenizer,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=config_text_encoder["trust_remote_code"],
            dual_cls=config_text_encoder['dual_cls'],
            custom_pretrain_weights=revised_dict
        )
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    if config_projection_head["name"].lower() == "mlp":
        projection_head = MLPProjectionHead(
            in_dim=embedding_dim, out_dim=config_projection_head["proj_dim"]
        )
    elif config_projection_head["name"].lower() == "linear":
        projection_head = LinearProjectionHead(embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"])
    else:
        raise KeyError(f"Not supported text encoder: {config_projection_head}")
    return projection_head


def load_image_classifier(config_image_classifier: Dict, feature_dim: int):
    if config_image_classifier["name"].lower() == "linear":
        _image_classifier = LinearClassifier(feature_dim=feature_dim, num_class=config_image_classifier["n_class"])
    else:
        raise KeyError(f"Not supported image classifier: {config_image_classifier}")

    return _image_classifier
