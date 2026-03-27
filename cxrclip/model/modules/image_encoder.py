import torch
from torch import nn
from torchvision.models.resnet import resnet50
from transformers import SwinModel
from timm.models.vision_transformer import PatchEmbed, Block
from transformers import AutoModel, AutoConfig
from cxrclip.model.modules.dinov3_vision_transformer import *
from cxrclip.model.modules.raddino_utils import RadDinoLocal, VisionHead

class HuggingfaceImageEncoder(nn.Module):
    def __init__(
        self,
        name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = "~/.cache/huggingface/hub",
        model_type: str = "vit",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        if pretrained and self.model_type == 'swin':
            # NOTE: does not support multiple CLS tokens
            self.image_encoder = SwinModel.from_pretrained(
                name,
                cache_dir=cache_dir, 
                local_files_only=local_files_only
            )
        else:
            raise NotImplementedError(f"Model not supported : {model_type}")

        if gradient_checkpointing and self.image_encoder.supports_gradient_checkpointing:
            self.image_encoder.gradient_checkpointing_enable()

        self.out_dim = self.image_encoder.config.hidden_size

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(pixel_values=image, interpolate_pos_encoding=True)
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)


class ResNet50(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True, cache_dir=''):
        super().__init__()
    
        assert(pretrained == True)
        self.resnet = resnet50(pretrained=False) # this eventually takes the pretrained model from CXR-CLIP anyways
        if pretrained:
            state_dict = torch.load(cache_dir)
            self.resnet.load_state_dict(state_dict, strict=True)
            print('loaded imagenet pretrained weights to ResNet50')

        self.out_dim = 2048
        del self.resnet.fc
        self.resnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.resnet)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

# # NOTE: ADD VIT-16-M3AE FROM CARZERO MODEL HERE.
# # checkout CARZERO.py and 
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = omega / embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class MRM(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone for CARZERO
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, dual_cls=False,
                 custom_pretrain_weights=None, load_pretrain=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dual_cls = False
        if dual_cls:
            self.dual_cls = True
            self.cls2_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        self.embed_dim = embed_dim
        self.out_dim = embed_dim

        # load the pretrained weights again that is vlm pretrained on cxr
        missing, unexpected = [], []
        if custom_pretrain_weights is not None and load_pretrain:
            missing, unexpected = self.load_state_dict(custom_pretrain_weights, strict=False if dual_cls else True)
            # print('Loaded custom pretrained weights.')
            # print('Missing:', missing)
            # print('Unexpected:', unexpected)
            assert (len(missing)+len(unexpected)) == 0 if not dual_cls else 1, "number of parameters do not match."

        # initialize the weights of the cls2_token
        if self.dual_cls:
            with torch.no_grad():
                self.cls2_token.copy_(self.cls_token)
            print('cls2_token is now initialized with the same weight as the cls_token too.')
        else:
            assert (len(missing)+len(unexpected)) == 0

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        
        # ipdb.set_trace()
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unlocalpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, 768))
        x = torch.einsum('nhwc->nchw', x)
        # imgs = x.reshape(shape=(x.shape[0], 768, h * p, h * p))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def image_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        # append cls2 token if applicable
        # note that cls2_token share the same position embedding as cls_token, this is intentional and relies on 
        # different learning signal to make the two embeddings serve different purposes.
        if self.dual_cls:
            cls2_token = self.cls2_token + self.pos_embed[:, :1, :]
            cls2_tokens = cls2_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1) if not self.dual_cls else torch.cat((cls_tokens, cls2_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # layer norm

        n_cls_token = 2 if self.dual_cls else 1

        return { 
            'cls_token': x[:, :n_cls_token, :], # TODO: handle situation where all dimension of shape > 1
            'patch_tokens': x[:, n_cls_token:, :]
        }

        # return  x[:, 0, :], x[:, 1:,:] if not self.dual_cls else x[:, 0, :], x[:, 1, :], x[:, 2:, :]
        # return  x[:, 1:, :].mean(1), x[:, 1:,:]  # use pooling as global feature
    
    # def forward_decoder(self, x, ids_restore):
    #     # embed tokens
    #     x = self.decoder_embed(x)

    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

    #     # add pos embed
    #     x = x + self.decoder_pos_embed

    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)

    #     # predictor projection
    #     x = self.decoder_pred(x)

    #     # remove cls token
    #     x = x[:, 1:, :]

    #     return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        # ipdb.set_trace()
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, last_n_hidden_layers=[-1], unlocal_patch=False):
        results = self.image_encoder(imgs)
        if unlocal_patch:
            local_latent = self.unlocalpatchify(results['patch_tokens'])
            results['patch_tokens'] = local_latent # overwrite.
        results['hidden_states'] = []
        return results
        # return global_latent, local_latent if not self.dual_cls else global1_latent, global2_latent, local_latent

class DINOv3(nn.Module):
    def __init__(self, dinov3_version_name, ckpt_path, dual_cls=False, load_pretrain=True):
        super().__init__()

        # string to function mapping
        create_dinov3 = globals()[dinov3_version_name]

        # create dinov3 with pretrained weights
        self.dinov3 = create_dinov3(
            weights=ckpt_path, 
            dual_cls=dual_cls, # toggle this to have either 1 cls token or 2 cls tokens
            pretrained=load_pretrain
        )
        self.out_dim = self.dinov3.embed_dim

    def forward(self, x):
        output = self.dinov3(x)
        results = { 
            'cls_token': output['x_norm_clstoken'],
            'patch_tokens': output['x_norm_patchtokens']
        }
        return results

class XrayDinov2_224(nn.Module):
    """
    model downloaded from https://huggingface.co/StanfordAIMI/dinov2-base-xray-224
    """
    def __init__(self, ckpt_dir, freeze_backbone=False):
        super().__init__()
        config = AutoConfig.from_pretrained(ckpt_dir, local_files_only=True)
        config.output_hidden_states = True  # need this to access the intermediate layer outputs

        self.model = AutoModel.from_pretrained(
            ckpt_dir,
            config=config,
            dtype="auto",      # or "auto" for GPU
            trust_remote_code=True,
            local_files_only=True
        )
        self.out_dim = 768

    def forward(self, x, last_n_hidden_layers=None):
        outputs = self.model(x)
        last_hidden_state = outputs.last_hidden_state
        preserved_hidden_states = [
            outputs.hidden_states[i] for i in last_n_hidden_layers
        ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []
        del outputs # free up some memory
        cls_tokens, patch_tokens = last_hidden_state[:, 0], last_hidden_state[:, 1:]
        assert patch_tokens.shape[1] == 256, 'The input image shape should be 224.'
        return { 
            'cls_token': cls_tokens.unsqueeze(1),
            'patch_tokens': patch_tokens,
            'hidden_states': preserved_hidden_states
        }

class RadDINO(nn.Module):
    
    def __init__(self, ckpt_dir, freeze_backbone, interpolate_pos_encoding=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.raddino_encoder = RadDinoLocal(ckpt_dir, freeze_backbone, interpolate_pos_encoding)
        self.out_dim = 768

        # two transformer header on top of the vision backbone
        if self.freeze_backbone:
            self.vision_head = VisionHead(
                input_dim=self.out_dim,
                num_heads=12,
                num_blocks=2, # number of transformer layer for training
                blocks_drop_path=True,
                # embed_dim=768,
                # use_class_token=True,
                # use_image_patch_tokens=False, # in dinov2.text is true, but not here
                # use_linear_projection=False # DOES NOT MATTER
            )
            assert isinstance(self.vision_head.linear_projection, nn.Identity), "should not have projection layer on vision_head."
            print('initiated vision head for training.')

    def forward(self, x, last_n_hidden_layers=None):
        hidden_states = None
        if not self.freeze_backbone:
            cls_tokens, patch_tokens, hidden_states = self.raddino_encoder.extract_features(x)
            hidden_states = [
                hidden_states[i] for i in last_n_hidden_layers
            ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []
        else:
            # freeze the features
            radino_output = self.raddino_encoder.model(x, output_hidden_states=True)
            last_hidden_state = radino_output.last_hidden_state
            finaloutput = self.vision_head(last_hidden_state)
            # true last layer output
            cls_tokens, patch_tokens = finaloutput.last_hidden_state[:, 0], finaloutput.last_hidden_state[:, 1:]
            hidden_states = radino_output.hidden_states + finaloutput.hidden_states
            del radino_output # free up some memory

            hidden_states = [
                hidden_states[i] for i in last_n_hidden_layers
            ] if last_n_hidden_layers is not None and last_n_hidden_layers != [-1] else []

            # need to add the hidden states from finaloutput to the hidden_states otherwise the hidden states 
            # only represents the states from the backbone

        return { 
            'cls_token': cls_tokens.unsqueeze(1),
            'patch_tokens': patch_tokens
        } | { 'hidden_states': hidden_states } if hidden_states is not None else {}