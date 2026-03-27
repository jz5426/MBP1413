from torch import nn
from torch.nn.init import trunc_normal_

# from cxr-clip
# class MLPProjectionHead(nn.Module):
#     def __init__(self, embedding_dim, projection_dim):
#         super().__init__()
#         # the input should be already layer normed
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.layer_norm = nn.LayerNorm(projection_dim)

#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         # x = self.layer_norm(x)
#         return x


class LinearProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)

class MLPProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim=768,
        out_dim=65536,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, out_dim=out_dim, use_bn=use_bn, bias=mlp_bias
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.orthogonal_(m.weight) # TODO: might try this.
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # def _reset_parameters(self, m) -> None:
    #     if self._qkv_same_embed_dim:
    #         nn.init.xavier_uniform_(m.weight)
    #     else:
    #         nn.init.xavier_uniform_(self.q_proj_weight)
    #         nn.init.xavier_uniform_(self.k_proj_weight)
    #         nn.init.xavier_uniform_(self.v_proj_weight)

    #     if self.in_proj_bias is not None:
    #         nn.init.constant_(self.in_proj_bias, 0.0)
    #         nn.init.constant_(self.out_proj.bias, 0.0)
    #     if self.bias_k is not None:
    #         nn.init.xavier_normal_(self.bias_k)
    #     if self.bias_v is not None:
    #         nn.init.xavier_normal_(self.bias_v)

    def forward(self, x):
        x = self.mlp(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, out_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
        return nn.Sequential(*layers)
