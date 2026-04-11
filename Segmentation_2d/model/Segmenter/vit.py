import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from core.modules.transformer import TransformerEncoderBlock, PatchEmbedding

def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        in_chans,
        class_num,
        patch_size,
        embed_dim,
        n_layer,
        n_head,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.0,
        **kwargs
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.patch_size = patch_size
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.n_heads = n_head
        self.class_num = class_num

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.n_patches + 1, embed_dim)
        )

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                **kwargs
            ) for i in range(n_layer)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, class_num)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.pre_logits = nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward(self, x, return_features=False):
        B, _, H, W = x.shape

        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // self.patch_size, W // self.patch_size),
                num_extra_tokens,
            )
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if return_features:
            return x

        x = x[:, 0]
        x = self.head(x)
        return x