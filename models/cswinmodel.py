import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from the_well.benchmark.models.common import BaseModel
from models.cswin_block import CSWinBlockFlex

class CSWinPredictor(nn.Module):
    
    def __init__(self, dim_in, dim_out, spatial_resolution, embed_dim=64, depth=4, num_heads=4, gradient_checkpointing=False):
        super().__init__()

        # Support non-square spatial resolutions as a tuple (H, W)
        self.spatial_resolution = spatial_resolution if isinstance(spatial_resolution, tuple) else (spatial_resolution, spatial_resolution)
        self.gradient_checkpointing = gradient_checkpointing
        
        # Patch Embedding (Map raw PDE channels to Transformer dimension)
        self.patch_embed = nn.Conv2d(dim_in, embed_dim, kernel_size=1)
        
        # Sequence of CSWin Transformer Blocks
        self.blocks = nn.ModuleList([
            CSWinBlockFlex(
                dim=embed_dim,
                reso=self.spatial_resolution,
                num_heads=num_heads,
                split_size=8,
                mlp_ratio=4.0
            ) for _ in range(depth)
        ])
        
        # Output Projection (Map Transformer dimension to Target PDE fields)
        self.head = nn.Conv2d(embed_dim, dim_out, kernel_size=1)

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x):
        # x is (B, C_in, H, W)
        
        # Embed and flatten
        x = self.patch_embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = self.optional_checkpointing(block, x)
            
        # Unflatten and project to output
        H, W = self.spatial_resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.head(x)
        
        return x


class CSWinModel(BaseModel):
    """
    Wrapper around CSwinTransformer to follow the_well BaseModel style.
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Build the actual PyTorch module
        self.model = CSWinPredictor(
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            spatial_resolution=self.spatial_resolution,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            gradient_checkpointing=gradient_checkpointing
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)