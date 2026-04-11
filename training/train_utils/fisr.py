"""
FISR x2 and x4 models
"""

from __future__ import annotations

import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_3d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)


def to_4d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, hw, c = x.shape
    return x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, int):
            shape_tuple = (int(normalized_shape),)
        else:
            shape_tuple = tuple(normalized_shape)
        shape = torch.Size(shape_tuple)
        if len(shape) != 1:
            raise ValueError("normalized_shape must be 1D")
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]):
        super().__init__()
        if isinstance(normalized_shape, int):
            shape_tuple = (int(normalized_shape),)
        else:
            shape_tuple = tuple(normalized_shape)
        shape = torch.Size(shape_tuple)
        if len(shape) != 1:
            raise ValueError("normalized_shape must be 1D")
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: str):
        super().__init__()
        if layer_norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.view(b, c, h, w)
        return self.project_out(out)


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: str,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PromptExtractorX2(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        f1 = self.branch1(x)
        f1 = F.interpolate(f1, size=(64, 64), mode="bilinear", align_corners=False)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        return f1, f2, f3


class PromptExtractorX4(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        f1 = self.branch1(x)
        f1 = F.interpolate(f1, size=(32, 32), mode="bilinear", align_corners=False)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        return f1, f2, f3


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ActiveBlockPromptBasis(nn.Module):
    def __init__(self, dim: int, num_task: int = 6, num_basis: int = 8, embed_dim: int = 128):
        super().__init__()
        self.basis_mlp = MultiLayerPerceptron(dim, hidden_features=num_basis, out_features=num_basis)
        self.task_mlp = MultiLayerPerceptron(embed_dim, hidden_features=num_task, out_features=num_task)
        self.prompt = nn.Parameter(torch.rand(1, num_task, num_basis, embed_dim, 1, 1))
        self.conv3x3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor, flux: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flux = torch.where(torch.isnan(flux), torch.full_like(flux, 0), flux)
        flux = flux.permute(0, 2, 3, 1)
        x_hw = x.permute(0, 2, 3, 1)

        x_basis = F.softmax(self.basis_mlp(x_hw).permute(0, 3, 1, 2).contiguous(), dim=1)
        x_task = F.softmax(self.task_mlp(flux).permute(0, 3, 1, 2).contiguous(), dim=1)

        prompt_bank = self.prompt.unsqueeze(0).repeat(b, 1, 1, 1, 1, 1, 1).squeeze(1)
        prompts = x_basis.unsqueeze(1).unsqueeze(-3) * prompt_bank
        prompts = torch.sum(prompts, dim=2)
        prompts = x_task.unsqueeze(-3) * prompts
        prompts = torch.sum(prompts, dim=1)
        prompts = self.conv3x3(prompts)
        return prompts


class FISR(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: list[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        decoder: bool = False,
        use_loss: str = "L1",
        use_attention: bool = False,
    ):
        super().__init__()

        self.use_loss = use_loss
        self.use_attention = use_attention
        self.decoder = decoder

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if self.decoder:
            self.prompt1 = ActiveBlockPromptBasis(embed_dim=64, num_task=7, num_basis=64, dim=96)
            self.prompt2 = ActiveBlockPromptBasis(embed_dim=128, num_task=7, num_basis=32, dim=192)
            self.prompt3 = ActiveBlockPromptBasis(embed_dim=320, num_task=7, num_basis=16, dim=384)

        self.flux_module = PromptExtractorX2()

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(int(dim * 2**2))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**1) + 192, int(dim * 2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(
            dim=int(dim * 2**2) + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2**2) + 512, int(dim * 2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2**1) + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2**1) + 224, int(dim * 2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        self.up3_1 = Upsample(int(dim * 2**1))

        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 1**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(int(dim * 1**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if targets is None or "flux_map" not in targets:
            raise ValueError("FISR forward requires targets with key 'flux_map' to preserve original STAR behavior")
        flux_map = self.flux_module(targets["flux_map"])

        if self.decoder:
            dec3_param = self.prompt3(latent, flux_map[2])
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, flux_map[1])
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, flux_map[0])
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up3_1(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        if self.training:
            if targets is None or "mask" not in targets or "hr" not in targets:
                raise ValueError("Training mode requires targets with keys 'hr' and 'mask'")

            mask = targets["mask"]
            mask_sum = mask.sum() + 1e-3
            if self.use_loss == "L1":
                base_loss = (torch.abs(out_dec_level1 - targets["hr"]) * mask).sum() / mask_sum
                loss_name = "l1_loss"
            elif self.use_loss == "L2":
                base_loss = ((out_dec_level1 - targets["hr"]) ** 2 * mask).sum() / mask_sum
                loss_name = "l2_loss"
            else:
                raise ValueError(f"Unsupported use_loss='{self.use_loss}'")

            losses = {loss_name: base_loss}
            total_loss = base_loss
            if self.use_attention:
                if "attn_map" not in targets:
                    raise ValueError("use_attention=True requires targets['attn_map']")
                attn_map = torch.nan_to_num(targets["attn_map"], nan=0.0)
                weighted_diff = torch.abs(out_dec_level1 - targets["hr"]) * attn_map
                flux_loss = weighted_diff.sum() / (attn_map.sum() + 1e-3)
                losses["flux_loss"] = 0.01 * flux_loss
                total_loss = base_loss + 0.01 * flux_loss
            return total_loss, losses

        return {"pred_img": out_dec_level1}


class FISRx4(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: list[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: list[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        layer_norm_type: str = "WithBias",
        decoder: bool = False,
        use_loss: str = "L1",
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_loss = use_loss
        self.use_attention = use_attention
        self.decoder = decoder

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        if self.decoder:
            self.prompt1 = ActiveBlockPromptBasis(embed_dim=64, num_task=7, num_basis=64, dim=96)
            self.prompt2 = ActiveBlockPromptBasis(embed_dim=128, num_task=7, num_basis=32, dim=192)
            self.prompt3 = ActiveBlockPromptBasis(embed_dim=320, num_task=7, num_basis=16, dim=384)

        self.flux_module = PromptExtractorX4()

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )
        self.down2_3 = Downsample(int(dim * 2**1))

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )
        self.down3_4 = Downsample(int(dim * 2**2))

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2**1) + 192, int(dim * 2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(
            dim=int(dim * 2**2) + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2**2) + 512, int(dim * 2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2**1) + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2**1) + 224, int(dim * 2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))
        self.up3_1 = Upsample(int(dim * 2**1))
        self.up4_1 = Upsample(int(dim * 1**1))

        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layer_norm_type=layer_norm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 1**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(24, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if targets is None or "flux_map" not in targets:
            raise ValueError("FISRx4 forward requires targets with key 'flux_map' to preserve original STAR behavior")
        flux_map = self.flux_module(targets["flux_map"])

        if self.decoder:
            dec3_param = self.prompt3(latent, flux_map[2])
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, flux_map[1])
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, flux_map[0])
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.up3_1(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.up4_1(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        if self.training:
            if targets is None or "mask" not in targets or "hr" not in targets:
                raise ValueError("Training mode requires targets with keys 'hr' and 'mask'")

            mask = targets["mask"]
            mask_sum = mask.sum() + 1e-3
            if self.use_loss == "L1":
                base_loss = (torch.abs(out_dec_level1 - targets["hr"]) * mask).sum() / mask_sum
                loss_name = "l1_loss"
            elif self.use_loss == "L2":
                base_loss = ((out_dec_level1 - targets["hr"]) ** 2 * mask).sum() / mask_sum
                loss_name = "l2_loss"
            else:
                raise ValueError(f"Unsupported use_loss='{self.use_loss}'")

            losses = {loss_name: base_loss}
            total_loss = base_loss
            if self.use_attention:
                if "attn_map" not in targets:
                    raise ValueError("use_attention=True requires targets['attn_map']")
                attn_map = torch.nan_to_num(targets["attn_map"], nan=0.0)
                weighted_diff = torch.abs(out_dec_level1 - targets["hr"]) * attn_map
                flux_loss = weighted_diff.sum() / (attn_map.sum() + 1e-3)
                losses["flux_loss"] = 0.01 * flux_loss
                total_loss = base_loss + 0.01 * flux_loss
            return total_loss, losses

        return {"pred_img": out_dec_level1}


def create_fisr_x2(**kwargs) -> FISR:
    return FISR(**kwargs)


def create_fisr_x4(**kwargs) -> FISRx4:
    return FISRx4(**kwargs)
