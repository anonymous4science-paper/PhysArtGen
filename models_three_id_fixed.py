import argparse
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

# Import necessary modules
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# ============ Correct Original Architecture Components ============

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2), (q, k, v))

        out = attention(q, k, v, q.shape[-1], mask, None)
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., glu=False):
        super().__init__()
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.w2 = nn.Linear(dim * mult, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.glu:
            x, gate = self.w1(x).chunk(2, dim=-1)
            x = x * F.gelu(gate)
        else:
            x = F.gelu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.0

        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls3 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t, context=None):
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x
        x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=context))) + x
        x = self.drop_path3(self.ls3(self.ff(self.norm3(x, t)))) + x
        return x

class LatentArrayTransformer(nn.Module):
    """
    Original Transformer architecture using AdaLayerNorm
    """
    def __init__(self, in_channels, t_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)
        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

    def forward(self, x, t, cond=None):
        # Time embedding processing
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        # Input projection
        x = self.proj_in(x)

        # Through Transformer blocks - Key: each block receives t_emb!
        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)
        
        x = self.norm(x)
        x = self.proj_out(x)
        return x

# ============ Fixed Three-ID Embedding ============

class ThreeIDEmbedding(nn.Module):
    """
    Fixed three-ID embedding layer: uses conflict-free identifier mapping
    """
    def __init__(self, major_categories=1, part_categories=3, max_identifiers=20, 
                 embedding_dim=512, identifier_mapping_file=None):
        super().__init__()
        
        self.major_category_emb = nn.Embedding(major_categories, embedding_dim // 4)
        self.part_category_emb = nn.Embedding(part_categories, embedding_dim // 4) 
        self.identifier_emb = nn.Embedding(max_identifiers, embedding_dim // 2)
        
        # Load identifier mapping
        self.identifier_to_id = {}
        if identifier_mapping_file and os.path.exists(identifier_mapping_file):
            with open(identifier_mapping_file, 'r') as f:
                self.identifier_to_id = json.load(f)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def convert_identifiers(self, identifiers):
        """Convert string identifiers to integer IDs (conflict-free)"""
        if isinstance(identifiers, (list, tuple)):
            # Process string list
            ids = []
            for ident in identifiers:
                if isinstance(ident, str):
                    if ident in self.identifier_to_id:
                        ids.append(self.identifier_to_id[ident])
                    else:
                        print(f"Warning: Unknown identifier '{ident}', using default ID 0")
                        ids.append(0)
                else:
                    ids.append(ident)  # Already an integer
            # Fix: don't specify device, let PyTorch handle device allocation automatically
            return torch.tensor(ids, dtype=torch.long)
        else:
            # Process single identifier or tensor
            return identifiers
    
    def forward(self, major_category_ids, part_category_ids, identifiers):
        """
        Args:
            major_category_ids: [B] - Major category ID 
            part_category_ids: [B] - Part category ID
            identifiers: [B] - Identifiers (string or numeric)
        """
        # Get various embeddings
        major_emb = self.major_category_emb(major_category_ids)  # [B, D/4]
        part_emb = self.part_category_emb(part_category_ids)     # [B, D/4]
        
        # Convert identifiers (key fix: conflict-free mapping)
        identifier_ids = self.convert_identifiers(identifiers)
        # Ensure tensor is on correct device (distributed training safe)
        if hasattr(identifier_ids, 'to'):
            identifier_ids = identifier_ids.to(major_category_ids.device)
        identifier_emb = self.identifier_emb(identifier_ids)     # [B, D/2]
        
        # Concatenate all embeddings
        combined = torch.cat([major_emb, part_emb, identifier_emb], dim=-1)  # [B, D]
        
        # Through fusion layer
        fused = self.fusion(combined)  # [B, D]
        
        return fused.unsqueeze(1)  # [B, 1, D] Add sequence dimension

# ============ Random Number Generator (unchanged) ============

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

# ============ EDM Sampler (unchanged) ============

def edm_sampler(
    net, latents, cond=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, three_id_emb=cond).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, three_id_emb=cond).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

# ============ EDM Loss Function (consistent with original project) ============

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(inputs) if augment_pipe is not None else (inputs, None)

        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()

# ============ Fixed Main Model ============

class ThreeIDEDMPrecond(torch.nn.Module):
    """
    Fixed three-ID conditional EDM model using correct AdaLayerNorm architecture
    """
    def __init__(self,
        n_latents = 512,
        channels = 8, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        n_heads = 8,
        d_head = 64,
        depth = 12,
        # Three-ID related parameters
        major_categories = 1,
        part_categories = 3, 
        max_identifiers = 20,
        identifier_mapping_file = None,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        d_model = n_heads * d_head
        
        # Three-ID embedding (fixed)
        self.three_id_embedding = ThreeIDEmbedding(
            major_categories=major_categories,
            part_categories=part_categories,
            max_identifiers=max_identifiers,
            embedding_dim=d_model,
            identifier_mapping_file=identifier_mapping_file
        )

        # Use original correct architecture
        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=d_model  # Conditional embedding dimension
        )

    def emb_three_ids(self, major_category_ids, part_category_ids, identifiers):
        """Get embeddings for three IDs"""
        return self.three_id_embedding(major_category_ids, part_category_ids, identifiers)

    def forward(self, x, sigma, major_category_ids=None, part_category_ids=None, 
                identifiers=None, three_id_emb=None, force_fp32=False, **model_kwargs):
        """
        Forward propagation
        """
        # Calculate conditional embedding
        if three_id_emb is not None:
            cond_emb = three_id_emb
        elif all(x is not None for x in [major_category_ids, part_category_ids, identifiers]):
            cond_emb = self.emb_three_ids(major_category_ids, part_category_ids, identifiers)
        else:
            raise ValueError("Must provide three IDs or pre-computed embedding")

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # EDM preconditioning calculation
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        # Model forward propagation (key: use original correct architecture)
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb.to(dtype))
        
        # Ensure output type is correct
        if F_x.dtype != torch.float32:
            F_x = F_x.to(torch.float32)
            
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    @torch.no_grad()
    def sample(self, major_category_ids=None, part_category_ids=None, identifiers=None, 
               three_id_emb=None, batch_seeds=None):
        """Generate samples"""
        # Calculate conditional embedding
        if three_id_emb is not None:
            cond = three_id_emb
        elif (major_category_ids is not None and part_category_ids is not None and identifiers is not None):
            cond = self.emb_three_ids(major_category_ids, part_category_ids, identifiers)
        else:
            raise ValueError("Must provide three IDs or pre-computed embedding")

        # Determine batch size and device
        if cond is not None:
            batch_size, device = cond.shape[0], cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        # Generate random noise
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)

        # EDM sampling
        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like)


# ============ Model Creation Functions ============

def three_id_kl_d512_m512_l8_edm(identifier_mapping_file=None):
    """Create fixed three-ID conditional model"""
    model = ThreeIDEDMPrecond(
        n_latents=512, 
        channels=8,
        n_heads=8,
        d_head=64,
        depth=12,
        major_categories=1,      # Currently only drawer
        part_categories=3,       # body, door, slider-drawer  
        max_identifiers=20,      # Adjust based on actual identifier count
        identifier_mapping_file=identifier_mapping_file
    )
    return model

def four_id_kl_d512_m512_l8_edm(identifier_mapping_file=None):
    """Create conditional model supporting 4ID (compatible with 3ID architecture)"""
    model = ThreeIDEDMPrecond(
        n_latents=512, 
        channels=8,
        n_heads=8,
        d_head=64,
        depth=12,
        major_categories=7,      # cabinet, desk, dishwasher, refrigerator, suitcase, table, trash
        part_categories=9,       # body, door, drawer, handle, lid, other, shelf, wheel, slider-drawer  
        max_identifiers=210,     # Adjust based on 4ID identifier count (object_id-instance_id combination)
        identifier_mapping_file=identifier_mapping_file
    )
    return model

# Register models
def _register_models():
    """Register models to current module"""
    globals()['three_id_kl_d512_m512_l8_edm'] = three_id_kl_d512_m512_l8_edm
    globals()['four_id_kl_d512_m512_l8_edm'] = four_id_kl_d512_m512_l8_edm

_register_models() 