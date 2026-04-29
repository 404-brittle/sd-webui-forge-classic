"""
ControlNet-LLLite for Anima DiT (Forge edition).

This module provides the core LLLite architecture: a lightweight conditioning
network that injects control signals into the DiT's attention and MLP layers
by monkey-patching the target nn.Linear modules.

Ported from sd-scripts/networks/control_net_lllite_anima.py with Forge-specific
adaptations (removed training helpers, simplified logging).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Anima target class names (from backend/nn/anima.py)
TARGET_ATTENTION_CLASS = "SelfCrossAttention"
TARGET_MLP_CLASS = "GPT2FeedForward"

# LLM Adapter sub-tree is excluded from LLLite targeting
LLM_ADAPTER_NAME = "llm_adapter"

# Architecture version recorded in state_dict metadata
LLLITE_ARCH_VERSION = "2"


# ---------------------------------------------------------------------------
# target_layers: atomic specifiers and presets
# ---------------------------------------------------------------------------

ATOMIC_SPECIFIERS: Tuple[str, ...] = (
    "self_attn_q_pre",      # selfattn.q_proj
    "self_attn_kv_pre",     # selfattn.k_proj + v_proj (always paired)
    "cross_attn_q_pre",     # crossattn.q_proj
    "mlp_fc1_pre",          # mlp.layer1 (GPT2FeedForward fc1)
)

PRESETS: Dict[str, Tuple[str, ...]] = {
    "self_attn_q":            ("self_attn_q_pre",),
    "self_attn_qkv":          ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q":  ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> Tuple[str, ...]:
    """Resolve a target_layers spec string to a canonical atomic tuple.

    Accepts:
      - A single preset name (e.g. "self_attn_qkv")
      - Comma-separated atomic specifiers (e.g. "self_attn_q_pre,mlp_fc1_pre")

    Returns a deduplicated tuple ordered by ATOMIC_SPECIFIERS.
    """
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with groups dividing channels, capped at 8."""
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    """Pre-activation ResBlock: GN -> SiLU -> Conv3x3 -> GN -> SiLU -> Conv3x3 + skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Parallel branches with different dilations + global average pooling,
    concatenated and projected back to the input channel count.
    """

    def __init__(self, ch: int, dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1  # + global
        self.proj = nn.Sequential(
            nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    """v2 conditioning trunk.

    Input (B, 3, H, W) -> Conv4x4 s4 -> Conv3x3 s1 -> Conv4x4 s4 -> ResBlocks
    -> 1x1 Conv -> flatten -> LayerNorm -> (B, S, cond_emb_dim)
    """

    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        ch_half = cond_dim // 2

        self.conv1 = nn.Conv2d(3, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])

        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()  # (B, S, C)
        h = self.out_norm(h)
        return h


class LLLiteModuleDiT(nn.Module):
    """A single LLLite module that wraps one target Linear layer.

    Injects a learned perturbation: x + cx where cx is derived from the
    conditioning image embedding via a down-mid-up bottleneck with FiLM.
    """

    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        # Wrap in list to prevent nn.Module from registering org_module's params
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features

        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        # FiLM: cond_local -> (gamma, beta), zero-init for identity
        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Set by parent ControlNetLLLiteDiT.set_cond_image()
        self.cond_emb: Optional[torch.Tensor] = None

        # Set by parent at init
        self.layer_idx: int = -1
        self._depth_embeds_ref: List[nn.Parameter] = []

    def apply_to(self):
        """Monkey-patch the original module's forward with ours."""
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def restore(self):
        """Restore the original forward."""
        if hasattr(self, 'org_forward'):
            self.org_module[0].forward = self.org_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb  # (B, S, cond_emb_dim)

        # CFG inference: cond_emb is for cond batch, repeat for uncond
        if x.shape[0] // 2 == cx.shape[0]:
            cx = cx.repeat(2, 1, 1)

        assert x.shape[1] == cx.shape[1], (
            f"LLLite seq mismatch ({self.lllite_name}): "
            f"x={x.shape[1]} vs cond_emb={cx.shape[1]}"
        )

        # Add depth embedding (zero-init, so identity at start)
        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            cond_local = cx + depth_e
        else:
            cond_local = cx

        h = F.silu(self.down(x))  # (B, S, mlp)

        # FiLM parameters from cond_local
        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        mid_in = torch.cat([cond_local, h], dim=-1)
        m = self.mid(mid_in)
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        y = self.org_forward(x + out)

        if is_5d:
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


class ControlNetLLLiteDiT(nn.Module):
    """ControlNet-LLLite for Anima DiT.

    Shares a single conditioning encoder across all target layers and
    attaches LLLiteModuleDiT to each matched Linear.
    """

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)

        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()

        # Conditioning encoder: (B, 3, H*16, W*16) -> (B, S, cond_emb_dim)
        self.conditioning1 = _Conditioning1(
            cond_dim, cond_emb_dim, cond_resblocks,
            use_aspp=use_aspp, aspp_dilations=aspp_dilations,
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)

        # Depth embedding: per-module zero-init bias
        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

        aspp_info = f"aspp={'on' + str(list(self.aspp_dilations)) if use_aspp else 'off'}"
        logger.info(
            f"ControlNet-LLLite (Anima v{LLLITE_ARCH_VERSION}): created {n} modules for "
            f"target={target_layers!r} (atomics={list(atomics)}), "
            f"cond_dim={cond_dim}, cond_resblocks={cond_resblocks}, {aspp_info}, "
            f"cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}"
        )

    @property
    def target_atomics_str(self) -> str:
        return ",".join(self.target_atomics)

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: Tuple[str, ...]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            return False

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: Tuple[str, ...],
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not isinstance(child, nn.Linear):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not isinstance(child, nn.Linear):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        """Set the conditioning image embedding for all modules.

        Args:
            cond_image: (B, 3, H*16, W*16) tensor. None to clear.
        """
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)  # (B, S, cond_emb_dim)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        """Monkey-patch all target Linear layers."""
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        """Restore all original forward methods."""
        for m in self.lllite_modules:
            m.restore()


# ---------------------------------------------------------------------------
# Save / load helpers (for loading pre-trained LLLite weights)
# ---------------------------------------------------------------------------

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _from_saved_state_dict(lllite: ControlNetLLLiteDiT, weights_sd: dict) -> dict:
    """Convert saved keys to internal state_dict format."""
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[:-len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(f"depth_embed slices missing for module idx(es) {missing}")
        out[_INTERNAL_DEPTH_KEY] = torch.stack(
            [depth_slices[i] for i in range(n_modules)], dim=0
        )

    return out


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    """Load LLLite weights from a .safetensors file."""
    if file.endswith(".safetensors"):
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    # Reject legacy format
    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in a legacy ControlNet-LLLite weight format "
            f"(keys starting with '{_INTERNAL_MODULES_PREFIX}'). The current code uses a "
            f"named-key format. Re-train with the current codebase."
        )

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    logger.info(f"loaded LLLite weights from {file}: {info}")
    return info
