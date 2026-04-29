"""
ControlNet-LLLite integration for Forge.

Provides a ControlModelPatcher subclass that detects LLLite weight files,
builds the ControlNetLLLiteDiT network, monkey-patches the Anima DiT model,
and manages the control image lifecycle during sampling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from backend import memory_management
from backend.nn.control_net_lllite_anima import (
    ControlNetLLLiteDiT,
    _from_saved_state_dict,
    parse_target_layers,
)

logger = logging.getLogger(__name__)


def _read_lllite_metadata(weights_path: str) -> Dict[str, str]:
    """Read metadata from a .safetensors LLLite weight file."""
    with safe_open(weights_path, framework="pt") as f:
        meta = f.metadata()
    return meta or {}


def _load_control_image(
    path: str, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load and normalize a control image to a (1, 3, H, W) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=dtype)


class ControlLLLitePatcher:
    """
    ControlModelPatcher-compatible class for ControlNet-LLLite on Anima DiT.

    This does NOT extend ControlModelPatcher because LLLite works by
    monkey-patching the DiT's Linear layers rather than producing control
    signals. Instead, it implements the same interface (try_build_from_state_dict,
    process_before_every_sampling, process_after_every_sampling) and is
    registered via add_supported_control_model.
    """

    def __init__(self, state_dict: dict, ckpt_path: str):
        self.state_dict = state_dict
        self.ckpt_path = ckpt_path
        self.lllite: Optional[ControlNetLLLiteDiT] = None
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0
        self.positive_advanced_weighting = None
        self.negative_advanced_weighting = None
        self.advanced_frame_weighting = None
        self.advanced_sigma_weighting = None
        self.advanced_mask_weighting = None

    @staticmethod
    def try_build_from_state_dict(state_dict: dict, ckpt_path: str):
        """Detect LLLite weights by the presence of lllite_conditioning1.* keys."""
        # LLLite weight files have keys like:
        #   lllite_conditioning1.conv1.weight
        #   lllite_dit_blocks_0_self_attn_q_proj.down.weight
        #   etc.
        has_cond = any(k.startswith("lllite_conditioning1.") for k in state_dict)
        has_modules = any(
            k.startswith("lllite_dit_") for k in state_dict
        )
        if not (has_cond or has_modules):
            return None

        logger.info(f"Detected ControlNet-LLLite weights: {ckpt_path}")
        return ControlLLLitePatcher(state_dict, ckpt_path)

    def process_after_running_preprocessors(self, process, params, *args, **kwargs):
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        """Build and apply LLLite before sampling begins.

        This method:
        1. Reads metadata from the weight file for config
        2. Builds ControlNetLLLiteDiT matching the Anima DiT architecture
        3. Loads weights
        4. Monkey-patches the target Linear layers
        5. Sets the control image
        """
        sd_model = process.sd_model
        if not hasattr(sd_model, 'forge_objects') or not hasattr(sd_model.forge_objects, 'unet'):
            logger.error("LLLite: sd_model does not have forge_objects.unet")
            return

        unet_patcher = sd_model.forge_objects.unet
        diffusion_model = unet_patcher.model.diffusion_model

        # Check that this is an Anima model
        model_class = diffusion_model.__class__.__name__
        if model_class not in ("Anima", "MiniTrainDIT"):
            logger.warning(
                f"LLLite: expected Anima/MiniTrainDIT model, got {model_class}. "
                f"Proceeding anyway."
            )

        # Read metadata for config
        meta = _read_lllite_metadata(self.ckpt_path)
        cond_emb_dim = int(meta.get("lllite.cond_emb_dim", "32"))
        mlp_dim = int(meta.get("lllite.mlp_dim", "64"))
        target_layers = meta.get(
            "lllite.target_atomics",
            meta.get("lllite.target_layers", "self_attn_q"),
        )
        cond_dim = int(meta.get("lllite.cond_dim", "64"))
        cond_resblocks = int(meta.get("lllite.cond_resblocks", "1"))
        use_aspp = meta.get("lllite.use_aspp", "false").lower() == "true"
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            from backend.nn.control_net_lllite_anima import ASPP_DEFAULT_DILATIONS
            aspp_dilations = ASPP_DEFAULT_DILATIONS

        version = meta.get("lllite.version", "?")
        logger.info(
            f"LLLite config (v{version}): cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}, "
            f"target_layers={target_layers}, cond_dim={cond_dim}, "
            f"cond_resblocks={cond_resblocks}, "
            f"use_aspp={use_aspp}{' dilations=' + str(list(aspp_dilations)) if use_aspp else ''}, "
            f"multiplier={self.strength}"
        )

        # Build LLLite
        device = memory_management.get_torch_device()
        dtype = torch.bfloat16  # Anima uses bfloat16

        self.lllite = ControlNetLLLiteDiT(
            diffusion_model,
            cond_emb_dim=cond_emb_dim,
            mlp_dim=mlp_dim,
            target_layers=target_layers,
            multiplier=self.strength,
            cond_dim=cond_dim,
            cond_resblocks=cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
        )

        # Load weights from the stored state_dict
        # Convert saved key format (lllite_conditioning1.*, lllite_dit_*) to internal format
        converted_sd = _from_saved_state_dict(self.lllite, self.state_dict)
        info = self.lllite.load_state_dict(converted_sd, strict=False)
        if info.missing_keys:
            logger.warning(f"LLLite missing keys: {info.missing_keys}")
        if info.unexpected_keys:
            logger.warning(f"LLLite unexpected keys: {info.unexpected_keys}")

        # Apply monkey-patches
        self.lllite.apply_to()
        self.lllite.to(device=device, dtype=dtype)
        self.lllite.eval().requires_grad_(False)

        # Register with the UNet patcher for memory management
        unet_patcher.add_extra_torch_module_during_sampling(self.lllite, cast_to_unet_dtype=False)

        # Set control image from the process's control tensor
        if cond is not None:
            # cond is the control image tensor from the UI (B, C, H, W) in [0, 1]
            # LLLite _Conditioning1 expects [-1, 1] range, so rescale
            cond_img = cond.to(device=device, dtype=dtype)
            cond_img = cond_img * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            self.lllite.set_cond_image(cond_img)
            logger.info(f"LLLite: set cond image shape={tuple(cond.shape)}")
        else:
            logger.warning("LLLite: no control image provided (cond is None)")

    def process_after_every_sampling(self, process, params, *args, **kwargs):
        """Clean up LLLite after sampling."""
        if self.lllite is not None:
            self.lllite.clear_cond_image()
            self.lllite.restore()
            # Free memory
            self.lllite.to("cpu")
            self.lllite = None
            logger.info("LLLite: cleaned up after sampling")


