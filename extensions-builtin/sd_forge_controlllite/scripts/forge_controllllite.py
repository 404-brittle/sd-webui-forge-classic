import logging

from modules_forge.shared import add_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher
from lib_controllllite.lib_controllllite import LLLiteLoader

logger = logging.getLogger(__name__)

opLLLiteLoader = LLLiteLoader().load_lllite


def _is_anima_lllite(state_dict: dict) -> bool:
    """Detect Anima DiT LLLite weights by checking for the Anima-specific key prefix.

    Anima LLLite weights use 'lllite_conditioning1.' and 'lllite_dit_' prefixes,
    while standard SD1.5/SDXL LLLite weights use plain 'conditioning1.' etc.
    """
    return any(k.startswith("lllite_conditioning1.") for k in state_dict) or any(
        k.startswith("lllite_dit_") for k in state_dict
    )


class ControlLLLitePatcher(ControlModelPatcher):
    @staticmethod
    def try_build_from_state_dict(state_dict, ckpt_path):
        if not any('lllite' in k for k in state_dict.keys()):
            return None

        # Anima DiT LLLite weights are handled by modules_forge.control_lllite
        if _is_anima_lllite(state_dict):
            from modules_forge.control_lllite import ControlLLLitePatcher as AnimaPatcher
            logger.info(f"Detected Anima DiT LLLite weights, delegating to Anima patcher: {ckpt_path}")
            return AnimaPatcher(state_dict, ckpt_path)

        return ControlLLLitePatcher(state_dict)

    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        return

    def process_before_every_sampling(self, process, cond, mask, *args, **kwargs):
        unet = process.sd_model.forge_objects.unet

        unet = opLLLiteLoader(
            model=unet,
            state_dict=self.state_dict,
            cond_image=cond.movedim(1, -1),
            strength=self.strength,
            steps=process.steps,
            start_percent=self.start_percent,
            end_percent=self.end_percent
        )[0]

        process.sd_model.forge_objects.unet = unet
        return


add_supported_control_model(ControlLLLitePatcher)
