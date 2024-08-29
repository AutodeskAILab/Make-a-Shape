import logging
import torch
from src.diffusion_progressive_module import (
    Trainer_Diffusion_Progressive_Network,
)
from src.networks.callbacks import EMACallback
import os
from huggingface_hub import hf_hub_download



def load_model(
    checkpoint_path,
    compile_model=False,
    device=None,
    eval=True,
):
    model = Trainer_Diffusion_Progressive_Network.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
    )

    if model.ema_state_dict is not None:
        # load EMA weights
        ema = EMACallback(decay=0.9999)
        ema.reload_weight = model.ema_state_dict
        ema.reload_weight_for_pl_module(model)
        ema.copy_to_pl_module(model)

    if compile_model:
        logging.info("Compiling models...")
        model.network.training_losses = torch.compile(model.network.training_losses)
        model.network.inference = torch.compile(model.network.inference)
        if hasattr(model, "clip_model"):
            model.clip_model.forward = torch.compile(model.clip_model.forward)
        logging.info("Done Compiling models...")

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model


class Model:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_path = pretrained_model_name_or_path
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="checkpoint.ckpt"
            )

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model = load_model(
            checkpoint_path,
            compile_model=False,
            device=device,
        )
        return model
