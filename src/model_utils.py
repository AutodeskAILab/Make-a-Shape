import logging
import torch
from src.diffusion_progressive_module import (
    Trainer_Diffusion_Progressive_Network,
)


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

