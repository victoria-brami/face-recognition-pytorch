from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch
from typing import List
from pytorch_lightning import Callback
from omegaconf import DictConfig
import hydra
from collections import OrderedDict

def load_image(path_image: str) -> Image.Image:
    """Load image from harddrive and return 3-channel PIL image.
    Args:
        path_image (str): image path
    Returns:
        Image.Image: loaded image
    """
    return Image.open(path_image).convert('RGB')

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    for (i, callback_elt) in callbacks_cfg.items():
        callback = hydra.utils.instantiate(callback_elt)
        callbacks.append(callback)
    return callbacks


def load_checkpoint(model: nn.Module, filename: Path, device: str, key: str='state_dict') -> nn.Module:
    """Loads a previous checkpoint for the model

    Args:
        model (nn.Module): network
        filename (Path): path to the checkpoint file
        device (str): either cuda or cpu
        key (str, optional): key where the weights are stored. Defaults to 'state_dict'.

    Returns:
        nn.Module: the model update with the given checkpoint
    """
    checkpoint = torch.load(filename, map_location=device)[key]
    return model.load_state_dict(checkpoint)
    

def rename_weight_dict_keys(model: nn.Module, checkpoint: OrderedDict) ->  None:
    """Rename the checkpoint weights keys so that they match with the new model

    Args:
        model (nn.Module): model to which we want to load the new weights
        checkpoint (OrderedDict): contains the weights we want to load, but the wrong keys

    Returns:
        OrderedDict: checkpoint file with updated keys
    """
    weights_dict = OrderedDict()
    for key, ch_k, value in zip(model.state_dict().keys(), checkpoint.keys(), checkpoint.values()):
        print(key, ch_k)
        weights_dict[key] = value
    model.load_state_dict(weights_dict)