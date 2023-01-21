from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch
from typing import List
from pytorch_lightning import Callback
from omegaconf import DictConfig
import hydra
from collections import OrderedDict
from torchvision import models

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

def extract_name_from_face_image_path(face_image_path: Path)  ->  str:
    """the name of the peoplt to whom the face belongs is cited in the picture name

    Args:
        face_image_path (Path): _description_

    Returns:
        str: _description_
    """
    split_face_image_name = Path(face_image_path).stem
    # remove image suffix _000X.jpg
    face_name, face_surname = split_face_image_name.split('_')[:2]
    face_full_name = face_name + " " + face_surname
    return face_full_name


def get_embedding_dims_from_checkpoint(checkpoint_file: OrderedDict) -> int:
    """_summary_

    Args:
        checkpoint_file (OrderedDict): _description_

    Returns:
        int: _description_
    """
    try: 
        return getattr(checkpoint_file['hyper_parameters']['net'], "model.fc.out_features")
    except:
        print(f"Could not find 'hyper_parameters' key in the checkpoint file")
    


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
    

def rename_weight_dict_keys(model: nn.Module, checkpoint_file: OrderedDict) ->  None:
    """Rename the checkpoint weights keys so that they match with the new model

    Args:
        model (nn.Module): model to which we want to load the new weights
        checkpoint (OrderedDict): contains the weights we want to load, but the wrong keys

    Returns:
        OrderedDict: checkpoint file with updated keys
    """
    weights_dict = OrderedDict()
    for key, ch_k, value in zip(model.state_dict().keys(), checkpoint_file.keys(), checkpoint_file.values()):
        print(key, ch_k)
        weights_dict[key] = value
    model.load_state_dict(weights_dict)

def instantiate_default_model(modelname: str) -> nn.Module:
    """_summary_

    Args:
        modelname (str): _description_

    Returns:
        nn.Module: _description_
    """
    if modelname == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    elif modelname == 'vit_l_16':
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
    elif modelname == 'vit_b_32':
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
    elif modelname == 'vit_l_32':
        model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
    elif modelname == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif modelname == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif modelname == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif modelname == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif modelname == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    return model


def get_model_backbone_from_checkpoint(checkpoint_file: OrderedDict) -> str:
    """_summary_

    Args:
        checkpoint_file (OrderedDict): _description_
    """
    try: 
        return getattr(checkpoint_file['hyper_parameters']['net'], "modelname")
    except:
        print(f"Could not find 'hyper_parameters' key in the checkpoint file")