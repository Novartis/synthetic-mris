import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
from model_architecture.resnet_backbone import resnet50_backbone


def get_medicalnet_backbone(dim: int, path_to_model: str):
    """Converts pretrained medicalnet to a useable format, removing the distribution wrapper.

    Args:
        dim (int): Desired image dimension.
        path_to_model (str): Location of the pretrained MedicalNet model.
    """

    # create model instance and put in DataParallel wrapper
    model = resnet50_backbone(sample_input_H=dim, sample_input_W=dim, sample_input_D=dim)
    net = torch.nn.DataParallel(model, device_ids=[0])

    # load state dict for pretrained model.
    net.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu"))["state_dict"])

    # send model to non-distributed (cpu) device.
    return net.module.to("cpu")
