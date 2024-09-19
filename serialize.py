import time
import cv2

import torch
import torch.nn.functional as F
import numpy as np
import dsacstar

import argparse
import math

from network import Network
from torchvision import transforms

def parse_args():
    p = argparse.ArgumentParser(
        description = 'Serialize a model for inference.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('network', help='Network file')
    p.add_argument('image', help='Test image file')
    p.add_argument('--tiny', '-tiny', action='store_true', default=True, help='Load a model with massively reduced capacity for a low memory footprint.')
    
    return p.parse_args()

def serialize_model(image, model):
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save("traced_model.pt")
    # output = traced_script_module(image)
    return

args = parse_args()
file_path = args.image
image = cv2.imread(file_path)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(480),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.44],
        std=[0.25]
    )
])

image = transform(image)
image = image.unsqueeze(0)	


# load 
network_name = args.network
network = Network(torch.zeros((3)), args.tiny)
network.load_state_dict(torch.load(network_name, map_location=torch.device('cpu')))
network.eval()

print(image.shape)
serialize_model(image, network)
