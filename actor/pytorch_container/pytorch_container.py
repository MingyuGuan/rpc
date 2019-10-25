from __future__ import print_function
import container_rpc
import pytorch_utils
import os
import sys
import json
import io
import base64

import numpy as np
import cloudpickle
import torch
# import importlib
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models, transforms
import PIL.Image
import requests

IMPORT_ERROR_RETURN_CODE = 3

PYTORCH_WEIGHTS_RELATIVE_PATH = "pytorch_weights.pkl"
PYTORCH_MODEL_RELATIVE_PATH = "pytorch_model.pkl"


def load_predict_func(file_path):
    if sys.version_info < (3, 0):
        with open(file_path, 'r') as serialized_func_file:
            return cloudpickle.load(serialized_func_file)
    else:
        with open(file_path, 'rb') as serialized_func_file:
            return cloudpickle.load(serialized_func_file)


def load_pytorch_model(model_path, weights_path):
    if sys.version_info < (3, 0):
        with open(model_path, 'r') as serialized_model_file:
            model = cloudpickle.load(serialized_model_file)
    else:
        with open(model_path, 'rb') as serialized_model_file:
            model = cloudpickle.load(serialized_model_file)

    model.load_state_dict(torch.load(weights_path))
    return model


class PyTorchContainer(container_rpc.ModelContainerBase):
    def __init__(self, path, input_type):
        self.input_type = container_rpc.string_to_input_type(input_type)
        modules_folder_path = "{dir}/modules/".format(dir=path)
        sys.path.append(os.path.abspath(modules_folder_path))
        predict_fname = "func.pkl"
        predict_path = "{dir}/{predict_fname}".format(
            dir=path, predict_fname=predict_fname)
        self.predict_func = load_predict_func(predict_path)

        torch_model_path = os.path.join(path, PYTORCH_MODEL_RELATIVE_PATH)
        torch_weights_path = os.path.join(path, PYTORCH_WEIGHTS_RELATIVE_PATH)
        self.model = load_pytorch_model(torch_model_path, torch_weights_path)

    def predict_ints(self, inputs):
        preds = self.predict_func(self.model, inputs)
        return [str(p) for p in preds]

    def predict_floats(self, inputs):
        preds = self.predict_func(self.model, inputs)
        return [str(p) for p in preds]

    def predict_doubles(self, inputs):
        preds = self.predict_func(self.model, inputs)
        return [str(p) for p in preds]

    def predict_bytes(self, inputs):
        preds = self.predict_func(self.model, inputs)
        return [str(p) for p in preds]

    def predict_strings(self, inputs):
        preds = self.predict_func(self.model, inputs)
        return [str(p) for p in preds]

def predict_torch_model(model, imgs):
    # First we define the preproccessing on the images:
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       normalize
    ])

    # Then we download the labels:
    labels = {int(key):value for (key, value)
              in requests.get('https://s3.amazonaws.com/outcome-blog/imagenet/labels.json').json().items()}

    # Prepare a batch from `imgs`
    img_tensors = []
    for img in imgs:
        img_tensor = preprocess(PIL.Image.open(io.BytesIO(img)))
        img_tensor.unsqueeze_(0)
        img_tensors.append(img_tensor)
    img_batch = torch.cat(img_tensors)
    
    # Perform a forward pass
    with torch.no_grad():
        model_output = model(img_batch)
        
    # Parse Result
    img_labels = [labels[out.data.numpy().argmax()] for out in model_output]
        
    return img_labels


if __name__ == "__main__":
    print("Get pytorch model")
    # use a pretrained pytorch model
    pytorch_model = models.squeezenet1_1(pretrained=True)
    print("Save prediction function")
    model_path = pytorch_utils.save_python_function(predict_torch_model)
    print("Save pytorch model")
    pytorch_utils.save_pytorch_model(pytorch_model, model_path)
    print("done")

    print("Starting PyTorchContainer container..")
    model_name = "pytorch_model"
    model_version = 1
    input_type = "imgs"
    rpc_service = container_rpc.RPCService(model_path, input_type)
    try:
        model = PyTorchContainer(rpc_service.get_model_path(),
                                 rpc_service.get_input_type())
        sys.stdout.flush()
        sys.stderr.flush()
    except ImportError:
        sys.exit(IMPORT_ERROR_RETURN_CODE)
    rpc_service.start(model, model_name, model_version)
