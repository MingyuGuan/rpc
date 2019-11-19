from __future__ import print_function
import container_rpc
import pytorch_utils
import os
import sys
import json
import io
import base64

import numpy as np
# import cloudpickle
from ray import cloudpickle as pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import jsonpickle
import mlflow.pytorch 
import mlflow.tensorflow

IMPORT_ERROR_RETURN_CODE = 3

FRAMEWORKS = {
  'torch'    : {
                  'load': mlflow.pytorch.load_model,
                  'save': mlflow.pytorch.save_model
  },
  'tensorflow' : {
                  'load' : mlflow.tensorflow.load_model,
                  'save' : mlflow.tensorflow.save_model
  },
}


# class PredictModelPytorch():
#   def __init__(self,transform, model):
#     self.transform = transform
#     self.model = model

#   def __call__(self,batch_data):
#     batch_size = len(batch_data)
#     transform_result = []
#     for i in range(batch_size):
#       data = Image.open(io.BytesIO(base64.b64decode(batch_data[i])))
#       if data.mode != "RGB":
#         data = data.convert("RGB")
#       data = self.transform(data)
#       transform_result.append(data)
#     model_input = torch.stack(transform_result)
#     model_input = Variable(model_input)
#     outputs = self.model(model_input)
#     _, predicted = outputs.max(1)
#     return predicted.numpy().tolist()

def load_model(model_path):
    intial_dir = os.getcwd()
    abs_model_path = os.path.abspath(model_path)
    os.chdir(abs_model_path)
    with open('metadata.json','r') as fp:
        metadata = jsonpickle.decode(json.load(fp))
    framework = metadata['framework']
    args_info = metadata['args_info']
    args_list = []
    for argName in args_info.keys():
      is_dir = args_info[argName]['is_dir']
      if not is_dir:
        with open(argName,'rb') as fp:
          arg = pickle.load(fp)
      else:
        arg = FRAMEWORKS[framework]['load'](argName)
      args_list.append(arg)
    model_class = pickle.loads(metadata['prediction_logic'])
    input_type = metadata['AbstractModelType']['input_type']
    output_type = metadata['AbstractModelType']['output_type']
    print(input_type)
    print(output_type)
    os.chdir(intial_dir)
    model = model_class(*args_list)
    return model, input_type, output_type

if __name__ == "__main__":
    print("Get pytorch model")
    # use a pretrained pytorch model
    # pytorch_model = models.squeezenet1_1(pretrained=True)
    # min_img_size = 224
    # transform = transforms.Compose([transforms.Resize(min_img_size),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize(
    #                                        mean=[0.485, 0.456, 0.406],
    #                                        std=[0.229, 0.224, 0.225])])
    # pytorch_model = models.squeezenet1_0(pretrained=True)

    print("Save prediction function")
    model_path = "../model_path"
    model, input_type, output_type = load_model(model_path)
    # model_path = pytorch_utils.save_python_function(predict_torch_model)
    # print("Save pytorch model")
    # pytorch_utils.save_pytorch_model(pytorch_model, model_path)
    print("done")

    print("Starting PyTorchContainer container..")
    model_name = "pytorch_model"
    model_version = 1
    # input_type = "strs"
    # output_type = "ints"
    rpc_service = container_rpc.RPCService(model_path, input_type, output_type)
    # try:
    #     model = PredictModelPytorch(transform,
    #                              pytorch_model)
    #     sys.stdout.flush()
    #     sys.stderr.flush()
    # except ImportError:
    #     sys.exit(IMPORT_ERROR_RETURN_CODE)
    rpc_service.start(model, model_name, model_version)
