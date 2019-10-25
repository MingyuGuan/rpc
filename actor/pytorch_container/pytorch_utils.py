from __future__ import print_function, with_statement, absolute_import
import shutil
import logging
from cloudpickle import CloudPickler
import os
import tempfile
import sys
import torch
cur_dir = os.path.dirname(os.path.abspath(__file__))

PYTORCH_WEIGHTS_RELATIVE_PATH = "pytorch_weights.pkl"
PYTORCH_MODEL_RELATIVE_PATH = "pytorch_model.pkl"

if sys.version_info < (3, 0):
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    from io import BytesIO as StringIO
    PY3 = True

logger = logging.getLogger(__name__)


def serialize_object(obj):
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(obj)
    return s.getvalue()


def save_python_function(func):
    predict_fname = "func.pkl"

    # Serialize function
    s = StringIO()
    c = CloudPickler(s, 2)
    c.dump(func)
    serialized_prediction_function = s.getvalue()

    # Set up serialization directory
    serialization_dir = os.path.abspath(tempfile.mkdtemp(suffix='srtml'))
    logger.info("Saving function to {}".format(serialization_dir))

    # Write out function serialization
    func_file_path = os.path.join(serialization_dir, predict_fname)
    if sys.version_info < (3, 0):
        with open(func_file_path, "w") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    else:
        with open(func_file_path, "wb") as serialized_function_file:
            serialized_function_file.write(serialized_prediction_function)
    logging.info("Serialized and supplied predict function")
    return serialization_dir

def save_pytorch_model(pytorch_model, path):
    # save Torch model
    torch_weights_save_loc = os.path.join(path,
                                          PYTORCH_WEIGHTS_RELATIVE_PATH)

    torch_model_save_loc = os.path.join(path,
                                        PYTORCH_MODEL_RELATIVE_PATH)

    try:
        torch.save(pytorch_model.state_dict(), torch_weights_save_loc)
        serialized_model = serialize_object(pytorch_model)
        with open(torch_model_save_loc, "wb") as serialized_model_file:
            serialized_model_file.write(serialized_model)
        logger.info("Torch model saved")
    except Exception as e:
        raise Exception("Error saving torch model: %s" % e)

def remove_temp_files(path):
    # Remove temp files
    shutil.rmtree(serialization_dir)