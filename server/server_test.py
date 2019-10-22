from __future__ import print_function
import server_rpc
import os
import sys
import numpy as np

class RPCTestModel(server_rpc.Model):
    def __init__(self, rpc_service, prediction):
        self._rpc_service = rpc_service
        self.prediction = prediction

    def predict(self, inputs):
        return [self.prediction] * len(inputs)



if __name__ == "__main__":
    print("Starting Client")
    rpc_service = server_rpc.RPCService()

    model = RPCTestModel(rpc_service, "0.88")
    model_name = "server"
    model_version = 1
    port = 7000

    sys.stdout.flush()
    sys.stderr.flush()
    # start rpc service
    rpc_service.start(port)
    
    #connect to container
    rpc_service.connect()

    # send first request
    input_type = "doubles"
    inputs = [8.8, 2.4, 5.7]
    rpc_service.send_prediction_request(input_type, inputs)
