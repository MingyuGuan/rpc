from __future__ import print_function
import actor_rpc
import os
import sys
import numpy as np

if __name__ == "__main__":
    print("Starting RPC actor..")
    rpc_service = actor_rpc.RPCService()
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
    num_outputs, outputs = rpc_service.send_prediction_request(input_type, inputs)
