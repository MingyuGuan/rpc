from __future__ import print_function
import actor_rpc
import os
import sys
import numpy as np
import base64

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
    input_type = "imgs"
    img1 = base64.b64encode(open("dog.jpg", "rb").read())
    img2 = base64.b64encode(open("cat.jpg", "rb").read())

    inputs = [img1, img2]
    num_outputs, outputs = rpc_service.send_prediction_request(input_type, inputs)
