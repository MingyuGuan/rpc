from __future__ import print_function
import actor_rpc
import os
import sys
import numpy as np
import base64
import docker
import time

if __name__ == "__main__":
    print("Starting RPC actor..")
    sys.stdout.flush()
    sys.stderr.flush()
    rpc_service = actor_rpc.RPCService()
    # start rpc service
    port = 7000
    rpc_service.start(port)

    print("Building and Deploying Container")
    name = "pytorch_container"
    version = 1
    model_path = "pytorch_container/"
    prediction_file = "pytorch_container.py"
    ports = {'7000': 7000}

    docker_client = docker.from_env()
    image = rpc_service.build_model(docker_client, 
                        name,
                        version,
                        model_path,
                        prediction_file,
                        port=7000,
                        base_image="alice97/serve-base",
                        container_registry=None,
                        pkgs_to_install=None)
    print("Create image successfully!")
    rpc_service.run_container(docker_client, image, detach=True, ports=ports)
    print("{} is is running..".format(name))

    # # wait container running stablely
    # time.sleep(5)

    print("Connecting to conatiner...")
    #connect to container
    rpc_service.connect_to_container()

    print("\nSending a request with inputs: dog.jpg, cat.jpg")
    # send first request
    input_type = "imgs"
    img1 = base64.b64encode(open("images/dog.jpg", "rb").read())
    img2 = base64.b64encode(open("images/cat.jpg", "rb").read())

    inputs = [img1, img2]
    num_outputs, outputs = rpc_service.send_prediction_request(input_type, inputs)
