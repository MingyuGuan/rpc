from __future__ import print_function
import actor_rpc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import base64
import docker
import time
from datetime import datetime

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

    print("Sending 100 request to plot cdf..")
    time_arr = []
    inputs = [img1]
    for i in range(100):
        t1 = datetime.now()
        num_outputs, outputs = rpc_service.send_prediction_request(input_type, inputs)
        t2 = datetime.now()
        time = (t2 - t1).total_seconds()
        time_arr.append(time)

    sorted_data = np.sort(time_arr)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)

    plt.xlabel('time(s)')
    plt.ylabel('CDF')
    plt.plot(sorted_data,yvals)

    plt.show()
