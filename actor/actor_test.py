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

    # start rpc service
    rpc_service = actor_rpc.RPCService()
    port = 7000
    rpc_service.start(port)

    # # deploy container
    # print("Building and Deploying Container")
    # name = "pytorch_container"
    # version = 1
    # model_path = "pytorch_container/"
    # prediction_file = "pytorch_container.py"
    # ports = {'7000': 7000}

    # docker_client = docker.from_env()
    # image = rpc_service.build_model(docker_client, 
    #                     name,
    #                     version,
    #                     model_path,
    #                     prediction_file,
    #                     port=7000,
    #                     base_image="alice97/serve-base",
    #                     container_registry=None,
    #                     pkgs_to_install=None)
    # print("Create image successfully!")
    # rpc_service.run_container(docker_client, image, detach=True, ports=ports)
    # print("{} is is running..".format(name))

    # connect to container
    print("Connecting to conatiner...")
    rpc_service.connect_to_container()

    # send first request
    print("\nSending a request with inputs: elephant.jpg")
    input_type = "strs"
    img1 = base64.b64encode(open("images/elephant.jpg", "rb").read())
    # img2 = base64.b64encode(open("images/cat.jpg", "rb").read())

    inputs = [img1]
    outputs, num_outputs = rpc_service.send_prediction_request(input_type, inputs)

    print("\nOUTPUTS:")
    print('\n'.join(map(str, outputs)))

    # stop contaienr
    # print("\nStopping container..")
    # rpc_service.stop_container()
    # print("Successfully quit")

    # print("\nSending 200 request to plot cdf..")
    # time_arr = []
    # inputs = [img1]
    # for i in range(200):
    #     t1 = datetime.now()
    #     num_outputs, outputs = rpc_service.send_prediction_request(input_type, inputs)
    #     t2 = datetime.now()
    #     latency = (t2 - t1).total_seconds()
    #     time_arr.append(latency)
    #     time.sleep(0.01)

    # sorted_data = np.sort(time_arr)
    # yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)

    # axes = plt.gca()
    # xmin = 0.1
    # xmax = 0.7
    # axes.set_xlim([xmin,xmax])
    # plt.xlabel('time(s)')
    # plt.ylabel('CDF')
    # plt.plot(sorted_data,yvals)

    # plt.show()
