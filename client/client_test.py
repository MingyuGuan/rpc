from __future__ import print_function
import client_rpc
import os
import sys
import numpy as np

if __name__ == '__main__':
	print("Start RPC Client...")

	client_ip = "127.0.0.1"
	client_port = 7000

	client_name = "mingyu"

	connected = False
	clipper_address = "tcp://{0}:{1}".format(self.clipper_ip,
                                             self.clipper_port)
	
	while True:
