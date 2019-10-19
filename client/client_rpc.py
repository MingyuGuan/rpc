from __future__ import print_function
import zmq
import threading
import numpy as np
import struct
import time
from datetime import datetime
import socket
import sys
import os
import yaml
import logging
from collections import deque
if sys.version_info < (3, 0):
    from subprocess32 import Popen, PIPE
else:
    from subprocess import Popen, PIPE
from prometheus_client import start_http_server
from prometheus_client.core import Counter, Gauge, Histogram, Summary
import clipper_admin.metrics as metrics

RPC_VERSION = 3

INPUT_TYPE_BYTES = 0
INPUT_TYPE_INTS = 1
INPUT_TYPE_FLOATS = 2
INPUT_TYPE_DOUBLES = 3
INPUT_TYPE_STRINGS = 4

REQUEST_TYPE_PREDICT = 0
REQUEST_TYPE_FEEDBACK = 1

MESSAGE_TYPE_NEW_CONTAINER = 0
MESSAGE_TYPE_CONTAINER_CONTENT = 1
MESSAGE_TYPE_HEARTBEAT = 2

HEARTBEAT_TYPE_KEEPALIVE = 0
HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA = 1

SOCKET_POLLING_TIMEOUT_MILLIS = 5000
SOCKET_ACTIVITY_TIMEOUT_MILLIS = 30000

EVENT_HISTORY_BUFFER_SIZE = 30

EVENT_HISTORY_SENT_HEARTBEAT = 1
EVENT_HISTORY_RECEIVED_HEARTBEAT = 2
EVENT_HISTORY_SENT_CONTAINER_METADATA = 3
EVENT_HISTORY_RECEIVED_CONTAINER_METADATA = 4
EVENT_HISTORY_SENT_CONTAINER_CONTENT = 5
EVENT_HISTORY_RECEIVED_CONTAINER_CONTENT = 6

MAXIMUM_UTF_8_CHAR_LENGTH_BYTES = 4
BYTES_PER_LONG = 8

# Initial size of the buffer used for receiving
# request input content
INITIAL_INPUT_CONTENT_BUFFER_SIZE = 1024
# Initial size of the buffers used for sending response
# header data and receiving request header data
INITIAL_HEADER_BUFFER_SIZE = 1024

INPUT_HEADER_DTYPE = np.dtype(np.uint64)

logger = logging.getLogger(__name__)

class EventHistory:
    def __init__(self, size):
        self.history_buffer = deque(maxlen=size)

    def insert(self, msg_type):
        curr_time_millis = time.time() * 1000
        self.history_buffer.append((curr_time_millis, msg_type))

    def get_events(self):
        return self.history_buffer


class PredictionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class Client(threading.Thread):
    def __init__(self, context, clipper_ip, clipper_port):
        threading.Thread.__init__(self)
        self.context = context
        self.clipper_ip = clipper_ip
        self.clipper_port = clipper_port
        self.event_history = EventHistory(EVENT_HISTORY_BUFFER_SIZE)

    def send_heartbeat(self, socket):
        if sys.version_info < (3, 0):
            socket.send("", zmq.SNDMORE)
        else:
            socket.send_string("", zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_HEARTBEAT))
        
        self.event_history.insert(EVENT_HISTORY_SENT_HEARTBEAT)
        print("Sent heartbeat!")

    def handle_heartbeat(self, socket):
    	msg_heartbeat_type_bytes = msgsocket.recv()
    	msg_heartbeat_type = struct.unpack("<I", msg_heartbeat_type_bytes)[0]
    	return msg_heartbeat_type == HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA

