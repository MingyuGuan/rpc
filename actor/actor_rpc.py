from __future__ import print_function
import zmq
# import threading
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

RPC_VERSION = 3

INPUT_TYPE_BYTES = 0
INPUT_TYPE_INTS = 1
INPUT_TYPE_FLOATS = 2
INPUT_TYPE_DOUBLES = 3
INPUT_TYPE_STRINGS = 4
INPUT_TYPE_IMAGES = 5

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

# Initial size of the buffer used for sending response and receiving
# request input content
INITIAL_CONTENT_BUFFER_SIZE = 1024
# Initial size of the buffers used for sending response
# header data and receiving request header data
INITIAL_HEADER_BUFFER_SIZE = 1024

INPUT_HEADER_DTYPE = np.dtype(np.uint64)

logger = logging.getLogger(__name__)

def string_to_input_type(input_str):
    input_str = input_str.strip().lower()
    byte_strs = ["b", "bytes", "byte"]
    int_strs = ["i", "ints", "int", "integer", "integers"]
    float_strs = ["f", "floats", "float"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str"]
    image_strs = ["img", "images", "image", "imgs"]

    if any(input_str == s for s in byte_strs):
        return INPUT_TYPE_BYTES
    elif any(input_str == s for s in int_strs):
        return INPUT_TYPE_INTS
    elif any(input_str == s for s in float_strs):
        return INPUT_TYPE_FLOATS
    elif any(input_str == s for s in double_strs):
        return INPUT_TYPE_DOUBLES
    elif any(input_str == s for s in string_strs):
        return INPUT_TYPE_STRINGS
    elif any(input_str == s for s in image_strs):
        return INPUT_TYPE_IMAGES
    else:
        return -1


def input_type_to_dtype(input_type):
    if input_type == INPUT_TYPE_BYTES:
        return np.dtype(np.int8)
    elif input_type == INPUT_TYPE_INTS:
        return np.dtype(np.int32)
    elif input_type == INPUT_TYPE_FLOATS:
        return np.dtype(np.float32)
    elif input_type == INPUT_TYPE_DOUBLES:
        return np.dtype(np.float64)
    elif input_type == INPUT_TYPE_STRINGS or input_type == INPUT_TYPE_IMAGES:
        return str


def input_type_to_string(input_type):
    if input_type == INPUT_TYPE_BYTES:
        return "bytes"
    elif input_type == INPUT_TYPE_INTS:
        return "ints"
    elif input_type == INPUT_TYPE_FLOATS:
        return "floats"
    elif input_type == INPUT_TYPE_DOUBLES:
        return "doubles"
    elif input_type == INPUT_TYPE_STRINGS:
        return "string"
    elif input_type == INPUT_TYPE_IMAGES:
        return "images"

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

class PredictionRequest:
    header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)

    def __init__(self, msg_id, input_type):
        """
        Parameters
        ----------
        msg_id : bytes
            The message id associated with the PredictRequest
            for which this is a response
        """
        self.msg_id = msg_id
        self.input_type = string_to_input_type(input_type)
        self.inputs = []
        self.num_inputs = 0

    def add_input(self, inp):
        """
        Parameters
        ----------
        input_ : string
        """
        # if not isinstance(inp, str):
        #     inp = str(inp)

        self.inputs.append(inp)
        self.num_inputs += 1

    def send(self, socket, event_history):
        """
        Sends the encapsulated response data via
        the specified socket

        Parameters
        ----------
        socket : zmq.Socket
        event_history : EventHistory
            The RPC event history that should be
            updated as a result of this operation
        """
        assert self.num_inputs > 0
        input_header, header_length_bytes = self._create_input_header()
        if sys.version_info < (3, 0):
            socket.send("", flags=zmq.SNDMORE)
        else:
            socket.send_string("", flags=zmq.SNDMORE)
        socket.send(struct.pack("<I", RPC_VERSION), zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)

        socket.send(struct.pack("<I", self.msg_id), flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", REQUEST_TYPE_PREDICT),
            flags=zmq.SNDMORE)

        socket.send(struct.pack("<Q", header_length_bytes), flags=zmq.SNDMORE)
        socket.send(input_header, flags=zmq.SNDMORE)

        for idx in range(self.num_inputs):
            if idx == self.num_inputs - 1:
                # Don't use the `SNDMORE` flag if
                # this is the last input being sent
                if self.input_type == INPUT_TYPE_STRINGS:
                    socket.send_string(self.inputs[idx])
                elif self.input_type == INPUT_TYPE_IMAGES:
                    socket.send(self.inputs[idx])
                else:
                    socket.send(self._format_input(self.inputs[idx]))
            else:
                if self.input_type == INPUT_TYPE_STRINGS:
                    socket.send_string(self.inputs[idx], flags=zmq.SNDMORE)
                elif self.input_type == INPUT_TYPE_IMAGES:
                    socket.send(self.inputs[idx], flags=zmq.SNDMORE)
                else:
                    socket.send(self._format_input(self.inputs[idx]), flags=zmq.SNDMORE)

        event_history.insert(EVENT_HISTORY_SENT_CONTAINER_CONTENT)

    def _format_input(self, inp):
        if self.input_type == INPUT_TYPE_BYTES:
            return struct.pack("<B", inp)
        elif self.input_type == INPUT_TYPE_INTS:
            return struct.pack("<I", inp)
        elif self.input_type == INPUT_TYPE_FLOATS:
            return struct.pack("<f", inp)
        elif self.input_type == INPUT_TYPE_DOUBLES:
            return struct.pack("<d", inp)
        elif self.input_type == INPUT_TYPE_STRINGS:
            return inp

    def _expand_buffer_if_necessary(self, size):
        """
        If necessary, expands the reusable input
        header buffer to accomodate content of the
        specified size

        size : int
            The size, in bytes, that the buffer must be
            able to store
        """
        if len(PredictionRequest.header_buffer) < size:
            PredictionRequest.header_buffer = bytearray(size * 2)

    def _create_input_header(self):
        """
        Returns
        ----------
        (bytearray, int)
            A tuple with the input header as the first
            element and the header length as the second
            element
        """
        header_length = BYTES_PER_LONG * (len(self.inputs) + 2)
        self._expand_buffer_if_necessary(header_length)
        header_idx = 0
        struct.pack_into("<Q", PredictionRequest.header_buffer, header_idx,
                         self.input_type)
        header_idx += BYTES_PER_LONG
        struct.pack_into("<Q", PredictionRequest.header_buffer, header_idx,
                         self.num_inputs)
        header_idx += BYTES_PER_LONG
        for inp in self.inputs:
            if isinstance(inp, str) or isinstance(inp, bytes):
                struct.pack_into("<Q", PredictionRequest.header_buffer,
                             header_idx, len(inp))
            else:
                struct.pack_into("<Q", PredictionRequest.header_buffer,
                             header_idx, input_type_to_dtype(self.input_type).itemsize)
            header_idx += BYTES_PER_LONG

        return PredictionRequest.header_buffer[:header_length], header_length

class ContainerMetadata():
    def __init__(self, model_name, model_version, model_input_type, container_rpc_version):
        self.model_name = model_name
        self.model_version  = model_version
        self.model_input_type = model_input_type
        self.container_rpc_version = container_rpc_version

class Actor():
    def __init__(self, context, actor_ip, actor_port):
        self.context = context
        self.actor_ip = actor_ip
        self.actor_port = actor_port
        self.event_history = EventHistory(EVENT_HISTORY_BUFFER_SIZE)
        # bool: whether a container is connected
        self.connected = False
        # unique identifier for each msg
        self.msg_id = 0

        self.output_header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)
        self.output_content_buffer = bytearray(
            INITIAL_CONTENT_BUFFER_SIZE)

    def validate_rpc_version(self, received_version):
        if received_version != RPC_VERSION:
            print(
                "ERROR: Received an RPC message with version: {clv} that does not match container version: {mcv}"
                .format(clv=received_version, mcv=RPC_VERSION))

    def send_heartbeat(self):
        if sys.version_info < (3, 0):
            self.socket.send("", zmq.SNDMORE)
        else:
            self.socket.send_string("", zmq.SNDMORE)
        # every outbount msg should begin with RPC version
        self.socket.send(struct.pack("<I", RPC_VERSION), zmq.SNDMORE)
        self.socket.send(struct.pack("<I", MESSAGE_TYPE_HEARTBEAT), zmq.SNDMORE)
        if not self.connected:
            self.socket.send(struct.pack("<I", HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA))
        else:
            self.socket.send(struct.pack("<I", HEARTBEAT_TYPE_KEEPALIVE))
        self.event_history.insert(EVENT_HISTORY_SENT_HEARTBEAT)

    def get_event_history(self):
        return self.event_history.get_events()

    def _get_prediction_request(self, input_type, inputs):
        """
        Returns
        -------
        PredictionRequest
            A prediction respuest containing inputs
        """
        request = PredictionRequest(self.msg_id, input_type)
        self.msg_id += 1
        for inp in inputs:
            request.add_input(inp)

        return request

    def send_prediction_request(self, input_type, inputs):
        t1 = datetime.now()
        request = self._get_prediction_request(input_type, inputs)
        request.send(self.socket, self.event_history)

        # recv result
        t2 = datetime.now()
        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]

        if msg_type == MESSAGE_TYPE_CONTAINER_CONTENT:
            msg_id_bytes = self.socket.recv()
            msg_id = int(struct.unpack("<I", msg_id_bytes)[0])
            print("Got response for request message %d " % msg_id)

            output_header_size_raw = self.socket.recv()
            output_header_size_bytes = struct.unpack(
                "<Q", output_header_size_raw)[0]

            typed_output_header_size = int(
                output_header_size_bytes /
                BYTES_PER_LONG)

            # adjust output buffer size
            if len(self.output_header_buffer
                   ) < output_header_size_bytes:
                self.ourput_header_buffer = bytearray(
                    output_header_size_bytes * 2)

            # While this procedure still incurs a copy, it saves a potentially
            # costly memory allocation by ZMQ. This savings only occurs
            # if the output header did not have to be resized
            output_header_view = memoryview(
                self.output_header_buffer)[:output_header_size_bytes]
            output_header_content = self.socket.recv(copy=False).buffer
            output_header_view[:
                              output_header_size_bytes] = output_header_content

            parsed_output_header = np.frombuffer(
                self.output_header_buffer,
                dtype=INPUT_HEADER_DTYPE)[:typed_output_header_size]

            # output_type should always be string
            num_outputs, output_sizes = parsed_output_header[
                0], parsed_output_header[1:]

            output_sizes = [
                int(output_size) for output_size in output_sizes
            ]

            outputs = self.recv_outputs_content(num_outputs, output_sizes)

            t3 = datetime.now()
            send_time = (t2 - t1).total_seconds()
            recv_time = (t3 - t2).total_seconds()
            print("send: %f s, recv: %f s" %
                              (send_time, recv_time))
            print("OUTPUTS:")
            print('\n'.join(outputs))
            return outputs, num_outputs
        else:
            print("Wrong message type %d, should be container content msg" % msg_type)
            raise

    def recv_outputs_content(self, num_outputs, output_sizes):
        # # Create an empty numpy array that will contain
        # # output string references
        # outputs = [np.empty(num_outputs, dtype=object)]
        # for i in range(num_outputs):
        #     # Obtain a memoryview of the received message's
        #     # ZMQ frame buffer
        #     output_item_buffer = socket.recv(copy=False).buffer
        #     # Copy the memoryview content into a string object
        #     output_str = output_item_buffer.tobytes()
        #     outputs[i] = output_str

        outputs = []
        for i in range(num_outputs):
            output_str = self.socket.recv_string()
            outputs.append(output_str)

        return outputs          

    def recv_prediction_response(self, msg_id):
        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]

        if msg_type == MESSAGE_TYPE_CONTAINER_CONTENT:
            msg_id_bytes = socket.recv()
            msg_id = int(struct.unpack("<I", msg_id_bytes)[0])
            print("Got response for request message %d " % msg_id)
            # list of byte arrays
            response_header = socket.recv()
            response_type = struct.unpack("<I", response_header)[0]

            if response_type == REQUEST_TYPE_PREDICT:
                pass
    
    def connect_to_container(self):
        print("Connecting to container...")
        self.actor_address = "tcp://{0}:{1}".format(self.actor_ip,
                                                 self.actor_port)
        sys.stdout.flush()
        sys.stderr.flush()

        self.input_header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)
        self.input_content_buffer = bytearray(
            INITIAL_CONTENT_BUFFER_SIZE)

        self.socket = self.context.socket(zmq.REP)
        self.socket.connect(self.actor_address)

        # recv heartbeat from container
        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]

        # send heartbeat response to ask for container metadata
        if msg_type == MESSAGE_TYPE_HEARTBEAT:
            self.event_history.insert(EVENT_HISTORY_RECEIVED_HEARTBEAT)
            sys.stdout.flush()
            sys.stderr.flush()
            if self.connected:
                print("Cannot connect to another container when already connected with one!")
                raise
            else:
                self.send_heartbeat()
        else:
            print("Wrong message type %d, should be heartbeat" % msg_type)
            raise

        self.socket.recv()
        msg_type_bytes = self.socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]
        if msg_type == MESSAGE_TYPE_NEW_CONTAINER:
            self.event_history.insert(
                EVENT_HISTORY_RECEIVED_CONTAINER_METADATA)
            print(
                "Received new container message from container!"
            )
            model_name = self.socket.recv_string()
            model_version = self.socket.recv_string()
            model_input_type = self.socket.recv_string()
            container_rpc_version_bytes = self.socket.recv()
            container_rpc_version = struct.unpack("<I", container_rpc_version_bytes)[0]

            self.container_meta = ContainerMetadata(model_name, model_version, model_input_type, container_rpc_version)
            self.connected = True
            print("Actor Model " + model_name + " is connected")
        else:
            print("Wrong message type %d, should be new container msg" % msg_type)
            raise

    def recv_string_content(self, num_inputs, input_sizes):
        # Create an empty numpy array that will contain
        # input string references
        inputs = np.empty(num_inputs, dtype=object)
        for i in range(num_inputs):
            # Obtain a memoryview of the received message's
            # ZMQ frame buffer
            input_item_buffer = self.socket.recv(copy=False).buffer
            # Copy the memoryview content into a string object
            input_str = input_item_buffer.tobytes()
            inputs[i] = input_str

        return inputs

    def recv_primitive_content(self, num_inputs, input_sizes,
                               input_dtype):
        def recv_different_lengths():
            # Create an empty numpy array that will contain
            # input array references
            inputs = np.empty(num_inputs, dtype=object)
            for i in range(num_inputs):
                # Receive input data and copy it into a byte
                # buffer that can be parsed into a writeable
                # array
                input_item_buffer = self.socket.recv(copy=True)
                input_item = np.frombuffer(
                    input_item_buffer, dtype=input_dtype)
                inputs[i] = input_item

            return inputs

        def recv_same_lengths():
            input_type_size_bytes = input_dtype.itemsize
            input_content_size_bytes = sum(input_sizes)
            typed_input_content_size = int(
                input_content_size_bytes / input_type_size_bytes)

            if len(self.input_content_buffer) < input_content_size_bytes:
                self.input_content_buffer = bytearray(
                    input_content_size_bytes * 2)

            input_content_view = memoryview(
                self.input_content_buffer)[:input_content_size_bytes]

            item_start_idx = 0
            for i in range(num_inputs):
                input_size = input_sizes[i]
                # Obtain a memoryview of the received message's
                # ZMQ frame buffer
                input_item_buffer = self.socket.recv(copy=False).buffer
                # Copy the memoryview content into a pre-allocated content buffer
                input_content_view[item_start_idx:item_start_idx +
                                   input_size] = input_item_buffer
                item_start_idx += input_size

            # Reinterpret the content buffer as a typed numpy array
            inputs = np.frombuffer(
                self.input_content_buffer,
                dtype=input_dtype)[:typed_input_content_size]

            # All inputs are of the same size, so we can use
            # np.reshape to construct an input matrix
            inputs = np.reshape(inputs, (len(input_sizes), -1))

            return inputs

        if len(set(input_sizes)) == 1:
            return recv_same_lengths()
        else:
            return recv_different_lengths()

class RPCService:
    def get_event_history(self):
        if self.actor:
            return self.actor.get_event_history()
        else:
            print("Cannot retrieve message history for inactive RPC service!")
            raise

    def start(self, actor_port, actor_ip = "127.0.0.1", input_type="doubles"):
        try:
            ip = socket.gethostbyname(actor_ip)
        except socket.error as e:
            print("Error resolving %s: %s" % (self.host, e))
            sys.exit(1)
        context = zmq.Context()
        self.actor = Actor(context, ip, actor_port)

    def connect(self):
        self.actor.connect_to_container() 

    def send_prediction_request(self, input_type, inputs):
        # return request result
        return self.actor.send_prediction_request(input_type, inputs)
