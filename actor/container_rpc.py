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
# import yaml
import logging
import base64
from collections import deque
if sys.version_info < (3, 0):
    from subprocess32 import Popen, PIPE
else:
    from subprocess import Popen, PIPE
import zlib
import pickle

import cloudpickle

import json
import jsonpickle
import mlflow.pytorch 
import mlflow.tensorflow

RPC_VERSION = 3

INPUT_TYPE_BYTES = 0
INPUT_TYPE_INTS = 1
INPUT_TYPE_FLOATS = 2
INPUT_TYPE_DOUBLES = 3
INPUT_TYPE_STRINGS = 4
INPUT_TYPE_ABSTRACT = 5

OUTPUT_TYPE_BYTES = 0
OUTPUT_TYPE_INTS = 1
OUTPUT_TYPE_FLOATS = 2
OUTPUT_TYPE_DOUBLES = 3
OUTPUT_TYPE_STRINGS = 4
OUTPUT_TYPE_ABSTRACT = 5

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

FRAMEWORKS = {
  'torch'    : {
                  'load': mlflow.pytorch.load_model,
                  'save': mlflow.pytorch.save_model
  },
  'tensorflow' : {
                  'load' : mlflow.tensorflow.load_model,
                  'save' : mlflow.tensorflow.save_model
  },
}

def string_to_input_type(input_str):
    input_str = input_str.strip().lower()
    byte_strs = ["b", "bytes", "byte"]
    int_strs = ["i", "ints", "int", "integer", "integers", "<class \'int\'>"]
    float_strs = ["f", "floats", "float", "<class \'float\'>"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str", "<class \'str\'>"]

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
    else:
        return INPUT_TYPE_ABSTRACT


def input_type_to_dtype(input_type):
    if input_type == INPUT_TYPE_BYTES:
        return np.dtype(np.int8)
    elif input_type == INPUT_TYPE_INTS:
        return np.dtype(np.int32)
    elif input_type == INPUT_TYPE_FLOATS:
        return np.dtype(np.float32)
    elif input_type == INPUT_TYPE_DOUBLES:
        return np.dtype(np.float64)
    elif input_type == INPUT_TYPE_STRINGS:
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
    else:
        return "abstract"

def string_to_output_type(output_str):
    output_str = output_str.strip().lower()
    byte_strs = ["b", "bytes", "byte"]
    int_strs = ["i", "ints", "int", "integer", "integers", "<class \'int\'>"]
    float_strs = ["f", "floats", "float", "<class \'float\'>"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str", "<class \'str\'>"]

    if any(output_str == s for s in byte_strs):
        return OUTPUT_TYPE_BYTES
    elif any(output_str == s for s in int_strs):
        return OUTPUT_TYPE_INTS
    elif any(output_str == s for s in float_strs):
        return OUTPUT_TYPE_FLOATS
    elif any(output_str == s for s in double_strs):
        return OUTPUT_TYPE_DOUBLES
    elif any(output_str == s for s in string_strs):
        return OUTPUT_TYPE_STRINGS
    else:
        return OUTPUT_TYPE_ABSTRACT

def output_type_to_dtype(output_type):
    if output_type == OUTPUT_TYPE_BYTES:
        return np.dtype(np.int8)
    elif output_type == OUTPUT_TYPE_INTS:
        return np.dtype(np.int32)
    elif output_type == OUTPUT_TYPE_FLOATS:
        return np.dtype(np.float32)
    elif output_type == OUTPUT_TYPE_DOUBLES:
        return np.dtype(np.float64)
    elif output_type == OUTPUT_TYPE_STRINGS:
        return str


def output_type_to_string(output_type):
    if output_type == OUTPUT_TYPE_BYTES:
        return "bytes"
    elif output_type == OUTPUT_TYPE_INTS:
        return "ints"
    elif output_type == OUTPUT_TYPE_FLOATS:
        return "floats"
    elif output_type == OUTPUT_TYPE_DOUBLES:
        return "doubles"
    elif output_type == OUTPUT_TYPE_STRINGS:
        return "string"
    else:
        return "abstract"


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


class Server():
    def __init__(self, context, clipper_ip, clipper_port):
        # threading.Thread.__init__(self)
        self.context = context
        self.clipper_ip = clipper_ip
        self.clipper_port = clipper_port
        self.event_history = EventHistory(EVENT_HISTORY_BUFFER_SIZE)

    def validate_rpc_version(self, received_version):
        if received_version != RPC_VERSION:
            print(
                "ERROR: Received an RPC message with version: {clv} that does not match container version: {mcv}"
                .format(clv=received_version, mcv=RPC_VERSION))

    def handle_prediction_request(self, prediction_request):
        """
        Returns
        -------
        PredictionResponse
            A prediction response containing an output
            for each input included in the specified
            predict response
        """
        predict_fn = self.get_prediction_function()
        outputs = predict_fn(prediction_request.inputs)

        # Type check the outputs:
        if not type(outputs) == list:
            raise PredictionError("Model did not return a list")
        if len(outputs) != len(prediction_request.inputs):
            raise PredictionError(
                "Expected model to return %d outputs, found %d outputs" %
                (len(prediction_request.inputs), len(outputs)))
        response = PredictionResponse(prediction_request.msg_id, self.model_output_type)
        for output in outputs:
            response.add_output(output)

        return response

    def handle_feedback_request(self, feedback_request):
        """
        Returns
        -------
        FeedbackResponse
            A feedback response corresponding
            to the specified feedback request
        """
        response = FeedbackResponse(feedback_request.msg_id, "ACK")
        return response

    def get_prediction_function(self):
        return self.model.__call__

    def get_event_history(self):
        return self.event_history.get_events()

    def run(self):
        print("Serving predictions for {0} input type.".format(
            input_type_to_string(self.model_input_type)))
        # connected = False
        clipper_address = "tcp://{0}:{1}".format(self.clipper_ip,
                                                 self.clipper_port)
        # poller = zmq.Poller()
        sys.stdout.flush()
        sys.stderr.flush()

        self.input_header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)
        self.input_content_buffer = bytearray(
            INITIAL_INPUT_CONTENT_BUFFER_SIZE)

        while True:
            socket = self.context.socket(zmq.REQ)
            socket.bind(clipper_address)
            self.send_heartbeat(socket)
            while True:
                t1 = datetime.now()
                # Receive delimiter between routing identity and content
                socket.recv()
                rpc_version_bytes = socket.recv()
                rpc_version = struct.unpack("<I", rpc_version_bytes)[0]
                self.validate_rpc_version(rpc_version)
                print("recv rpc version: " + str(rpc_version))
                msg_type_bytes = socket.recv()
                msg_type = struct.unpack("<I", msg_type_bytes)[0]
                if msg_type == MESSAGE_TYPE_HEARTBEAT:
                    self.event_history.insert(EVENT_HISTORY_RECEIVED_HEARTBEAT)
                    print("Received heartbeat!")
                    sys.stdout.flush()
                    sys.stderr.flush()
                    heartbeat_type_bytes = socket.recv()
                    heartbeat_type = struct.unpack("<I",
                                                   heartbeat_type_bytes)[0]
                    if heartbeat_type == HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA:
                        self.send_container_metadata(socket)
                    continue
                elif msg_type == MESSAGE_TYPE_NEW_CONTAINER:
                    self.event_history.insert(
                        EVENT_HISTORY_RECEIVED_CONTAINER_METADATA)
                    print(
                        "Received erroneous new container message from Clipper!"
                    )
                    continue
                elif msg_type == MESSAGE_TYPE_CONTAINER_CONTENT:
                    self.event_history.insert(
                        EVENT_HISTORY_RECEIVED_CONTAINER_CONTENT)
                    msg_id_bytes = socket.recv()
                    msg_id = int(struct.unpack("<I", msg_id_bytes)[0])

                    print("Got start of message %d " % msg_id)
                    # list of byte arrays
                    request_header = socket.recv()
                    request_type = struct.unpack("<I", request_header)[0]

                    if request_type == REQUEST_TYPE_PREDICT:
                        input_header_size_raw = socket.recv()
                        input_header_size_bytes = struct.unpack(
                            "<Q", input_header_size_raw)[0]

                        typed_input_header_size = int(
                            input_header_size_bytes /
                            INPUT_HEADER_DTYPE.itemsize)

                        # adjust input buffer size
                        if len(self.input_header_buffer
                               ) < input_header_size_bytes:
                            self.input_header_buffer = bytearray(
                                input_header_size_bytes * 2)

                        # While this procedure still incurs a copy, it saves a potentially
                        # costly memory allocation by ZMQ. This savings only occurs
                        # if the input header did not have to be resized
                        input_header_view = memoryview(
                            self.input_header_buffer)[:input_header_size_bytes]
                        input_header_content = socket.recv(copy=False).buffer
                        input_header_view[:
                                          input_header_size_bytes] = input_header_content

                        parsed_input_header = np.frombuffer(
                            self.input_header_buffer,
                            dtype=INPUT_HEADER_DTYPE)[:typed_input_header_size]

                        input_type, num_inputs, input_sizes = parsed_input_header[
                            0], parsed_input_header[1], parsed_input_header[2:]

                        input_dtype = input_type_to_dtype(input_type)
                        input_sizes = [
                            int(inp_size) for inp_size in input_sizes
                        ]

                        print("recv request with " + str(num_inputs) + " inputs of " + input_type_to_string(input_type) + " type")
                        print(input_type)
                        if input_type == INPUT_TYPE_STRINGS:
                            inputs = self.recv_string_content(
                                socket, num_inputs, input_sizes)
                        elif input_type == INPUT_TYPE_ABSTRACT:
                            inputs = self.recv_abstract_content(socket, num_inputs)
                        else:
                            inputs = self.recv_primitive_content(
                                socket, num_inputs, input_sizes, input_dtype)

                        t2 = datetime.now()

                        if int(input_type) != int(self.model_input_type):
                            print((
                                "Received incorrect input. Expected {expected}, "
                                "received {received}").format(
                                    expected=input_type_to_string(
                                        int(self.model_input_type)),
                                    received=input_type_to_string(
                                        int(input_type))))
                            raise

                        t3 = datetime.now()

                        prediction_request = PredictionRequest(
                            msg_id_bytes, inputs)
                        response = self.handle_prediction_request(
                            prediction_request)

                        t4 = datetime.now()

                        response.send(socket, self.event_history)

                        recv_time = (t2 - t1).total_seconds()
                        parse_time = (t3 - t2).total_seconds()
                        handle_time = (t4 - t3).total_seconds()

                        print("recv: %f s, parse: %f s, handle: %f s" %
                              (recv_time, parse_time, handle_time))

                        sys.stdout.flush()
                        sys.stderr.flush()

                    else:
                        feedback_request = FeedbackRequest(msg_id_bytes, [])
                        response = self.handle_feedback_request(received_msg)
                        response.send(socket, self.event_history)
                        print("recv: %f s" % ((t2 - t1).total_seconds()))

                sys.stdout.flush()
                sys.stderr.flush()

    def recv_string_content(self, socket, num_inputs, input_sizes):
        # Create an empty numpy array that will contain
        # input string references
        inputs = np.empty(num_inputs, dtype=object)
        for i in range(num_inputs):
            # Obtain a memoryview of the received message's
            # ZMQ frame buffer
            input_item_buffer = socket.recv(copy=False).buffer
            # Copy the memoryview content into a string object
            input_str = bytearray(input_item_buffer.tobytes())
            inputs[i] = input_str

        return inputs

    def recv_abstract_content(self, socket, num_inputs):
        # Create an empty numpy array that will contain
        # input string references
        inputs = []
        for i in range(num_inputs):
            # Obtain a memoryview of the received message's
            # ZMQ frame buffer
            input_compress = socket.recv()
            # Copy the memoryview content into a string object
            p_input = zlib.decompress(input_compress)
            inp = pickle.loads(p_input)
            inputs[i] = inp
        return inputs

    def recv_primitive_content(self, socket, num_inputs, input_sizes,
                               input_dtype):
        def recv_different_lengths():
            # Create an empty numpy array that will contain
            # input array references
            inputs = np.empty(num_inputs, dtype=object)
            for i in range(num_inputs):
                # Receive input data and copy it into a byte
                # buffer that can be parsed into a writeable
                # array
                input_item_buffer = socket.recv(copy=True)
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
            print("num_inputs is %d" % num_inputs)
            for i in range(num_inputs):
                input_size = input_sizes[i]
                # Obtain a memoryview of the received message's
                # ZMQ frame buffer
                input_item_buffer = socket.recv(copy=False).buffer
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
            print("length of inputs is %d" % len(inputs))
            inputs = np.reshape(inputs, (len(input_sizes), -1))

            return inputs

        if len(set(input_sizes)) == 1:
            return recv_same_lengths()
        else:
            return recv_different_lengths()

    def send_container_metadata(self, socket):
        if sys.version_info < (3, 0):
            socket.send("", zmq.SNDMORE)
        else:
            socket.send("".encode('utf-8'), zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_CONTAINER), zmq.SNDMORE)
        socket.send_string(str(self.model_input_type), zmq.SNDMORE)
        socket.send_string(str(self.model_output_type), zmq.SNDMORE)
        socket.send(struct.pack("<I", RPC_VERSION))
        self.event_history.insert(EVENT_HISTORY_SENT_CONTAINER_METADATA)
        print("Sent container metadata!")
        sys.stdout.flush()
        sys.stderr.flush()

    def send_heartbeat(self, socket):
        if sys.version_info < (3, 0):
            socket.send("", zmq.SNDMORE)
        else:
            socket.send_string("", zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_HEARTBEAT))
        self.event_history.insert(EVENT_HISTORY_SENT_HEARTBEAT)
        print("Sent heartbeat!")


class PredictionRequest:
    """
    Parameters
    ----------
    msg_id : bytes
        The raw message id associated with the RPC
        prediction request message
    inputs :
        One of [[byte]], [[int]], [[float]], [[double]], [string]
    """

    def __init__(self, msg_id, inputs):
        self.msg_id = msg_id
        self.inputs = inputs

    def __str__(self):
        return self.inputs


class PredictionResponse:
    header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)

    def __init__(self, msg_id, output_type):
        """
        Parameters
        ----------
        msg_id : bytes
            The message id associated with the PredictRequest
            for which this is a response
        """
        self.msg_id = msg_id
        self.outputs = []
        self.num_outputs = 0
        self.output_type = output_type

    def add_output(self, output):
        """
        Parameters
        ----------
        output : string
        """
        # if not isinstance(output, str):
        #     output = str(output)
        self.outputs.append(output)
        self.num_outputs += 1

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
        assert self.num_outputs > 0
        output_header, header_length_bytes = self._create_output_header()
        if sys.version_info < (3, 0):
            socket.send("", flags=zmq.SNDMORE)
        else:
            socket.send_string("", flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)
        socket.send(self.msg_id, flags=zmq.SNDMORE)
        socket.send(struct.pack("<Q", header_length_bytes), flags=zmq.SNDMORE)
        socket.send(output_header, flags=zmq.SNDMORE)
        for idx in range(self.num_outputs):
            if idx == self.num_outputs - 1:
                # Don't use the `SNDMORE` flag if
                # this is the last output being sent
                if self.output_type == OUTPUT_TYPE_STRINGS:
                    socket.send_string(self.outputs[idx])
                else:
                    socket.send(self._format_output(self.outputs[idx]))
            else:
                if self.output_type == OUTPUT_TYPE_STRINGS:
                    socket.send_string(self.outputs[idx], flags=zmq.SNDMORE)
                else:
                    socket.send(self._format_output(self.outputs[idx]), flags=zmq.SNDMORE)

        event_history.insert(EVENT_HISTORY_SENT_CONTAINER_CONTENT)

    def _format_output(self, outp):
        if self.output_type == OUTPUT_TYPE_BYTES:
            return struct.pack("<B", outp)
        elif self.output_type == OUTPUT_TYPE_INTS:
            return struct.pack("<I", outp)
        elif self.output_type == OUTPUT_TYPE_FLOATS:
            return struct.pack("<f", outp)
        elif self.output_type == OUTPUT_TYPE_DOUBLES:
            return struct.pack("<d", outp)
        elif self.output_type == OUTPUT_TYPE_STRINGS:
            return outp
        elif self.output_type == OUTPUT_TYPE_ABSTRACT:
            p = pickle.dumps(outp)
            z = zlib.compress(p)
            return z

    def _expand_buffer_if_necessary(self, size):
        """
        If necessary, expands the reusable output
        header buffer to accomodate content of the
        specified size

        size : int
            The size, in bytes, that the buffer must be
            able to store
        """
        if len(PredictionResponse.header_buffer) < size:
            PredictionResponse.header_buffer = bytearray(size * 2)

    def _create_output_header(self):
        """
        Returns
        ----------
        (bytearray, int)
            A tuple with the output header as the first
            element and the header length as the second
            element
        """
        header_length = BYTES_PER_LONG * (len(self.outputs) + 1)
        self._expand_buffer_if_necessary(header_length)
        header_idx = 0
        struct.pack_into("<Q", PredictionResponse.header_buffer, header_idx,
                         self.num_outputs)
        header_idx += BYTES_PER_LONG
        for output in self.outputs:
            if isinstance(output, str) or isinstance(output, bytes):
                struct.pack_into("<Q", PredictionResponse.header_buffer,
                                 header_idx, len(output))
            elif self.output_type == INPUT_TYPE_ABSTRACT:
                struct.pack_into("<Q", PredictionResponse.header_buffer,
                             header_idx, 0) #TODO
            else:
                struct.pack_into("<Q", PredictionResponse.header_buffer,
                             header_idx, output_type_to_dtype(self.output_type).itemsize)
            header_idx += BYTES_PER_LONG

        return PredictionResponse.header_buffer[:header_length], header_length


class FeedbackRequest():
    def __init__(self, msg_id, content):
        self.msg_id = msg_id
        self.content = content

    def __str__(self):
        return self.content


class FeedbackResponse():
    def __init__(self, msg_id, content):
        self.msg_id = msg_id
        self.content = content

    def send(self, socket):
        socket.send("", flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)
        socket.send(self.msg_id, flags=zmq.SNDMORE)
        socket.send(self.content)


class RPCService:
    def __init__(self, input_type, output_type):
        self.input_type = input_type
        self.output_type = output_type

    def get_input_type(self):
        return self.input_type

    def get_output_type(self):
        return self.output_type

    def get_event_history(self):
        if self.server:
            return self.server.get_event_history()
        else:
            print("Cannot retrieve message history for inactive RPC service!")
            raise

    def start(self, model, port = 7000, host = "0.0.0.0",):
        """
        Args:
            model (object): The loaded model object ready to make predictions.
        """
        self.port = port
        self.host = host
        
        try:
            ip = socket.gethostbyname(self.host)
        except socket.error as e:
            print("Error resolving %s: %s" % (self.host, e))
            sys.exit(1)
        context = zmq.Context()
        self.server = Server(context, ip, self.port)
        self.server.model_input_type = string_to_input_type(self.input_type)
        self.server.model_output_type = string_to_output_type(self.output_type)
        self.server.model = model

        self.server.run()


def load_model():
    with open('metadata.json','r') as fp:
        metadata = jsonpickle.decode(json.load(fp))
    framework = metadata['framework']
    args_info = metadata['args_info']
    args_list = []
    for argName in args_info.keys():
      is_dir = args_info[argName]['is_dir']
      if not is_dir:
        with open(argName,'rb') as fp:
          arg = cloudpickle.load(fp)
      else:
        arg = FRAMEWORKS[framework]['load'](argName)
      args_list.append(arg)
    model_class = cloudpickle.loads(metadata['prediction_logic'])
    input_type = str(metadata['AbstractModelType']['input_type'])
    output_type = str(metadata['AbstractModelType']['output_type'])
    model = model_class(*args_list)
    return model, input_type, output_type

if __name__ == "__main__":
    print("Load Model...")
    model, input_type, output_type = load_model()

    print("Starting PyTorchContainer container..")
    rpc_service = RPCService(input_type, output_type)
    rpc_service.start(model)