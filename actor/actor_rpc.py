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
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
else:
    from subprocess import Popen, PIPE
    from io import BytesIO as StringIO
import docker
import tempfile
import tarfile
import zlib
import pickle

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
    int_strs = ["i", "ints", "int", "integer", "integers", "builtins.int"]
    float_strs = ["f", "floats", "float", "builtins.float"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str", "builtins.str"]

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
    int_strs = ["i", "ints", "int", "integer", "integers", "builtins.int"]
    float_strs = ["f", "floats", "float", "builtins.float"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str", "builtins.str"]

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
                    socket.send(self.inputs[idx])
                else:
                    socket.send(self._format_input(self.inputs[idx]))
            else:
                if self.input_type == INPUT_TYPE_STRINGS:
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
        elif self.input_type == INPUT_TYPE_ABSTRACT:
            p = pickle.dumps(inp)
            z = zlib.compress(p)
            return z

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
            elif self.input_type == INPUT_TYPE_ABSTRACT:
                struct.pack_into("<Q", PredictionRequest.header_buffer,
                             header_idx, 0) #TODO
            else:
                struct.pack_into("<Q", PredictionRequest.header_buffer,
                             header_idx, input_type_to_dtype(self.input_type).itemsize)
            header_idx += BYTES_PER_LONG

        return PredictionRequest.header_buffer[:header_length], header_length

class ContainerMetadata():
    def __init__(self, model_input_type, model_output_type, container_rpc_version):
        self.model_input_type = int(model_input_type)
        self.model_output_type = int(model_output_type)
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

    def build_model(self, docker_client, 
                    name,
                    version,
                    model_path,
                    prediction_file,
                    port=7000,
                    base_image="alice97/serve-base",
                    container_registry=None,
                    pkgs_to_install=None):
        version = str(version)
        run_cmd = ''
        if pkgs_to_install:
            run_as_lst = 'RUN pip install'.split(' ')
            run_cmd = ' '.join(run_as_lst + pkgs_to_install)
        with tempfile.NamedTemporaryFile(mode="w+b", suffix="tar") as context_file:
            # Create build context tarfile
            with tarfile.TarFile(fileobj=context_file, mode="w") as context_tar:
                from shutil import copyfile
                # copyfile("__init__.py", os.path.join(model_path, "__init__.py"))
                copyfile("container_rpc.py", os.path.join(model_path, "container_rpc.py"))
                # copyfile("pytorch_container.py", os.path.join(model_path, "pytorch_container.py"))
                context_tar.add(model_path)
                try:
                    df_contents = StringIO(
                        str.encode(
                            "FROM {container_name}\n{run_command}\n COPY {model_path} /model\n WORKDIR /model\n EXPOSE {port}\n CMD [ \"python3\", \"./{prediction_file}\" ]".
                            format(
                                container_name=base_image,
                                model_path=model_path,
                                prediction_file=prediction_file,
                                run_command=run_cmd,
                                port=port)))
                    df_tarinfo = tarfile.TarInfo('Dockerfile')
                    df_contents.seek(0, os.SEEK_END)
                    df_tarinfo.size = df_contents.tell()
                    df_contents.seek(0)
                    context_tar.addfile(df_tarinfo, df_contents)
                except TypeError:
                    df_contents = StringIO(
                        "FROM {container_name}\n{run_command}\n COPY {model_path} /model\n WORKDIR /model\n EXPOSE {port}\n CMD [ \"python3\", \"./{prediction_file}\" ]".
                        format(
                            container_name=base_image,
                            model_path=model_path,
                            prediction_file=prediction_file,
                            run_command=run_cmd,
                            port=port))
                    df_tarinfo = tarfile.TarInfo('Dockerfile')
                    df_contents.seek(0, os.SEEK_END)
                    df_tarinfo.size = df_contents.tell()
                    df_contents.seek(0)
                    context_tar.addfile(df_tarinfo, df_contents)
            # Exit Tarfile context manager to finish the tar file
            # Seek back to beginning of file for reading
            context_file.seek(0)
            image = "{name}:{version}".format(
                name=name, version=version)
            image_result, build_logs = docker_client.images.build(fileobj=context_file, custom_context=True, tag=image)

        return image

    def run_container(self, docker_client, image, detach=True, cmd=None, name=None, ports=None,
                    labels=None, environment=None, log_config=None, volumes=None,
                    user=None):
        self.container = docker_client.containers.run(
            image,
            detach=True,
            command=cmd,
            name=name,
            ports=ports,
            labels=labels,
            environment=environment,
            volumes=volumes,
            user=user,
            log_config=log_config)

    def stop_container(self):
        self.container.stop()

    def remove_container(self):
        self.container.remove()

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
            # print("send: %f s, recv: %f s" %
                              # (send_time, recv_time))
                              
            return outputs, num_outputs
        else:
            print("Wrong message type %d, should be container content msg" % msg_type)
            raise

    def recv_outputs_content(self, num_outputs, output_sizes):
        outputs = []
        for i in range(num_outputs):
            if self.container_meta.model_output_type == OUTPUT_TYPE_STRINGS:
                output = self.socket.recv_string()
            elif self.container_meta.model_output_type == OUTPUT_TYPE_INTS:
                output_bytes = self.socket.recv()
                output = struct.unpack("<I", output_bytes)[0]
            elif self.container_meta.model_output_type == OUTPUT_TYPE_DOUBLES:
                output_bytes = self.socket.recv()
                output = struct.unpack("<d", output_bytes)[0]
            elif self.container_meta.model_output_type == OUTPUT_TYPE_FLOATS:
                output_bytes = self.socket.recv()
                output = struct.unpack("<f", output_bytes)[0]
            elif self.container_meta.model_output_type == OUTPUT_TYPE_BYTES:
                output_bytes = self.socket.recv()
                output = struct.unpack("<B", output_bytes)[0]
            elif self.container_meta.model_output_type == OUTPUT_TYPE_ABSTRACT:
                output_bytes = self.socket.recv()
                p_output = zlib.decompress(output_bytes)
                output = pickle.loads(p_output)
            outputs.append(output)

        return outputs
    
    def connect_to_container(self):
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
            model_input_type = self.socket.recv_string()
            model_output_type = self.socket.recv_string()
            container_rpc_version_bytes = self.socket.recv()
            container_rpc_version = struct.unpack("<I", container_rpc_version_bytes)[0]

            self.container_meta = ContainerMetadata(model_input_type, model_output_type, container_rpc_version)
            self.connected = True
            print("Actor Model is connected")
        else:
            print("Wrong message type %d, should be new container msg" % msg_type)
            raise

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

    def connect_to_container(self):
        self.actor.connect_to_container() 

    def send_prediction_request(self, input_type, inputs):
        # return request result
        return self.actor.send_prediction_request(input_type, inputs)

    def build_model(self, docker_client, name, version, model_path, prediction_file,
                    port=7000, base_image="alice97/serve-base", container_registry=None,
                    pkgs_to_install=None):
        return self.actor.build_model(
                docker_client, 
                name,
                version,
                model_path,
                prediction_file,
                port=7000,
                base_image="alice97/serve-base",
                container_registry=None,
                pkgs_to_install=None)

    def run_container(self, docker_client, image, name=None, detach=True, cmd=None, ports=None,
                    labels=None, environment=None, log_config=None, volumes=None,
                    user=None):
        self.actor.run_container(
                docker_client,
                image,
                detach=True,
                cmd=cmd,
                name=name,
                ports=ports,
                labels=labels,
                environment=environment,
                volumes=volumes,
                user=user,
                log_config=log_config)

    def stop_container(self):
        self.actor.stop_container()

    def remove_container(self):
        self.actor.remove_container()
