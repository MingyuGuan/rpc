from __future__ import print_function
import container_rpc
import sys


class NoopContainer(container_rpc.ModelContainerBase):
    def __init__(self, prediction="1.0"):
        self.prediction = prediction

    def _predict(self, inputs):
        return [self.prediction] * len(inputs)

    def predict_ints(self, inputs):
        return self._predict(inputs)

    def predict_floats(self, inputs):
        return self._predict(inputs)

    def predict_doubles(self, inputs):
        return self._predict(inputs)

    def predict_bytes(self, inputs):
        return self._predict(inputs)

    def predict_strings(self, inputs):
        return self._predict(inputs)


if __name__ == "__main__":
    print("Starting No-Op container")
    rpc_service = container_rpc.RPCService()

    model = NoopContainer()
    model_name = "noop_container"
    model_version = 1
    model_path = "/fake_path"

    sys.stdout.flush()
    sys.stderr.flush()
    rpc_service.start(model, model_name, model_version, model_path)
