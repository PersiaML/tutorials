from persia.ctx import InferCtx
from persia.prelude import forward_directly_from_bytes

from ts.torch_handler.base_handler import BaseHandler

from abc import ABC
import torch
import os

class PersiaHandler(BaseHandler, ABC):
    def initialize(self, context):
        super().initialize(context)
        self.persia_context = InferCtx()

    def preprocess(self, data):

        with self.persia_context as ctx:
            batch = data[0].get('batch')
            batch = bytes(batch)
            batch = forward_directly_from_bytes(batch, 0)

            model_input, _ = ctx.prepare_features(batch, is_training=False)

        return model_input

    def inference(self, data, *args, **kwargs):
        denses, sparses = data
        with torch.no_grad():
            results = self.model(denses, sparses, *args, **kwargs)
        return results

    def postprocess(self, data):
        data = torch.reshape(data, (-1,))
        data = data.tolist()
        return [data]