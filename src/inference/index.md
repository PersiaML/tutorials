Inference
======

To do inference for trained models, we need to deploy PersiaML middleware service, PersiaML embedding service, and [TorchServe] services.

When a TorchServe inference server receives requests, it first looks up embeddings on PersiaML services, and then does the forward pass for the DNN part.

[TorchServe] is a flexible framework for serving PyTorch models. In this page, we will introduce how to deploy a PerisaML model with it.

In the following sections, we first introduce how to create a custom handler for TorchServe to query embeddings during inference. Then introduce how to save models during training and load models during inference. Then we introduce how to deploy various services for inference. Finally, we introduce how to query the inference service to get the inference result.

## 1. Create PersiaML handler for TorchServe

With TorchService, customized operations (like preprocess or postprocess) can be done with simple Python scripts, called [custom handler].

There are ways to write custom handler, one of them is [custom-handler-with-class-level-entry-point].

Here is an example to define a custom handler retrieving PersiaML embeddings:

```python
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
        batch = data[0].get('batch')
        batch = bytes(batch)
        batch = forward_directly_from_bytes(batch, 0)

        model_input = self.persia_context.prepare_features(batch)
        return model_input

    def inference(self, data, *args, **kwargs):
        denses, sparses = data
        with torch.no_grad():
            results = self.model(denses, sparses)
        return results

    def postprocess(self, data):
        data = torch.reshape(data, (-1,))
        data = data.tolist()
        return [data]
```

## 2. Save and load PersiaML model

The sparse part and the dense part of a PerisaML model are saved separately.

For the dense part, it is saved directly by PyTorch with [TorchScript]:

```python
jit_model = torch.jit.script(model)
jit_model.save('/your/model/dir/you_model_name.pth')
```

Then, to serve the dense part with TorchServe, use [torch-model-archiver] to package it.

```bash
torch-model-archiver --model-name you_model_name --version 1.0 --serialized-file /your/model/dir/you_model_name.pth --handler /your/model/dir/persia_handler.py
```

Sparse model can be saved and loaded with PerisaML Python API, see [Model Checkpointing](../model-checkpointing/index.md) for details.

## 3. Deploy PerisaML services and TorchServe

TorchServe can be launched with:

```bash
torchserve --start --ncs --model-store /workspace/serve/model/ --models you_model_name.mar
```

There are configurations in [`global_config.yaml`](https://github.com/PersiaML/tutorials/blob/docs/monitoring/src/configuring/index.md#global-config) when deploy embedding servers and middlewware for inference.

```yaml
PersiaInferConfig:
  # list of embedding servers(ip:port)
  servers:
    - emb_server_1:8000
    - emb_server_2:8000
  # embedding server will load this ckpt when start
  initial_sparse_checkpoint: /your/sparse/model/dir
```

## 4. Query inference result with gRPC

There are ways to [get predictions from a model] with TorchServe. One of them is using [gRPC API](https://github.com/pytorch/serve#using-grpc-apis-through-python-client) through a [gRPC client](https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py).

The input data is constructed in the same way as in training, Here is an example:
```python
import grpc
import inference_pb2
import inference_pb2_grpc

def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

def infer(stub, model_name, model_input):
    with open(model_input, 'rb') as f:
        data = f.read()

    input_data = {'data': data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))

    try:
        prediction = response.prediction.decode('utf-8')
        print(prediction)
    except grpc.RpcError as e:
        exit(1)

batch_size = 128
feature_dim = 16
denses = [np.random.rand(batch_size, 13).astype(np.float32)]
sparse = []
for sparse_idx in range(26):
    sparse.append((
        f'feature{sparse_idx + 1}',
        [np.random.randint(1000000, size=feature_dim).astype(np.uint64) for _ in range(batch_size)]
    ))

batch_data = PyPersiaBatchData()
batch_data.add_dense(denses)
batch_data.add_sparse(sparse)

model_input = batch_data.to_bytes()
infer(get_inference_stub(), 'you_model_name', model_input)
```

## 5. Model incremental update

It is crucial to keep the model for inference up to date. For huge sparse models, PersiaML provides incremental updates, so that online prediction services only receives model differences during training to update the online model for inference. This dramatically reduces the model latency between training and inference.

During training, an incremental update file will be dumped periodically. During inference, PersiaML services keep scanning a directory to find if there is a new incremental update file to load.

Relavant configurations in [`global_config.yaml`](https://github.com/PersiaML/tutorials/blob/docs/monitoring/src/configuring/index.md#global-config) are `enable_incremental_update`, `incremental_buffer_size` and `incremental_dir`.


## 6. Manage dense models on TorchServe

To update dense model with sparse model, it can be managed by torchserve through its [management api]. After generating the `.mar` file according to the above steps, its path can be sent to torchserve with [grpc client].



[TorchServe]: https://github.com/pytorch/serve
[custom-handler-with-class-level-entry-point]: https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handler-with-class-level-entry-point
[custom handler]: https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers
[TorchScript]: https://pytorch.org/docs/stable/jit.html
[torch-model-archiver]:https://github.com/pytorch/serve/blob/master/model-archiver/README.md
[get predictions from a model]: https://github.com/pytorch/serve#get-predictions-from-a-model
[management api]: https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api
