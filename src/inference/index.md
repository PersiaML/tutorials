Inference
======

When deploying online inference services, like training, middlewares and embedding servers also needs to be deployed.

However, unlike training, there is no trainer, instead an inference service. After a inference server receives requests, it first lookup Embedding remotely, and then do forwarding locally.

[TorchServe] is a flexible and easy to use tool for serving PyTorch models, in this section, we will show you how to deploy a PerisaML model with it.

## 1. Prepare handler for torchserve

Customized behavior(e.g. preprocess or postprocess) of TorchServe could be defined by writing a Python script, which called [custom handler]. TorchServe executes this code when it runs. 

There are ways to write custom handler, one of them is [custom-handler-with-class-level-entry-point].

Here is an example:

```
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

## 2. Prepare model

The sparse and dense parts of a PerisaML model are saved separately.

For dense part, it can be saved directly through torch api, like [TorchScript], for example:

```
jit_model = torch.jit.script(model)
jit_model.save('/your/model/dir/you_model_name.pth')
```

Then, use [torch-model-archiver] to package all model artifacts into a single model archive file. This file can then be redistributed and served by anyone using TorchServe.

```
torch-model-archiver --model-name you_model_name --version 1.0 --serialized-file /your/model/dir/you_model_name.pth --handler /your/model/dir/persia_handler.py
```

Sparse model could be saved by PerisaML api, see [Model Checkpointing](../model-checkpointing/index.md).

## 3. Deploy torchserve and Perisa servers

torchserve could be launched by this command:
```
torchserve --start --ncs --model-store /workspace/serve/model/ --models you_model_name.mar
```
There are configures in `PersiaInferConfig` in `global_config.yaml` when deploy embedding servers and middlewware when inferencing.

```
PersiaInferConfig:
  # list of embedding servers(ip:port)
  servers:
    - emb_server_1:8000
    - emb_server_2:8000
  # embedding server will load this ckpt when start
  initial_sparse_checkpoint: /your/sparse/model/dir
```

## 4. Launch request to torchserve by grpc client

There are ways to [get predictions from a model] for torchserve. One of them is using [grpc apis] through a [grpc client].

The data construction process is the same as training, Here is an example:
```
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

## 5. Incremental update of sparse model

The real-time level of the model has an impact on the online accuracy. However, saving a huge embedding model frequently will incur a lot of overhead. Therefore, Persia supports incremental updates, saving the incremental part(Recently updated gradient) of embedding only.

For training, a incremental update packet will be dumped to storage when gradient updated. while for infer, embedding server keep scanning a directory to find if there is a new packet to load.

There are configures for incremental update in `global_config.yaml`

* `enable_incremental_update`: whether to enbale incremental update
* `incremental_buffer_size`: buffer size of incremental update. Indices will be insert into a hashset when update gradient, when the size of hashset is execced buffer size, dump an incremental update packet to storage.
* `incremental_dir`: the path of incremental update packet dumped or loaded.
* `storage`: dump incremental update packet to ceph or hdfs.

## 6. manage dense model to torch serve

A dense model can be managed by torchserve through its [management api]. After generating the `.mar` file according to the above steps, its path can be sent to torchserve with [grpc client].




[torchserve]: https://github.com/pytorch/serve
[custom-handler-with-class-level-entry-point]: https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handler-with-class-level-entry-point
[custom handler]: https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers
[TorchScript]: https://pytorch.org/docs/stable/jit.html
[torch-model-archiver]:https://github.com/pytorch/serve/blob/master/model-archiver/README.md
[grpc client]: https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py
[get predictions from a model]: https://github.com/pytorch/serve#get-predictions-from-a-model
[grpc apis]: https://github.com/pytorch/serve#using-grpc-apis-through-python-client
[management api]: https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api