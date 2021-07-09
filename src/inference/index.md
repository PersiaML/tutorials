Inference
======

persia is able to be deployed with the similar way as its training phase.

In this section, we will show you how to deploy a model with [torchserve][].

## 1. Prepare your handler for torchserve

You can write a [custom handler][] which derive from torchserve BaseHandler, to lookup embedding from embedding server in preprocess procedure.

An implment of a custom handler is:

```
class PersiaHandler(BaseHandler, ABC):
    def initialize(self, context):
        super().initialize(context)
        middleware_services = [os.environ["MIDDLEWARE_SERVICE"]]
        self.persia_backend = PersiaEmbeddingClientPyClass(0, middleware_services)
        self.persia_context = TrainCtx()

    def preprocess(self, data):

        batch = data[0].get('batch')
        batch = bytes(batch)
        # this function copy tensor to device, and lookup from embedding server, return PythonTrainBatch class
        batch = self.persia_backend.batch_to_device_dirct(batch, 0)

        # this function construct torch tensor which is similar with training phase
        model_input, _ = self.persia_context.prepare_features(batch, is_training=False)

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
```

## 2. Prepare your model

You can dump you dense model by torch, like this:
```
jit_model = torch.jit.script(model)
jit_model.save('/your/model/dir/deepctr_v0.pth')
```

and then, use [torch-model-archiver][] to convert this model to mar package which is need by torchserve.

```
torch-model-archiver --model-name deepctr --version 1.0 --serialized-file /your/model/dir/deepctr_v0.pth --handler /your/model/dir/persia_handler.py
```

You can dump you sparse model by calling persia backend, like this:
```
ctx.dump('/your/ceph_dir/or/hdfs_dir/')
```

## 3. Deploy torchserve and perisa servers

You can start torchserve by command:
```
torchserve --start --ncs --model-store /workspace/serve/model/ --models deepctr.mar
```


## 4. Launch request to torchserve by grpc client

You can request torchserve by grpc client.

after generate proto code [follow this], you can use grpc like this:
```
    persia_backend = PersiaEmbeddingClientPyClass(0, [])

    batch_size = 128
    feature_dim = 32
    denses = [np.random.rand(batch_size, feature_dim).astype(np.float32)]
    sparse = []
    for sparse_idx in range(3):
        sparse.append((
            f'feature_{sparse_idx}',
            [np.random.randint(10000000000, size=feature_dim).astype(np.uint64) for _ in range(batch_size)]
        ))

    batch_data = PyPersiaBatchData()
    batch_data.add_dense(denses)
    batch_data.add_sparse(sparse)

    model_input = persia_backend.batch_to_bytes(batch_data)
    infer(get_inference_stub(), 'deepctr', model_input)
```

## 5. Incremental update of persia embedding server

persia supports incremental updates, you can modify the `global_config.yaml` file to enbale incremental config.

for training, a incremental update packet will be dumped to storage when gradient updated.

while for infer, embedding server keep scanning a directory to find if there is a new packet to load.

* `enable_incremental_update`: whether to enbale incremental update
* `incremental_buffer_size`: buffer size of incremental update. Indices will be insert into a hashset when update gradient, when the size of hashset is execced buffer size, dump an incremental update packet to storage.
* `incremental_dir`: the path of incremental update packet dumped or loaded.
* `storage`: dump incremental update packet to ceph or hdfs.

## 6. update dense model to torch serve

You can update dense model version with [management api][], like this:
```
def register(stub, model_name):
    params = {
        'url': f"/workspace/model/{model_name}.mar",
        'initial_workers': 1,
        'synchronous': True,
        'model_name': model_name
    }
    try:
        response = stub.RegisterModel(management_pb2.RegisterModelRequest(**params))
        print(f"Model {model_name} registered successfully")
    except grpc.RpcError as e:
        print(f"Failed to register model {model_name}.")
        print(str(e.details()))
        exit(1)
```

generaly, you can dump dense model every several step of training, like this:
```
model_path = f"/workspace/model/deepctr_step_{batch_idx}.pth"
jit_model = torch.jit.script(model)
jit_model.save(model_path)
subprocess.check_call(
    f"torch-model-archiver \
        --model-name deepctr \
        --version {batch_idx} \
        --serialized-file {model_path} \
        --handler {handler_path} \
        --export-path {mar_path} -f",
    shell=True
)
```


[torchserve]: https://github.com/pytorch/serve
[custom handler]: https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handler-with-class-level-entry-point
[torch-model-archiver]:https://github.com/pytorch/serve/blob/master/model-archiver/README.md
[follow this]: https://github.com/pytorch/serve#using-grpc-apis-through-python-client
[management api]: https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api