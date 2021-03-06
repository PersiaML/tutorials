# Training Context
<!--
PERSIA training context is a configurable context that help you to set the corresponding embedding training configuration. PERSIA support both gpu nn_worker and cpu nn_worker.Different type of nn_worker may not support the same feature.Usually the gpu nn_worker will do will perform than the cpu nn_worker. -->

A PERSIA training context manages training environments on NN workers.

<!-- toc -->
## PERSIA Training Context Complete Example

Here is a complete example for the usage of PERSIA training context.

```python
import torch

from persia.ctx import TrainCtx
from persia.embedding.optim import Adagrad
from persia.env import get_rank, get_world_size
from persia.data import Dataloder, PersiaDataset, StreamingDataset

from model import DNN


if __name__ == "__main__":
    model = DNN()
    rank, world_size = get_rank(), get_world_size()

    device_id = 0
    torch.cuda.set_device(device_id)
    model.cuda(device_id)

    dense_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    embedding_optimizer = Adagrad(lr=1e-2)
    loss_fn = torch.nn.BCELoss(reduction="mean")

    with TrainCtx(
        model=model,
        embedding_optimizer=embedding_optimizer,
        dense_optimizer=dense_optimizer,
        device_id=device_id,
    ) as ctx:
        train_dataloader = Dataloder(StreamingDataset(10))

        for (batch_idx, data) in enumerate(train_dataloader):
            (output, labels) = ctx.forward(data)
            loss = loss_fn(output, label[0])
            scaled_loss = ctx.backward(loss)
```

In the following section, we will introduce several configuration options when creating a PERSIA training context.

<!-- We will introduce several configurations that may help you to configure your task while using PERSIA in training job. -->

## EmbeddingConfig

[EmbeddingConfig](https://persiaml.pages.dev/main/autoapi/persia/embedding/#persia.embedding.EmbeddingConfig) defines embedding hyperparameters.

- `emb_initialization`: The default initialization of PERSIA embedding is `Uniform` distribution. Value is a tuple of the lower and upper bound of embedding uniform initialization.
- `admit_probability`: The probability (0<=, <=1) of admitting a new embedding.
- `weight_bound`: Restrict each element value of an embedding in `[-weight_bound, weight_bound]`.

```python
from persia.embedding import EmbeddingConfig
from persia.ctx import TrainCtx

embedding_config = EmbeddingConfig(
    emb_initialization=(-1, 1),
    admit_probability=0.8,
    weight_bound=1
)

TrainCtx(
    embedding_config=embedding_config,
    ...
)

```

## Mixed Precision Training

The `mixed_precision` feature in PERSIA training is only supported on gpu NN workers because it depends on [pytorch amp](https://pytorch.org/docs/stable/amp.html).

<!-- And it only improves the speed of the dense model training and reduce the corresponding device memory cost. It won increase or reduce the data for the embedding. -->

## Distributed Option

[Distributed Option](https://persiaml.pages.dev/main/autoapi/persia/distributed/#module-persia.distributed) defines the implementation of data parallelism among PERSIA NN workers.
<!-- Distributed training in PERSIA is easy to configuration. We already integrated two distributed option for you to use. -->

- [DDP](https://pytorch.org/docs/stable/distributed.html) (by default): Native pytorch distributed training data parallelism implementation.
- [Bagua](https://tutorials.baguasys.com/introduction): A deep learning training acceleration framework for PyTorch.

**Configuring DDPOption**

In `DDPOption`, you can configure the `backend` and `init_method`.
<!-- The default `init_method` is `"tcp"` who needs `master_address` and `master_port`. -->

```python
from persia.distributed import DDPOption

backend = "nccl"
# backend = "gloo"

init_method = "tcp"
# init_method = "file"

DDPOption(backend="nccl", init_method=init_method)
```

**Configuring BaguaDistributedOption**

<!-- Bagua support multiple data parallelism implementation that may help you speedup the training speed. Review the [doc](https://tutorials.baguasys.com/algorithms/) to select the best one for you. -->

There are several data parallelism implementations in Bagua, see [Bagua Documentation](https://tutorials.baguasys.com/algorithms/) to learn more about Bagua.

```python
from persia.ctx import TrainCtx
from persia.distributed import BaguaDistributedOption

algorithm = "gradient_allreduce"
# algorithm = low_precision_decentralized
# algorithm = bytegrad
# algorithm = async

bagua_args = {}
bagua_option = BaguaDistributedOption(
    algorithm,
    **bagua_args
)

TrainCtx(
    distributed_option=bagua_option
)
```
