# Training Context
PERSIA training context is a configurable context that help you to set the corresponding embedding training configuration.PERSIA support both gpu nn_worker and cpu nn_worker.Different type of nn_worker may not support the same feature.Usually the gpu nn_worker will do will perform than the cpu nn_worker.

We will introduce several configurations that may help you to configure your task while using PERSIA in training job.

## EmbeddingConfig

- emb_initialization: The default initialization of PERSIA embedding is `Uniform` distribution.Lower and upper bound of embedding uniform initialization.
- admit_probability: The probability (0<=, <=1) of admitting a new embedding.
- weight_bound: Restrict each element value of an embedding in [-weight_bound, weight_bound].

```python
from persia.embedding import EmbeddingConfig
from persia.ctx import TrainCtx

embedding_config = EmbeddingConfig(
    emb_initialization=(-1, 1),
    admit_probability=0.8,
    weight_bound=1
)

TrainCtx(
    embedding_config=embedding_config
)
```
## Mixed Precision Training

Notice that the mixed_precision feature in PERSIA training is only support on gpu nn_worker because of the feature is supported by `torch.amp`.And it only improve the speed of the dense model training and reduce the corresponding device memory cost.It won increase or reduce the data for the embedding.

## Distributed Option
Distributed training in PERSIA is easy to configuration.We already integrated two distributed option for you to use.

- [DDP](https://pytorch.org/docs/stable/distributed.html): Native pytorch distributed training dataparallel.Default distributed setting, both support cpu nn_worker and gpu nn_worker.
- [Bagua](https://tutorials.baguasys.com/introduction): Bagua is a deep learning training acceleration framework for PyTorch.**Only support on gpu nn_worker.**

**Configure DDPOption**
```python
from persia.distributed import DDPOption

backend = "nccl"
# backend = "gloo"

init_method = "tcp"
# init_method = "file"

master_addr = "localhost"
master_port = 2307

DDPOption(backend="nccl", init_method=init_method, master_addr=master_addr, master_port=master_port)
o
```

**Configure BaguaDistributedOption**

Bagua support multiple algorithms that may help you speedup the training speed.Review the [doc](https://tutorials.baguasys.com/algorithms/) to select the best one for you.

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