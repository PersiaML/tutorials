# Training Context
PERSIA training context is a configurable context that help you to set the corresponding embedding training configuration.

There provide serval configuration that you may use during a distributed training.

## EmbeddingConfig

The default initialization of PERSIA embedding is `Uniform` distribution.You can configure the `upper` and `lower` value of `Uniform` distribution.

## Distributed Option
PERSIA support both gpu nn_worker and cpu nn_worker.You can select the suitable scene by configure the `TrainCtx`.

```python
from persia.distributed import DDPOption

DDPOption()
```