# Customization

Before we introduce how to customize a PERSIA training task, let's take a look at how PERSIA's different components work together.
The training process can be summarized by the following figure:

<center>
<img src="img/persia_workflow.png" width="600">
</center>

There are a few files you can customize in PERSIA:

TODO: keep order consistent with the following sections
1. Data preprocessing configuration file: `data_loader.py`
2. Model definition configuration file: `train.py`
3. Embedding configuration file: `embedding_config.yaml`
4. Embedding PS configuration file: `global_config.yaml`
5. Launcher configuration:
    1. If you are using k8s, `train.yaml`
    2. if you are using docker compose
    3. ...

## Training Data

**PersiaBatch consists of three parts, contiguous data, categorical data and label data.**

<img src="./img/persia_batch_description.svg" width="80%" style="margin:auto">

TODO: make all naming consistent with paper

### Non-ID Type Features
Non-ID Type Features is a tensor or vector that contains numerical data.For example the click_num, income, price, labor time or some numerical type data could be concat as the contiguous data and become a part of training data.

In PERSIA batch data, contiguous data is alias as dense data.It is described as a 2d tensor with float datatype. 


### ID Type Features
ID Type Features is a sparse tensor that contains variable length of discrete value. Such user_id, photo_id, client_id. There should at least exists categorical name and dimension to describe a categorical data. PERSIA parameter server will project the discrete value in categorical data to a vector and the dimension of vector is equal to the value you describe before.It is simple to add one categorical data in PERSIA, modify the embedding config file and add the categorical name and its dimension.Both `embedding-worker` and `embedding-parameter-server` will load the embedding config file to apply the categorical data configuration.

In below code, we define three categorical data.For each categorical data the requirement fields are category name and the embedding dimension.

_[yaml_config](https://github.com/PersiaML/PERSIA/blob/main/examples/src/getting_started/config/embedding_config.yml)_
```yml
slot_configs:
  id:
    dim: 8
    embedding_summation: true # optional field
  age:
    dim: 8
  gender:
    dim: 8
```

_more advanced features: embedding_config_chapter.md_

### Labels
Label data in PERSIA batch is a 2d `float32` tensor that support add the classification target and regression target.

### Customize PERSIA Batch Data

_[data_loader](https://github.com/PersiaML/PERSIA/blob/main/examples/src/getting_started/data_loader.py)_
```python (data_preprocessing.py)
import numpy as np

from persia.prelude import PyPersiaBatchData

batch_data = PyPersiaBatchData()

# categorical name should be the same with the categorical name which 
# already defined in embedding_config.yml.
categorical_names = [
    "id",
    "age",
    "gender"
]

batch_size = 1024
dim = 256

batch_data.add_non_id_type_feature(np.ones((batch_size, dim), dtype=np.float32))

categorical_data_num = 3
max_categorical_len = 65536

batch_categorical_data = []
for categorical_idx in range(categorical_data_num):
    batch_categorical_data_item = []
    for batch_idx in range(batch_size):
        cnt_categorical_len = np.random.randint(0, max_categorical_len)
        sample_data = np.random.one((cnt_categorical_len), dtype=np.uint64)
        batch_categorical_data_item.append(sample_data)
    batch_categorical_data.append((categorical_names[categorical_idx], batch_sparse_data))

# add mock sparse data into PyPersiaBatchData 
batch_data.add_id_type_features(batch_categorical_data)
batch_data.add_label(np.ones((1024, 2), dtype=np.float32))
```

more advanced features: ...

- advanced 1 (under construction)

## Model Definition

### Define DNN model
For DNN model definition, you can design any model structure as you wanted.The only restriction is to set the DNN model forward function signature as below form.

```python
from typing import List

import torch

class DNN(nn.Module):
    def forward(self, non_id_tensors: List[torch.Tensor], embedding_tensors: List[torch.Tensor]):
        ...
```

### Modify  Embedding Optimizer
Here provide many sparse optimizers in `persia.embedding.optim` module.You can choose the suitable optimizer to adapt your requirement.

### Customize PERSIA Training Context 
Final step is create the training context to acquire dataloder and sparse embedding process

_[train](https://github.com/PersiaML/PERSIA/blob/main/examples/src/getting_started/train.py)_
```python
from torch import nn
from torch.optim import SGD

from persia.ctx import TrainCtx
from persia.data import StreamDataset, Dataloader
from persia.env import get_local_rank
from persia.embedding.optim import Adagrad

prefetch_size = 10
dataset = StreamDataset(prefetch_size)

local_rank = get_local_rank()

use_cuda = True
if use_cuda:
    device_id = get_local_rank()
    torch.cuda.set_device(device_id)
    model.cuda(device_id)
    mixed_precision = True
else:
    mixed_precision = False
    device_id = None

# DNN parameters optimizer
dense_optimizer = SGD(model.parameters(), lr=0.1)
# Embedding parameters optimizer
embedding_optimizer = Adagrad(lr=1e-3)

loss_fn = nn.BCELoss()

with TrainCtx(
    model=model,
    embedding_optimizer=embedding_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
    mixed_precision=mixed_precision
) as ctx:

    train_data_loader = Dataloader(dataset)
    for (batch_idx, data) in enumerate(loader):
        output, labels = ctx.forward(data)
        label = labels[0]
        loss= loss_fn(output, target)
        scaled_loss = ctx.backward(loss)
        logger.info(f"current idx: {batch_idx} loss: {loss}")

```

_more advanced features: [TrainCtx]("../training-context")_

## Configuring Embedding Worker

An embedding worker runs asynchronous updating algorithm for getting the embedding parameters from the embedding parameter server; aggregating embedding vectors (potentially) and putting embedding gradients back to embedding parameter server. You can learn the details of the system design through 4.2 section in our [paper](https://arxiv.org/abs/2111.05897). Generally, you only need to adjust the number of instances and resources according to your workload.

## Configuring Embedding Parameter Server

An embedding parameter server manages the storage and update of the embedding parameters according to [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) policies. So you need to configure capacity of the LRU cache in the configuration file according to your workload and available memory size. In addition, the capacity means the max number of embedding vectors, not the number of parameters.

more advanced features: See [Configuration](../configuration/index.md)


## Launcher configuration

### docker compose launcher

### k8s launcher

The PERSIA Operator is a Kubernetes [custom resource definitions](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/). You can define your distributed persia task by an operator file.

Here is an example for an operator file

```yaml
# train.yaml

apiVersion: persia.com/v1
kind: PersiaJob
metadata:
  name: adult-income  # persia job name, need to be globally unique
  namespace: default  # k8s namespace to deploy to this job
spec:
  # the following path are the path inside the container
  globalConfigPath: /home/PersiaML/examples/src/getting_started/config/global_config.yml
  embeddingConfigPath: /home/PersiaML/examples/src/getting_started/config/embedding_config.yml
  nnWorkerPyEntryPath: /home/PersiaML/examples/src/getting_started/train.py
  dataLoaderPyEntryPath: /home/PersiaML/examples/src/getting_started/data_loader.py
  env:
    - name: PERSIA_NATS_IP
      value: nats://persia-nats-service:4222

  embeddingParameterServer:
    replicas: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "4"

  embeddingWorker:
    replicas: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "4"

  nnWorker:
    replicas: 1
    nprocPerNode: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "12"
        nvidia.com/gpu: "1"
    env:
      - name: CUBLAS_WORKSPACE_CONFIG
        value: :4096:8

  dataloader:
    replicas: 1
    resources:
      limits:
        memory: "8Gi"
        cpu: "1"

---
# a nats operator
apiVersion: "nats.io/v1alpha2"
kind: "NatsCluster"
metadata:
  name: "persia-nats-service"
spec:
  size: 1
  natsConfig:
    maxPayload: 52428800
  resources:
    limits:
      memory: "8Gi"
      cpu: "2" 
```
more advanced features: See [kubernetes-integration](../kubernetes-integration/index.md)


## Deploy Trained Model for inference

See [Inference](../inference/index.md).


