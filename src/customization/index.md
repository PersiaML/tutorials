
# Customization


Before we introduce how to customize a PERSIA training task, let's take a look at how PERSIA's different components work together.
The training process can be summarized by the following figure:

<center>
<img src="img/persia_workflow.png" width="1000">
</center>

There are a few files you can customize in PERSIA:

1. Data preprocessing configuration file: `data_loader.py`
2. Model definition configuration file: `train.py`
3. Embedding configuration file: `embedding_config.yaml`
4. Embedding PS configuration file: `global_config.yaml`
5. Launcher configuration:
    1. If you are using K8S, `k8s.train.yaml`
    2. If you are using docker compose, `docker-compose.yml` and `.docker.env`
    3. If you are using honcho, `Procfile` and `.honcho.env`

* [Training Data](#training-data)
  * [Add ID Type Features](#add-id-type-feature)
  * [Add Non-ID Type Features](#add-non-id-type-feature)
  * [Add Labels](#add-label)
  * [Send PersiaBatch](#send-persia-batch)
* [Model Definition](#model-definition)
  * [Define DNN model](#define-dnn-model)
  * [Define Embedding Optimizer](#modify-embedding-optimizer)
  * [Customize PERSIA Training Context](#customize-persia-training-context)
* [Configuring Embedding Worker](#configuring-embedding-worker)
* [Configuring Embedding Parameter Server](#configuring-embedding-parameter-server)
* [Launcher configuration](#launcher-configuration)
  * [K8s launcher](#k8s-launcher)
  * [Honcho Launcher](#honcho-launcher)
  * [Docker Compose Launcher](#docker-compose-launcher)
* [Build PERSIA Runtime Image Locally](#biuld-persia-runtime-image-locally)
* [Deploy Trained Model for inference](#deploy-trained-model-for-inference)

## Training Data

A `PersiaBatch` is consists of three parts: contiguous data, categorical data and label data.

### Add ID Type Feature
`IDTypeFeature` contains variable length of categorical data. `IDTypeFeature` store the `List[np.array]` data which is a list of sparse matrix. Note that it only accepts `np.uint64` elements.

For example, you can add `user_id` and `photo_id` data into a `IDTypeFeatureSparse`.

```python
import numpy as np

from persia.embedding.data import IDTypeFeatureSparse

id_type_features = []

# add user_id data
user_id_batch_data = [
  np.array([1000, 1001], dtype=np.uint64),
  np.array([1000,], dtype=np.uint64),
  np.array([], dtype=np.uint64), # allow empty sample
  np.array([1000, 1001, 1024], dtype=np.uint64),
  np.array([1000,] * 200, dtype=np.uint64),
]

id_type_features.append(IDTypeFeatureSparse(user_id_batch_data, "user_id"))

# add photo_id data
photo_id_batch_data = [
  np.array([2000, 1001], dtype=np.uint64),
  np.array([3000,], dtype=np.uint64),
  np.array([5001], dtype=np.uint64), 
  np.array([4000, 1001, 1024], dtype=np.uint64),
  np.array([4096,] * 200, dtype=np.uint64),
]

id_type_features.append(IDTypeFeatureSparse(photo_id_batch_data, "photo_id"))
```

After adding `IDTypeFeature`, you have to add corresponding `id_type_feature` config in `embedding_config.yml`. See [configuration](../configuration/index.md) for more details about how to config the `id_type_feature`, such as `dim`, `sqrt_scaling`, etc.

_more advanced [id_type_feature processing](../data-processing/index.md#id-type-feature)_


### Add Non-ID Type Feature

You are also able to add multiple `NonIDTypeFeature` into `PersiaBatch` with various datatype. Concatting multiple `non_id_type_features` with same datatype into one `np.array` can avoid memory fragmentation and reduce the time of type check. For example, you are able to add `height`, `income` or even `image` data.

```python
import numpy as np

from persia.embedding.data import NonIDTypeFeature

non_id_type_features = [] 

# height data
height_batch_data = np.array([
  [170],
  [183],
  [175],
  [163],
  [177],
], dtype=np.float32)

non_id_type_features.append(NonIDTypeFeature(height_batch_data, "height"))

# income data
income_batch_data = np.array([
  [37000],
  [7000],
  [2000],
  [6660],
  [3000],
], dtype=np.float32)

non_id_type_features.append(NonIDTypeFeature(income_batch_data, name="income"))

# add income_with_height
income_with_height = np.hstack([height_batch_data, income_batch_data])
non_id_type_features.append(NonIDTypeFeature(income_with_height, name="income_with_height"))

# add image data
image_data = np.ones((5, 224, 224, 3), dtype=np.uint8)
non_id_type_features.append(NonIDTypeFeature(image_data, name="LSVR_image"))
```

### Add Label

Adding a label is as same as the `NonIDTypeFeature`, you can add different datatype label data such as `bool`, `float32`, etc.

```python
import numpy as np

from persia.embedding.data import Label

labels = []
# add ctr label data
ctr_batch_data = np.array([
  0,
  1,
  0,
  1,
  1
], dtype=np.bool)

labels.append(Label(ctr_batch_data, "ctr"))

# add income label data
income_batch_data = np.array([
  [37000],
  [7000],
  [2000],
  [6660],
  [3000],
], dtype=np.float32)
labels.append(Label(income_batch_data, "income"))

# add ctr with income, but will cost extra bytes to cast ctr_batch_data from bool to float32
ctr_with_income = np.hstack([ctr_batch_data, income_batch_data])
labels.append(Label(ctr_with_name, "ctr_with_income"))
```

### Send PersiaBatch

Use `persia.ctx.DataCtx` to send the data to `nn_worker` and `embedding_worker` after the `PersiaBatch` created:

```python
import numpy as np

from persia.ctx import DataCtx
from persia.embedding.data import PersiaBatch, IDTypeFeatureSparse

id_type_features = [
  IDTypeFeatureSparse("empty_sample", np.array([[]] * 5, dtype=np.uint64))
]

persia_batch = PersiaBatch(
  id_type_features=id_type_features,
  requires_grad=False
)

with DataCtx() as ctx:
  ctx.send_data(persia_batch)
```

## Model Definition

### Define DNN model

You can define any DNN model structure as you want, only note that the forward function signature of the model should be same with follow.

```python
from typing import List

import torch

class DNN(nn.Module):
    def forward(
      self, 
      non_id_type_feature_tensors: List[torch.Tensor],
      id_type_feature_embedding_tensors: List[torch.Tensor]
    ):
        ...
```

### Modify Embedding Optimizer

There are several kinds of embedding optimizers in PERSIA. For more details, see [api doc](https://persiaml.pages.dev/main/autoapi/persia/embedding/optim/).

```python
from persia.embedding.optim import SGD, Adagrad, Adam
from persia.ctx import TrainCtx

lr = 1e-3
sgd_embedding_optimizer = SGD(lr)
adagrad_embedding_optimizer = Adagrad(lr)
adam_embedding_optimizer = Adam(lr)
```

### Customize PERSIA Training Context

After above, a PERSIA training context should be created to acquire dataloder and manage sparse embedding.

```python
# train.py

from torch import nn
from torch.optim import SGD

from persia.ctx import TrainCtx
from persia.data import StreamDataset, Dataloader
from persia.env import get_local_rank
from persia.embedding.optim import Adagrad

prefetch_size = 10
dataset = StreamDataset(prefetch_size)

local_rank = get_local_rank()

device_id = get_local_rank()
torch.cuda.set_device(device_id)
model.cuda(device_id)

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
) as ctx:

    train_data_loader = Dataloader(dataset)
    for (batch_idx, data) in enumerate(loader):
        output, labels = ctx.forward(data)
        label = labels[0]
        loss= loss_fn(output, target)
        scaled_loss = ctx.backward(loss)
        logger.info(f"current idx: {batch_idx} loss: {loss}")

```

_more advanced features: [TrainCtx](../training-context/index.md)_

## Configuring Embedding Worker

An embedding worker runs asynchronous updating algorithm for getting the embedding parameters from the embedding parameter server; aggregating embedding vectors (potentially) and putting embedding gradients back to embedding parameter server. You can learn the details of the system design through 4.2 section in our [paper](https://arxiv.org/abs/2111.05897). Generally, you only need to adjust the number of instances and resources according to your workload. See [K8S launcher](#k8s-launcher).

## Configuring Embedding Parameter Server

An embedding parameter server manages the storage and update of the embedding parameters according to [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) policies. So you need to configure capacity of the LRU cache in the configuration file according to your workload and available memory size. In addition, the capacity means the max number of embedding vectors, not the number of parameters. Here is an example.

```yaml
# global_config.yaml

common_config:
  job_type: Train
embedding_parameter_server_config:
  capacity: 1000000
```

_more advanced features: See [Configuration](../configuration/index.md#global-configuration)_


## Launcher configuration

There are launchers to help you launch a PERSIA training task.
<!-- 
We provide the different launcher to satisfy your requirements. The below launchers can run the PERSIA task in different handy level. -->

- K8S launcher: Kubernetes launcher is easy to deploy large scale training.
- docker-compose launcher: Docker compose is the other way like `K8S` but is more lightweight.
- honcho launcher: A Procfile manager that need to build PERSIA in manually(Currently persia can build in linux, macOS, windows10.). It is hard for inexperienced ones to install the requirement. But is friendly for developers to develop and debug.

All of these launchers use environment variables(`PERSIA_GLOBAL_CONFIG`, `PERSIA_EMBEDDING_CONFIG`, `PERSIA_NN_WORKER_ENTRY`, `PERSIA_DATALOADER_ENTRY`) to assign the path of the PERSIA configuration files.


### K8S Launcher

The PERSIA Operator is a Kubernetes [custom resource definitions](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/). You can define your distributed PERSIA training task by an operator file.

Here is an example for an operator file

```yaml
# k8s.train.yaml

apiVersion: persia.com/v1
kind: PersiaJob
metadata:
  name: adult-income  # persia job name, need to be globally unique
  namespace: default  # k8s namespace to deploy to this job
spec:
  # path of PERSIA configuration files.
  persiaEnv:
    PERSIA_GLOBAL_CONFIG: /home/PERSIA/examples/src/adult-income/config/global_config.yml
    PERSIA_EMBEDDING_CONFIG: /home/PERSIA/examples/src/adult-income/config/embedding_config.yml
    PERSIA_NN_WORKER_ENTRY: /home/PERSIA/examples/src/adult-income/train.py
    PERSIA_DATALOADER_ENTRY: /home/PERSIA/examples/src/adult-income/data_loader.py
  env:
    - name: PERSIA_NATS_IP
      value: nats://persia-nats-service:4222  # hostname need to be same with nats operator's name

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
      - name: ENABLE_CUDA
        value: "1"

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
_more advanced features: See [kubernetes-integration](../kubernetes-integration/index.md)_

### Docker Compose Launcher

We have prepared the `.docker.env` and `docker-compose.yml` files for you to launch PERSIA training task with docker compose. Following are steps to update the PERSIA task.

**Configuring ENV**

Required fields in `.docker.env`

* `DOCKER_COMPOSE`: should be set to `1`.
* `REPLICA_SIZE`: `replica_size` for PERSIA modules.

Optional fields in `.docker.env`
* `NPROC_PER_NODE`: number of processes per node to specify.
* `ENABLE_CUDA`: use cuda or not.

```env
# .docker.env file
ENABLE_CUDA=1
NPROC_PER_NODE=1
```

**Configuring docker-compose File**

Required fields in `docker-compose.yml`

* `TASK_SLOT_ID`: This fields is required for all service in `docker-compose.yml`. The docker engine will use regex to extract docker `slot_id` into `TASK_SLOT_ID`. PERSIA will read it as `REPLICA_INDEX`.
* `REPLICAS`: This fields is required for all service in `docker-compose.yml`. PERSIA will read it as `REPLICA_SIZE`.

Optional fields in `docker-compose.yml`
* `ENABLE_CUDA`: use cuda or not

```yaml
# docker-compose.yml

version: "3.2"
services:
  persia_nats_service:
    image: nats:latest
    deploy:
      replicas: 1
          
  data_loader:
    env_file:
      - .docker.env
    depends_on:
      - nn_worker
      - embedding_worker
      - persia_nats_service
    image: persiaml/persia-cuda-runtime:latest
    command: persia-launcher data-loader /workspace/data_loader.py
    volumes:
      - type: bind
        source: .
        target: /workspace
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure

  nn_worker:
    env_file:
      - .docker.env
    environment:
      NCCL_SOCKET_IFNAME: eth0
      CUBLAS_WORKSPACE_CONFIG: :4096:8
    image: persiaml/persia-cuda-runtime:latest
    command: persia-launcher nn-worker /workspace/train.py
    volumes:
      - type: bind
        source: .
        target: /workspace
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure

  embedding_worker:
    env_file:
      - .docker.env
    depends_on:
      - server
    image: persiaml/persia-cuda-runtime:latest
    command: persia-launcher embedding-worker --embedding-config $$PERSIA_EMBEDDING_CONFIG --global-config $$PERSIA_GLOBAL_CONFIG
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
    volumes:
      - type: bind
        source: .
        target: /workspace

  server:
    env_file:
      - .docker.env
    image: persiaml/persia-cuda-runtime:latest
    command: persia-launcher embedding-server --embedding-config $$PERSIA_EMBEDDING_CONFIG --global-config $$PERSIA_GLOBAL_CONFIG
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
    volumes:
      - type: bind
        source: .
        target: /workspace
```

### Honcho Launcher

Honcho launcher is convenient for debug. You can simulate distributed environment by editing the `Procfile` and `.honcho.env` file.

**Configuring Env**

There are fields when launching the PERSIA task with Honcho:

Required fields in `.honcho.env`

* `PERSIA_NATS_IP`: set for nats-server ip address.

Optional fields in `.honcho.env`

* `ENABLE_CUDA`: use cuda or not.

```env
# .honcho.env

HONCHO=1 # required by PERSIA to determine the rank

REPLICA_INDEX=0 # required by PERSIA to determine the replica_index for data_loader
REPLICA_SIZE=1 # required by PERSIA to determine the replica_size for data_loader

ENABLE_CUDA=0 # enable cuda or not
NPROC_PER_NODE=1 # number of processes per node to specify.
ENABLE_CUDA=0 # enable cuda or not

# default nats_server ip address
PERSIA_NATS_IP=nats://0.0.0.0:4222 
```
**Configuring Procfile**

You can add multiple replica of PERSIA modules as you want in `Procfile`.
For example, by adding `embedding_server{replica_num}` and `embedding_worker{replica_num}`, you can launch three `embedding-parameter-server` and two `embedding-worker` subprocesses.

```bash
# Procfile

# data_loader
data_loader: REPLICA_SIZE=1 REPLICA_INDEX=0 python3 data_loader.py && cat 
# nn_worker
nn_worker: persia-launcher nn-worker train.py --nproc-per-node=NPROC_PER_NODE --node-rank=0 --nnodes=1
# launch three subprocesses of embedding parameter server
embedding_server0: persia-launcher embedding-parameter-server --embedding-config config/embedding_config.yml --global-config config/global_config.yml --replica-index 0 --replica-size 3 --port 10000
embedding_server1: persia-launcher embedding-parameter-server --embedding-config config/embedding_config.yml --global-config config/global_config.yml --replica-index 1 --replica-size 3 --port 10001
embedding_server2: persia-launcher embedding-parameter-server --embedding-config config/embedding_config.yml --global-config config/global_config.yml --replica-index 2 --replica-size 3 --port 10002
# launch two subprocess of embedding worker server
embedding_worker1: persia-launcher embedding-worker --embedding-config config/embedding_config.yml --global-config config/global_config.yml --replica-index 0 --replica-size 2 --port 9000
embedding_worker2: persia-launcher embedding-worker --embedding-config config/embedding_config.yml --global-config config/global_config.yml --replica-index 1 --replica-size 2 --port 9001
# nats_server
nats_server: nats-server 

```
## Build PERSIA Runtime Image Locally

Persia runtime image can be built from local. You can use your customized docker image to launch a PERSIA training task by both kubernetes and docker-compose.

Use following instructions to build persia-runtime-image:

```bash
git clone https://github.com/PersiaML/PERSIA.git
# docker image name: persiaml/persia-cuda-runtime:dev
cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
```

## Deploy Trained Model for inference

See [Inference](../inference/index.md).


