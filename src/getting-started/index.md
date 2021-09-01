# Getting Started

According to DLRM example to make you construct machine learning application based on Persia swiftly

## Setup

1. Download the Kaggle Display Advertising Challenge dataset for DLRM example
    ```bash
    cd examples/DLRM/data  
    curl -L -o data.tar.gz https://ndownloader.figshare.com/files/10082655
    # wget https://ndownloader.figshare.com/files/10082655 -o data.tar.gz
    tar -zxvf data.tar.gz && rm -rf data.tar.gz
    ```
2. Initialize the persia-core git submodule `git submodule init --udpate`
3. Prepare for runtime docker image
    ```bash
    make build_all 
    ```

## Process training data
Kaggle Display Advertising Challenge dataset contain two parts of data that calls `dense` and `sparse`.  `Dense data` is represent by a 1D vector that come from a set of statistics data or extract by `DNN model` extract from image, video, audio or etc. Dense data should have same dimension for each sample. `Sparse data` also represents by a 1D vector but the dimension could be various for each sample. `Sparse data` is a list of category data, for example age, gender, user_id, book_id or etc.PerisaML framework converted `sparse data` to fixed size `dense tensor(1d or 2d)` by the process called `embedding lookup`. 

### Data preprocess
Process Kaggle Display Advertising Challenge raw dataset to `numpy.ndarray` by `numpy`

```python
import numpy as np

"""
sample of train.txt
0       1       1       5       0       1382    4       15      2       181     1       2               2       68fd1e64      80e26c9b        fb936136        7b4723c4        25c83c98        7e0ccccf        de7995b8        1f89b562     a73ee510 a8cd5504        b2cb9c98        37c9c164        2824a5f6        1adce6ef        8ba8b39a        891b62e7     e5ba7672 f54016b9        21ddcdc9        b1252a9d        07b5194c                3a171ecb        c5c50484        e8b83407        9727dd16

"""
source_data = "train.txt"
batch_size = 5
batch_data = []

with open(source_data, "r") as file:
    for line in file:
        splitter = denseline.split("\t")
        target = np.int32(line[0])
        dense_sample = np.array(splitter[1:14], dtype=np.float32)
        sparse_sample = np.array(
            list(map(lambda x: int(x, 16), line[14:])), dtype=np.uint64
        ) 
        batch_data.append((dense_sample, sample_sample, target))
        if len(batch_data) == batch_size:
            # process the below
            ...
```

### Persia data structure to store training data
PersiaML provide specific data structure for sparse training scence. The structure can add multiple dense, sparse and target data. 
```python
from persia.prelude import PyPersiaBatchData

persia_batch_data = PyPersiaBatchData()
batch_dense_data, batch_all_sparse_data, batch_target_data = zip(*batch_data)
```

### Add dense data
`PyPersiaBatchData` provide the `add_dense` function to add a list of 2d float32 numpy array. It also provide some specific functions to add dense data in various datatype such as `add_dense_f32`, `add_dense_i32`, `add_dense_f64`, `add_dense_i64`. 
```python
persia_batch_data = PyPersiaBatchData()
batch_dense_data = np.stack(batch_dense_data)

persia_batch_data.add_dense_f32(batch_dense_data)
```

### Add sparse data
Add multiple categories sparse data into `PyPersiaBatchData`, each category data should have the same batch size and a unique namespace to share feature space. 
```python
"""
    In Kaggle Display Advertising Challenge dataset, every categories only lookup one sparse id in each sample. 
    sample below:

    batch_size = 5
    sparse_feature1 = [
        np.array([0]),
        np.array([1]),
        np.array([2]),
        np.array([3])
        np.array([5])
    ]

    sparse_feature2 = [
        np.array([2]),
        np.array([3]),
        np.array([4]),
        np.array([2])
        np.array([6])
    ]
"""
batch_sparse_category_data = []
for (idx, batch_sparse_data) in enumerate(batch_all_sparse_data):
    category_name = f"sparse_feature{idx}"
    sparse_array = []
    for i in range(batch_size):
        sparse_array.append(batch_sparse_data[i:i+1])
    batch_sparse_category_data.append((category_name, sparse_array))
persia_batch_data.add_sparse(batch_sparse_category_data)
```

### Add target data
Target data (ground truth) is define as a 2d float32 numpy array, user can add multiple target data into `PyPersiaBatchData` for multi-task Learning.
```python
batch_target_data = np.array(batch_target_data, dtype=np.float32)
batch_target_data = np.stack([batch_target_data[i] for i in range(batch_size)]) 
persia_batch_data.add_target(batch_target_data) # can invoke multiple times for multi task training
```

### Data transfer 
Init the `persia_backend` to transfer the `persia batch data` to the persia-middleware and persia-trainer. 
```python
from persia.ctx import DataCtx
with DataCtx():
    ctx.send_data(persia_batch_data) # register sparse data and transfer remain part to trainer service
```

_review DLRM datacompose codebase at examples/DLRM/data_compose.py_

## Define model
Construct the DLRM model after finished the data definition. The Model forward entry receive dense tensors and sparse tensors.
```python
from typing import List

import torch
class DLRM(nn.Module):
    def __init__(self, ln, sigmoid_layer):
        ...

    def forward(self, dense: List[torch.Tensor], sparse: List[torch.Tensor]):
        ...

model = DLRM()
```
_review DLRM model codebase at examples/DLRM/model.py_

## Define optimizer
Define optimizer in sparse training is as simple as normal torch training. Dense optimizer is define for update the `DNN model`.Sparse optimizer is define for update the sparse sparse embedding.`perisa.sparse.optim` provide common optimizer for different training scences.
```python
from torch.optim import SGD
from persia.sparse.optim import Adagrad

# DENSE parameters optimizer
dense_optimizer = SGD(model.parameters(), lr=0.1)
# Sparse embedding optimizer
sparse_optimizer = Adagrad(lr=1e-3)
```

## Create training context
Finally step is create the training context to acquire dataloder and sparse embedding process
```python
from persia.ctx import TrainCtx

scaler = torch.cuda.amp.GradScaler() # for half training

with TrainCtx(
    model=model,
    sparse_optimizer=sparse_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
) as ctx:
    for (batch_idx, data) in enumerate(ctx.data_loader()):
        (dense, sparse), target = ctx.prepare_features(data)
        output, aux_loss = model((dense, sparse))
        bce_loss = loss_fn(output, target)
        loss = aux_loss + bce_loss
        scaled_loss = ctx.backward(loss)
        logger.info(f"current idx: {batch_idx} loss: {loss}")

```
_review DLRM model codebase at examples/DLRM/train.py_

## Start training

```bash
cd examples/DLRM/ && make run 
# run make stop to remove docker stack job
# make stop
```
