# Customize a PERSIA Job

## TODO: workflow diagram


<!-- 1. 数据定义
2. how to adapt existing dataset -->
## Training Data

Training data in PersiaML consists of three parts, contiguous data (dense), categorical data (sparse) and label data (target). When training with Persia, first format the original training data into the corresponding Persia data format, and then add them to `persia.prelude.PyPersiaBatchData`.

### Contiguous Data
We define the *Contiguous Data* as *Dense Data* in our library. Mixed datatypes are supported. One can add multiple 2D *Dense Data* of different datatypes to `PyPersiaBatchData` by invoking the corresponding methods. Note that the shape of all 2D Dense data should be equal. 

### Categorical Data
We define the *Categorical Data* as *Sparse Data* in our library. It is important to add the name to each *Sparse Data* for later embedding lookup. A *Categorical Data* is composed of a batch of 1d tensors of variable length.

Every categorical data you wanner added should be define in `embedding_config.yml`.Both `middleware-server` and `embedding-server` will load the `embedding_config.yml` file to apply the categorical data configuration.

In below code, we define three categorical data.For each categorical data the requirement fields only category name and the embedding dimension.

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
### Label Data
We use the *Target Data* to represent *Label*. There only accpet the *Target Data* which `ndim` equal to 2 .

### Customize Persia Batch Data

*code example*
```python
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

batch_data.add_dense(np.ones((batch_size, dim), dtype=np.float32))

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
batch_data.add_sparse(batch_categorical_data)
batch_data.add_target(np.ones((1024, 2), dtype=np.float32))
```

more advanced features: ...

## Model Definition

### Define DNN model
For DNN model definition, you can design any model structure as you wanted.The only restriction is to set the DNN model forward function signature as below form.

```python
from typing import List

import torch

class DNN(nn.Module):
    def forward(self, dense: torch.Tensor, sparse: List[torch.Tensor]):
        ...
```

### Modify Sparse Optimizer
Here provide many sparse optimizer in `persia.sparse.optim` module.You can choose the suitable optimizer to adapt your requirement.

### Customize PersiaML Training Context 
Finally step is create the training context to acquire dataloder and sparse embedding process

```python
from torch import nn
from torch.optim import SGD

from persia.ctx import TrainCtx
from persia.data import StreamDataset, Dataloader
from persia.env import get_local_rank
from persia.sparse.optim import Adagrad

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
sparse_optimizer = Adagrad(lr=1e-3)

loss_fn = nn.BCELoss()

with TrainCtx(
    model=model,
    sparse_optimizer=sparse_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
    mixed_precision=mixed_precision
) as ctx:

    train_data_loader = Dataloader(dataset)
    for (batch_idx, data) in enumerate(loader):
        output, target = ctx.forward(data)
        loss= loss_fn(output, target)
        scaled_loss = ctx.backward(loss)
        logger.info(f"current idx: {batch_idx} loss: {loss}")

```

more advanced features: ..

## Middleware

## Parameter Sever

more advanced features: ..
## Evaluation

more advanced features: ..

## Deployment
more advanced features: ..

XXXX

more advanced features: ...


