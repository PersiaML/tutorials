# Data Processing
PersiaML data format is consist of three part, contiguous data, categorical data and corresponding label data. Parsing the origin training data to Persia data format and then add them into the python class `persia.prelude.PyPersiaBatchData`.

## Contiguous Data
We define the *Contiguous Data* as *Dense data* in our library.User can add multiple 2D *Dense Data* of different datatype into `PyPersiaBatchData` by invoke the corresponding methods.The shape of 2d tensor should be equal.

*datatype current `PyPersiaBatchData` support*
- add_dense_f32 => np.float32 
- add_dense_f64 => np.float64
- add_dense_i32 =>  np.int32
- add_dense_i64 => np.int64

*code example*
```python
import numpy as np

from persia.prelude import PyPersiaBatchData

batch_data = PyPersiaBatchData()

batch_size = 1024
dim = 256

batch_data.add_dense_f32(np.ones((batch_size, dim), dtype=np.float32))
batch_data.add_dense_i32(np.ones((batch_size, dim), dtype=np.int32))
batch_data.add_dense_f64(np.ones((batch_size, dim), dtype=np.float64))
batch_data.add_dense_i64(np.ones((batch_size, dim), dtype=np.int64))
```
## Categorical Data
We define the *Categorical Data* as *Sparse Data* in our library. It is important to add the name to each *Sparse Data* for purpose of later embedding lookup phase.A *Categorical Data* is composed by a batch of variable length of 1d tensor.

*code example*
```python
import numpy as np

from persia.prelude import PyPersiaBatchData

sparse_data_num = 3
batch_size = 1024
max_sparse_len = 65536

# gen mock sparse data
batch_sparse_datas = []
for feature_idx in range(sparse_data_num):
    batch_sparse_data = []
    for batch_idx in range(batch_size):
        cnt_sparse_len = np.random.randint(0, max_sparse_len)
        sparse_data = np.random.one((cnt_sparse_len), dtype=np.uint64)
    batch_sparse_datas.append((batch_sparse_data, f"feature_{feature_idx}"))

# add mock sparse data into PyPersiaBatchData 
batch_data = PyPersiaBatchData()
batch_data.add_sparse(batch_sparse_datas)
```

## Label Data
We define the *Label Data* as the *Target Data*.*Target Data* should be a `2d flaot32 tensor`.

*code example*
```python
import numpy as np

from persia.prelude import PyPersiaBatchData

batch_data = PyPersiaBatchData()
batch_size = 1024
batch_data.add_target(np.ones((1024, 2), dtype=np.float32))
```