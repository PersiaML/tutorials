# Data Processing Advanced

## Adding more Contiguous Data

*datatypes `PyPersiaBatchData` currently support*
- add_dense_f32 => np.float32 
- add_dense_f64 => np.float64
- add_dense_i32 =>  np.int32
- add_dense_i64 => np.int64


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
