# Data Processing Advanced
To adapt most of recommendation scene, the scene that data come from different way, different datatype and shape, PERSIA provide the `PersiaBatch` to resolve this problem.

It can receive the multiple datatype and dynamic shape of numerical data and 

## ID Type Features
ID type feature is the sparse 2d vector that define as the list of list with a feature_name in PERSIA(`Tuple[str, List[List]]`) .Each sample in the id_type_feature can be variable length.

`PersiaBatch` only accept  ID type feature with the `np.uint64` datatype.

**Process the Fix length ID-Type-Feature**

Almost all public recommendation dataset concat multiple id_type_features in one `numpy.array`.For each on of id_type_feature it have one id for each sample.Below code help you understand how to process such kind of dataset and add the id_type_feature into `PersiaBatch`.

```python
import numpy as np

id_type_feature_names = [
    "gender", "user_id", "photo_id"
]

id_type_feature_data = np.array([
    [0, 100001, 200001],
    [1, 100002, 300002],
    [0, 100003, 400002],
    [0, 100005, 410002],
    [1, 100006, 400032],
], dtype=np.uint64)

batch_size = 5
id_type_features = []

for id_type_feature_idx, id_type_feature_name in enumerate(id_type_feature_names):
    id_type_feature = []
    for batch_idx in range(batch_size):
        id_type_feature.append(id_type_feature_data[batch_idx: batch_idx + 1, id_type_feature_idx].reshape(-1))
    id_type_features.append((id_type_feature_name, id_type_feature))
print(id_type_features)
```

**Process the Variable Length ID-Type-Feature**

under construction
## Adding Multiple Non-ID Type Features and Labels
Non-ID type features and Labels
```python
import numpy as np

from persia.optim.data import PersiaBatch

PersiaBatch
```

## Add Meta info
`PersiaBatch` provide the meta field to handle unstructured data, you can use serialize the 
```python
import json
import time 

from persia.optim.data import PersiaBatch

batch_info = {
    "batch_id": 100000000,
    "timestamp": time.time()
}

persia_batch = PersiaBatch(id_type_featuers, meta=json.dumps(batch_info))
```