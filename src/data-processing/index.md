# Data Processing Advanced
To adapt most recommendation scene, the scene that data come from different way, different datatype and shape, PERSIA provide the `PersiaBatch` to resolve this problem.

It can receive the multiple datatype and dynamic shape of numerical data and 

## ID Type Features
ID type feature is the sparse 2d vector that define as the list of list with a feature_name in PERSIA(`Tuple[str, List[List]]`) .Each sample in the id_type_feature can be variable length.

`PersiaBatch` only accept  ID type feature with the `np.uint64` datatype.

**Process the Fix length ID-Type-Feature**

Almost all public recommendation dataset concat multiple id_type_features in one `numpy.array`.For each on of id_type_feature it have one ID for each sample.Below code help you understand how to process such kind of dataset and add the id_type_feature into `PersiaBatch`.

```python
import numpy as np

from persia.embedding.data import PersiaBatch


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

persia_batch = PersiaBatch(id_type_features)
```

**Process the Variable Length ID-Type-Feature**

Variable length of id_type_feature is not usually show on public recommendation dataset, but it is very important for the people when owns the huge amount of user interactive data especially for some huge Internet company.And this type of feature will be assigned the §max_variable_lengthed§ in most framework. It is hard to increase the id_type_feature length to inifinitly but always keep the training speed dropdown slightly.The id_type_feature can improve the DNN result significant as the §max_variable_length§ increase.Below code help you understand how to process id_type_feature which has the variable length.

```python
import numpy as np

from persia.embedding.data import PersiaBatch


id_type_feature_names = [
    "gender", "user_id", "photo_id"
]

gender_data = [
    [0],
    [1],
    [0],
    [0],
    [1]
]

user_id_data = [
    [100001, 100003, 100005, 100020],
    [100001],
    [100001, 200001, 300001],
    [400001, 100001],
    [100001]
]

photo_id_data = [
    [400032, 400031],
    [400032, 400332, 420032, 400332,],
    [400032],
    [], # support empty id_type_feature but still need to add it to keep batch construction
    [400032, 401032, 400732, 460032, 500032]
]

id_type_feature_data = [
    gender_data, user_id_data, photo_id_data
]

batch_size = 5
id_type_features = []

for id_type_feature_idx, id_type_feature_name in enumerate(id_type_feature_names):
    id_type_feature = []
    for batch_idx in range(batch_size):
        id_type_feature.append(np.array(id_type_feature_data[id_type_feature_idx][batch_idx: batch_idx + 1], dtype=np.uint64).reshape(-1))
    id_type_features.append((id_type_feature_name, id_type_feature))
print(id_type_features)

persia_batch = PersiaBatch(id_type_features)
```

## Adding Multiple Non-ID Type Features and Labels
Non-ID type features and Labels can be variable datatype and shape.The restrictions to them is to check the datatype support or not and the batch_size is same as id_type_feature or not.Below code help you understand adding these two type of data.

TODO(wangyulong): add support datatype
```python
import numpy as np

from persia.optim.data import PersiaBatch

batch_size = 5
id_type_features = [("empty_id_type_feature_with_batch_size", [np.array([], dtype=np.uint64)] * batch_size)]

persia_batch = PersiaBatch(id_type_features)


# add non_id_type_feature
# int8 image_embedding from DNN Extractor
persia_batch.add_non_id_type_feature(np.ones((batch_size, 256), dtype=np.int8))
# general statistics such as average income, height, weight
# you can merge the non_id_type_feature together with same datatype
persia_batch.add_non_id_type_feature(np.eye((batch_size, 3) dtype=np.float32))
# image_pixel_data or RS data with multiple dimension
persia_batch.add_non_id_type_feature(np.ones((batch_size, 3, 224, 224), dtype=np.int8))

# add label 
# multiple label classification label
persia_batch.add_label(np.ones((batch_size, 4), dtype=np.bool))
# regression label
persia_batch.add_label(np.ones((batch_size), dtype=np.float32))
```

## Add Meta info
`PersiaBatch` provide the meta field to handle unstructured data, you can use serialize the 
```python
import json
import time 

from persia.optim.data import PersiaBatch

batch_size = 5
id_type_features = [("empty_id_type_feature_with_batch_size", [np.array([], dtype=np.uint64)] * batch_size)]

batch_info = {
    "batch_id": 100000000,
    "timestamp": time.time()
}
persia_batch = PersiaBatch(id_type_featuers, meta=json.dumps(batch_info))
```