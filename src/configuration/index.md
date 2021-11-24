Configuration
======

In order to achieve the best performance on various training and inference jobs, PERSIA servers provide a handful of configuration options via two config files, a global configuration file usually named as `global_config.yaml`, and an embedding configuration file usually named as `embedding_config.yaml`. The global configuration allows one to define job type and general behaviors of servers, whereas embedding configuration provides definition of embedding details for individual sparse features.



## Global Configuration

Global configuration specifies the configuration of the current PERSIA job. The path to the global configuration file should be parsed as argument `--global-config` when launching embedding PS or embedding worker.

Here is an example for `global_config.yaml`.

```yaml
common_config:
  metrics_config:
    enable_metrics: true
    push_interval_sec: 10
  job_type: Train
  checkpointing_config:
    num_workers: 8
embedding_parameter_server_config:
  capacity: 1000000
  num_hashmap_internal_shards: 1
  enable_incremental_update: false
  incremental_buffer_size: 5000000
  incremental_channel_capacity: 1000
embedding_worker_config:
  forward_buffer_size: 1000
```

Depending on the scope, `global_config` was divided into three major sections, namely `common_config`, `embedding_parameter_server_config` and `embedding_worker_config`. `common_config` configures the job type (`job_type`) and metrics server. `embedding_parameter_server_config` configures the embedding parameter server, and `embedding_worker_config` provides configurations for the embedding worker. The following is a detailed description of each configuration.

### common_config

#### checkpointing_config

* `num_workers(int, default=4)`: The concurrency of embedding dumping, loading and incremental update.

#### job_type

The job_type of PresiaML can be either `Train` or `Infer`.

When `job_type` is `Infer`, additional configurations including `servers` and `initial_sparse_checkpoint` have to be provided. Here is an example:

```yaml
common_config:
  job_type: Infer
    servers:
      - emb_server_1:8000
      - emb_server_2:8000
    initial_sparse_checkpoint: /your/sparse/model/dir
```

* `servers(list of str, required)`: list of embedding servers each in the `ip:port` format.
* `initial_sparse_checkpoint(str, required)`: Embedding server will load this ckpt when start.


#### metrics_config
`metrics_config` defines a set of configuration options for monitoring. See [Monitoring](../monitoring/index.md) for details.


* `enable_metrics(bool, default=false)`: Whether to enable metrics.
* `push_interval_sec(int ,default=10)`: The interval of pushing metrics to the promethus pushgateway server.
* `job_name(str, default=persia_defalut_job_name)`: A name to distinguish your job from others.


### embedding_parameter_server_config
`embedding_parameter_server_config` specifies the configuration for the embedding parameter server.
* `capacity(int, default=1,000,000,000)`: The capacity of each embedding server. Once the number of indices of an embedding server exceeds the capacity, it will evict embeddings according to [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) policies.
* `num_hashmap_internal_shards(int, default=100)`: The number of internal shard of an embedding server. Embeddings are saved in a HashMap which contains multiple shards (sub-hashmaps). Since the CRUD operations need to acquire the lock of a hashmap, acquiring the lock of the sub-hashmap instead of the whole hashmap will be more conducive to concurrency between CRUD operations.
* `full_amount_manager_buffer_size(int, default=1000)`: The buffer size of full amount manager. In order to achieve better performance, the embedding server does not traverse the hashmap directly during full dump. Instead, Embedding is submitted asynchronously through full amount manager.
* `enable_incremental_update(bool, default=false)`: Whether to enable incremental update.
* `incremental_buffer_size(int, default=1,000,000)`: Buffer size for incremental update. Embeddings will be inserted into this buffer after each gradient update, and will only be dumped when the buffer is full. Only valid when `enable_incremental_update=true`.
* `incremental_dir(str, default=/workspace/incremental_dir/)`: The directory for incremental update files to be dumped or loaded.

### embedding_worker_config

* `forward_buffer_size(int, default=1000)`: Buffer size for prefoard batch data from data loader.

## Embedding Config

In addition to `global_config`, detailed settings related to sparse feature embeddings are provided in a separate embedding configuration file usually named `embedding_config.yaml`. The path to the embedding config file should be parsed as argument `--embedding-config` when running PERSIA servers.

Here is an example for `embedding_config.yaml`.

```yaml
feature_index_prefix_bit: 8
slot_config:
  workclass:
    dim: 8
    embedding_summation: true
  education:
    dim: 8
    embedding_summation: true
  marital_status:
    dim: 8
    embedding_summation: true
  occupation:
    dim: 8
    embedding_summation: true
  relationship:
    dim: 8
    embedding_summation: true
  race:
    dim: 8
    embedding_summation: true
  gender:
    dim: 8
    embedding_summation: true
  native_country:
    dim: 8
    embedding_summation: true
feature_groups:
  group1:
    - workclass
    - education
    - race
  group2:
    - marital_status
    - occupation

```

The following is a detailed description of each configuration. `required` means there are no default values.

 * `feature_index_prefix_bit(int, default=8)`: Number of bits occupied by each feature group. To avoid hash collisions between different features, the first `n(n=feature_index_prefix_bit)` bits of an index(u64) are taken as the feature bits, and the last `64-n` bits are taken as the index bits. The original id will be processed before inserted into the hash table, following `ID = original_ID % 0~2^(64-n) + index_prefix << (64-n)`. Slots in the same feature group share the same `index_prefix`, which is automatically generated by PERSIA according to the `feature_groups`.

 * `slots_config(map, required)`: `slots_config` contains all the definitions of Embedding. The key of the map is the feature name, and the value of the map is a struct named `SlotConfig`. The following is a detailed description of configuration in a `SlotConfig`.
    * `dim(int, required)`: dim of embedding.
    * `sample_fixed_size(int, default=10)`: raw embedding placeholder size to fill 3d tensor -> (bs, sample_fix_sized, dim).
    * `embedding_summation(bool, default=true)`: whether to reduce(summation) embedding before feeding to dense net.
    * `sqrt_scaling(bool, default=false)`: whether to numerical scaling embedding values.
    * `hash_stack_config(struct, default=None)`: a method to represent a large number of sparse features with a small amount of Embedding vector. It means mapping the original ID to `0~E (E=embedding_size)` through `n (n=hash_stack_rounds)` different hash functions, such as `ID_1, ID_2... ID_n`. Each such ID corresponds to an embedding vector, then performs reduce(summation) operation among these embedding vectors, as input to the dense net of the original ID.
       * `hash_stack_rounds(int, default=0)`: Embedding hash rounds.
       * `embedding_size(int, default=0)`: Embedding hash space of each rounds.

* `feature_groups(map, default={})`: Feature group division. Refer to the description of `feature_index_prefix_bit`. Feature in one feature group will share the same index prefix.
