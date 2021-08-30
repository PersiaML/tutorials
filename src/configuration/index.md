Configuration
======

## Global Configuration

In order to achieve best performance on various training and inference jobs, PersiaML servers provide a handful of configuration options via a yaml config file usually named as `global_config.yaml`. The path to the config file should be parsed as an argument `--global-config` when running PersiaML servers.

Here is an example for `global_config.yaml`.

```yaml
common_config:
  metrics_config:
    enable_metrics: false
    push_interval_sec: 10
  intent: Train
shard_server_config:
  capacity: 100000000
  num_hashmap_internal_shards: 128
  full_amount_manager_buffer_size: 1000
  num_persistence_workers: 4
  num_signs_per_file: 5000000
  enable_incremental_update: false
  incremental_buffer_size: 5000000
  incremental_channel_capacity: 1000
middleware_config:
  forward_buffer_size: 1000
```

Depending on the scope, `global_config` was divided into three major sections, namely `common_config`, `shard_server_config` and `middleware_config`. `common_config` provides global settings for PersiaML training and inference jobs. `shard_server_config` provides configurations for the PersiaML embedding server, and `middleware_config` provides configurations for the PersiaML middleware. The following is a detailed description of each configuration.

### common_config

#### metrics_config

* `enable_metrics(bool, default=false)`: Whether to enable metrics.
* `push_interval_sec(int ,default=10)`: The interval of pushing metrics to the promethus pushgateway server.
* `job_name(str, default=persia_defalut_job_name)`: A name to distinguish your job from others.


#### intent

The intent of PresiaML, can be either `Train` or `Infer`.

When intent is `Infer`, additional configurations including `servers` and `initial_sparse_checkpoint` have to be provided. Here is an example:

```yaml
common_config:
  intent: Infer
    servers:
      - emb_server_1:8000
      - emb_server_2:8000
    initial_sparse_checkpoint: /your/sparse/model/dir
```

* `servers(list of str, required)`: list of embedding servers each in the `ip:port` format.
* `initial_sparse_checkpoint(str, required)`: Embedding server will load this ckpt when start.


### shard_server_config

* `capacity(int, default=1_000_000_000)`: The capacity of each embedding server. Once the number of indices of an embedding server exceeds the capacity, it will evict embeddings according to [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) policies.
* `num_hashmap_internal_shards(int, default=100)`: The number of internal shard of an embedding server. Embeddings are saved in a HashMap which contains multiple shards(sub-hashmap). Since the CRUD operations needs to acquire the lock of a hashmap, acquiring the lock of the sub-hashmap instead of the whole hashmap will be more conducive to concurrency between CRUD operations.
* `full_amount_manager_buffer_size(int, default=1000)`: The buffer size of full amount manager. A full amount manager is used to manage all of the embeddings in a embedding server.
* `num_persistence_workers(int, default=4)`: The concurrency of embedding dumping, loading and incremental update.
* `num_signs_per_file(int, default=1_000_000)`, Number of embeddings to be saved in each file in the checkpoint directory.
* `enable_incremental_update(bool, default=false)`: Whether to enable incremental update.
* `incremental_buffer_size(int, default=1_000_000)`: Buffer size for incremental update. This is the number of embeddings in each incremental update file.
* `incremental_dir(str, default=/workspace/incremental_dir/)`: The directory for incremental update files to be dumped or loaded.

### middleware_configs

* `forward_buffer_size(int, default=1000)`: Buffer size for prefoard batch data from data loader.

## Embedding Config

In addition to `global_config`, detailed settings related to embedding is provided in a separate embedding configuration file usually named as `embedding_config.yaml`.The path to the embedding config file should be parsed as argument `--embedding-config` when running PersiaML servers.

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

The following is a detailed description of each configuration.

 * `feature_index_prefix_bit(int, default=8)`: Number of bits occupied by feature group. In order to avoid hash collisions between different features, the first n(n=feature_index_prefix_bit) bits of an index(u64) is taken as the feature bits, and the last (64-n) bits is taken as the index bits. The original id will be processed before inserted into the hash table, following `ID = original_ID % 0~2^(64-n) + index_prefix << (64-n)`. Slots in a same feature group share a same `index_prefix`, which is automatically generated by Persia according to the `feature_groups`.

 * `slot_config(map, required)`: slot_config contains all the definitions of Embedding. The key of map is feature name, the value of map is a struct named as `SlotConfig`. The following is a detailed description of configuration in a `SlotConfig`.
    * `dim(int, required)`: dim of embedding.
    * `sample_fixed_size(int, default=10)`: raw embedding placeholder size to fill 3d tensor -> (bs, sample_fix_sized, dim).
    * `embedding_summation(bool, default=true)`: wether to reduce(summation) embedding before feed to dense net.
    * `sqrt_scaling(bool, default=false)`: wether to numerical scaling embedding values.
    * `hash_stack_config(struct)`: A method to represent a large number of sparse features with a small amount of Embedding vector. It means mapping the original ID to `0~E (E=embedding_size)` through `n (n=hash_stack_rounds)` different hash functions, such as `ID_1, ID_2... ID_n`. Each such ID corresponds to an embedding vector, then perform reduce(summation) operation among these embedding vectors, as input to dense net of the the original ID.
       * `hash_stack_rounds(int, default=0)`: Embedding hash rounds.
       * `embedding_size(int, default=0)`: Embedding hash space of each rounds.

* `feature_groups(map, default={})`: Feature group division. Refer to the description of `feature_index_prefix_bit`, feature in one feature group will share a same index prefix.
