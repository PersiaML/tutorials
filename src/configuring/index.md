Configuring
======

There are configures for PersiaML servers(middleware or embedding server) defined in a file usually named as `global_config.yaml`. This file is one of the args for PersiaML servers binary file.

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
  storage: Ceph
  enable_incremental_update: false
  incremental_buffer_size: 5000000
  incremental_channel_capacity: 1000
middleware_config:
  forward_buffer_size: 1000
```

The following is a detailed description of each configuration.

## common_config

### metrics_config

* `enable_metrics(bool, default=false)`: Whether to enable metrics.
* `push_interval_sec(int ,default=10)`: The interval of pushing metrics to the promethus pushgateway server.
* `job_name(str, default=persia_defalut_job_name)`: A name to distinguish your job from others.


### intent

The intent of PresiaML can be `Train`, `Eval` or `Infer`

There are additional configures when intent is `Infer`, here is an example.

```yaml
common_config:
  intent: Infer
    servers:
      - emb_server_1:8000
      - emb_server_2:8000
    initial_sparse_checkpoint: /your/sparse/model/dir
```

* `servers(list of str, required)`: `ip:port` list of embedding servers.
* `initial_sparse_checkpoint(str, required)`: Embedding server will load this ckpt when start.


## shard_server_config

* `capacity(int, default=1_000_000_000)`: The capacity of each embedding server. Once the num of indices of an embedding server exceed the capacity, it will evict embeddings according to [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) policies. 
* `num_hashmap_internal_shards(int, default=100)`: The num of internal shard of an embedding server. Embeddings are saved in a HashMap which contains shards. Each of shards holds a lock, can reduce lock contention.
* `full_amount_manager_buffer_size(int, default=1000)`: The buffer size of full amount manager. A full amount manager is used to manage all of the embeddings in a embedding server.
* `num_persistence_workers(int, default=4)`: The concurrency of embedding dumping, loading and incremental update.
* `num_signs_per_file(int, default=1_000_000)`, Number of embeddings to be saved in each file in the checkpoint directory.
* `storage(str, default=Ceph)`: Storage type of embedding. Can be "Ceph" or "Hdfs".
* `enable_incremental_update(bool, default=false)`: Whether to enable incremental update.
* `incremental_buffer_size(int, default=1_000_000)`: Buffer size for incremental update. This is the number of embeddings in each incremental update file.
* `incremental_dir(str, default=/workspace/incremental_dir/)`: The directory for incremental update files to be dumped or loaded.

## middleware_configs

* `forward_buffer_size(int, default=1000)`: Buffer size for prefoard batch data from data loader.