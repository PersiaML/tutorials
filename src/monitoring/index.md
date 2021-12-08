Monitoring
======

Monitoring and alerting is crucial for a distributed system. PERSIA provides integration with [Prometheus] for this purpose.

Services in PERSIA push their metrics to a [PushGateway], and the gateway then exposes these metrics to Prometheus.

## Step to Enable Metrics in PERSIA

### Enable Metrics in Configuration

Add the following configurations in [`global_config.yaml`](../configuration/index.md).

`job_name(str)` is a name to distinguish your job from others. It can be, for example, `dlrm_v1.0`.

```yaml
PersiaMetricsConfig:
  enable_metrics: true
  job_name: your_job_name
```

### Deploy PushGateway

See [official documentation](https://github.com/prometheus/pushgateway) for details. Here is an example for deploying gateway by [docker-compose].

The default push address on PERSIA services is `metrics_gateway:9091`, which can be overridden by the environment variable `PERSIA_METRICS_GATEWAY_ADDR`.

```yaml
version: "3.3"
services:
    data_loader:
        ...

    nn_workler:
        ...

    embedding_worker:
        ...

    server:
        ...

    persia_nats_service:
        ...

    metrics_gateway:
        image: prom/pushgateway:latest
        deploy:
            replicas: 1

    prometheus:
        image: prom/prometheus:latest
        deploy:
            replicas: 1
        command: "--config.file=/workspace/config/prometheus.yml"

    grafana:
        image: grafana/grafana:latest
        ports:
            - "3000:3000/tcp"
        deploy:
            replicas: 1
        environment:
            GF_PATHS_PROVISIONING: /workspace/grafana/provisioning/

```

You can check what metrics are there by running

```bash
curl metrics_gateway:9091/metrics
```

in a service container.

By configuring the `GF_PATHS_PROVISIONING` environment variable, you can specify the [grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/) directory to access our preset grafana panels.

### Collecting Metrics

To collect metrics from the gateway, you need a prometheus service to do that for you.

Details of how to setup a prometheus service in various environments can be found in prometheus documentation, for example [docker_sd_config], [kubernetes_sd_config] or [dockerswarm_sd_config].

## Metrics in PERSIA

### Accuracy related

|  Key   | Description  |
|  ----  | ----  |
| `index_miss_count`  | count of missing indices when lookup. There may be various reasons for the missing of embeddings, e.g. lookup a new index or the index has been evicted. |
| `index_miss_ratio`  | ratio of missing indices for all features when lookup for one batch. |
| `gradient_id_miss_count` | number of missing indices when updating gradient. This will only happen when embedding was evicted before gradient update.|
| `estimated_distinct_id` | estimated number of distinct indices for each feature.|
| `batch_unique_indices_rate` | unique index rate in one batch. |
| `staleness` | staleness of sparse model. The iteration of dense model run one by one, while the embedding lookup happened before concurrently. The staleness describe the delay of embeddings. The value of staleness start with 0, increase one when lookup a batch, decrease one when a batch update its gradients|
| `nan_grad_skipped` | nan gradient count caused by dense part. |


### Efficiency related

|  Key   | Description  |
|  ----  | ----  |
| `lookup_hashmap_time_cost_sec` | time cost of embedding lookup **on embedding server**, mainly spent on looking up from hash table. |
| `num_pending_batches` | number of batches that are already sent to embedding worker but still waiting for NN worker to trigger lookup. The pending batches are stored in forward buffer, whose capacity is configurable by [`global_config.yaml`](https://github.com/PersiaML/tutorials/blob/docs/monitoring/src/configuring/index.md#embedding_worker_config). Once the buffer is full, the embedding worker may not accept new batches.|
| `lookup_preprocess_time_cost_sec` | time cost of preprocess for embedding lookup **on embedding worker**. Include ID hashing, dividing id accroding feature groups and embedding servers.|
| `lookup_rpc_time_cost_sec` | time cost of embedding lookup **on embedding worker** for a batch, include lookup on embedding server (`lookup_hashmap_time_cost_sec`) and network transmission. |
| `lookup_postprocess_time_cost_sec` | lookup postprocess time cost **on embedding worker**, mainly embedding summation. |
| `lookup_total_time_cost_sec` | total time cost of lookup, preprocess and postprocess **on embedding worker**. `lookup_total_time_cost_sec = lookup_preprocess_time_cost_sec + lookup_rpc_time_cost_sec + lookup_postprocess_time_cost_sec` |
| `update_gradient_time_cost_sec` | update gradient time cost **on embedding worker** for a batch. |


[Prometheus]: https://prometheus.io/docs/introduction/overview/
[PushGateway]: https://github.com/prometheus/pushgateway
[docker-compose]: https://docs.docker.com/compose/
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
[docker_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#docker_sd_config
[kubernetes_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#kubernetes_sd_config
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
