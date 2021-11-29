Monitoring
======

Monitoring and alerting is crucial for a distributed system, PERSIA provides integration with [Prometheus] for this purpose.

Services in PERSIA push their metrics to a [PushGateway], the gateway then exposes these metrics to Prometheus.

## Step to enable metrics in PERSIA

1. Enable metrics in configuration

Add the following configurations in [`global_config.yaml`](../configuration/index.md).

`job_name(str)` is a name to distinguish your job from others. It can be, for example, `dlrm_v1.0`.

```yaml
PersiaMetricsConfig:
  enable_metrics: true
  job_name: your_job_name
```

2. Deploy PushGateway

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

You can test the metrics are there by doing:

```bash
curl metrics_gateway:9091/metrics
```

in a service container.

By configuring the GF_PATHS_PROVISIONING environment variable, you can specify the [grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/) directory, to access our preset grafana panels.

3. Collecting metrics

To collect metrics from the gateway, you need a prometheus service to do that for you.

Details of how to setup in various environments can be found in for example [docker_sd_config], [kubernetes_sd_config] or [dockerswarm_sd_config].

## Metrics in PERSIA

1. Accuracy related

|  Key   | Description  |
|  ----  | ----  |
| `index_miss_count`  | miss count of indices when lookup. There may be reasons for the missing of embeddings, e.g. lookup a new index or the index has been evicted. |
| `index_miss_ratio`  | miss ratio of indices for all features when lookup for one batch. |
| `gradient_id_miss_count` | num of not found indices when updating gradient. This will happen when embedding evicted before update gradient only.|
| `estimated_distinct_id` | estimated distinct id for each feature.|
| `batch_unique_indices_rate` | unique indices rate in one batch. |
| `staleness` | staleness of sparse model. The iteration of dense model run one by one, while the embedding lookup happened before concurrently. The staleness describe the delay of embeddings. The value of staleness start with 0, increase one when lookup a batch, decrease one when a batch update its gradients|
| `nan_grad_skipped` | nan gradient count caused by dense part. |


2. Efficiency related

|  Key   | Description  |
|  ----  | ----  |
| `lookup_mixed_batch_time_cost` | lookup embedding time cost on embedding server |
| `num_pending_batches` | num batches already sent to embedding worker but still waiting for NN worker to trigger lookup. The pending batches stored in forward buffer, which capacity is configurable by [`global_config.yaml`](https://github.com/PersiaML/tutorials/blob/docs/monitoring/src/configuring/index.md#embedding_worker_config). Once the buffer full, embedding worker may not accept new batches.|
| `lookup_create_requests_time_cost` | lookup preprocess time cost on embedding worker. Include ID hashing, dividing id accroding feature groups and embedding servers.|
| `lookup_rpc_time_cost` | lookup embedding time cost on embedding worker for a batch, include lookup on embedding server and network transmission. |
| `update_gradient_time_cost` | update gradient time cost on embedding worker for a batch. |
| `summation_time_cost` | lookup postprocess time cost on embedding worker, mainly is embedding summation. |
| `lookup_batched_time_cost` | lookup, preprocess and postprocess time cost on embedding worker. |


[Prometheus]: https://prometheus.io/docs/introduction/overview/
[PushGateway]: https://github.com/prometheus/pushgateway
[docker-compose]: https://docs.docker.com/compose/
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
[docker_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#docker_sd_config
[kubernetes_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#kubernetes_sd_config
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
