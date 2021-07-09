Monitoring
======

Monitoring is necessary for a distributed systems, PersiaML use [Prometheus] to monitor status of cluster.

Nodes in PerisaML cluster push their metrics to a [PushGateway], the Pushgateway then exposes these metrics to Prometheus.


## Step to enable metrics in PerisaML

1. Config Metrics of PersiaML 

There is `PersiaMetricsConfig` section in `global_config.yaml`. To enable push metrics, set `enable_metrics(bool)` to `true`.

`job_name(str)` is a key of metrics, to distinguish your train job from others. It can be, for example, `dlrm_v1.0`.

```
PersiaMetricsConfig:
  enable_metrics: true
  job_name: your_job_name
```

2. Deploy PushGateway

Every single job should hold a gateway node, instead of sharing it across jobs, to avoid bottlenecks caused by single point.

Here is an example for deploying gateway by [docker-compose], as the default push address of PersiaML is `metrics_gateway:9091`.

```
version: "3.3"
services:
    data_compose:
        ...

    trainer:
        ...

    middleware:
        ...

    server:
        ...

    persia_nats_service:
        ...

    metrics_gateway:
        image: prom/pushgateway:latest
        deploy:
            replicas: 1
```

Other deployment methods can be achieved by modifying environment variable `PERSIA_METRICS_GATEWAY_ADDR` to update push address of PersiaML nodes.

3. Service Discovery for Main Prometheus Server

TODO

[dockerswarm_sd_config]

[dns_sd_config]


## Metrics in PerisaML

1. Accuracy related

|  name   | implication  |
|  ----  | ----  |
| index_miss_count  |  |
| index_miss_ratio  |  |
| gradient_id_miss_count |  |
| estimated_distinct_id |  |
| batch_unique_indices_rate |  |
| staleness |  |
| nan_count |  |
| nan_grad_skipped |  |


2. Efficiency related

|  name   | implication  |
|  ----  | ----  |
| lookup_mixed_batch_time_cost |  |
| num_pending_batches |  |
| lookup_create_requests_time_cost |  |
| lookup_rpc_time_cost |  |
| update_gradient_time_cost |  |
| summation_time_cost |  |
| lookup_batched_time_cost |  |






[Prometheus]: https://prometheus.io/docs/introduction/overview/
[PushGateway]: https://github.com/prometheus/pushgateway
[docker-compose]: https://docs.docker.com/compose/
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
[dns_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dns_sd_config