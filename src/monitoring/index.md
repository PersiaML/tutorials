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

It would be better that every single job hold a gateway node, instead of sharing it across jobs, to avoid bottlenecks caused by single point.

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

Other deployment methods can be achieved by setting environment variable `PERSIA_METRICS_GATEWAY_ADDR` to update push address of PersiaML nodes.

In any of container in this job, metrics could be queried by this command:

```
curl metrics_gateway:9091/metrics
```

3. Service Discovery for Prometheus Server

To gather metrics for all jobs, push gateway service should be discoveried by the prometheus server.

Here is methods to discovery push gateway service by [docker_sd_config], [kubernetes_sd_config] or [dockerswarm_sd_config].


## Metrics in PerisaML

1. Accuracy related

|  name   | implication  |
|  ----  | ----  |
| index_miss_count  | miss count of indices when lookup |
| index_miss_ratio  | miss ratio of indices when lookup for one batch |
| gradient_id_miss_count | num of not found indices when updating gradient. |
| estimated_distinct_id | estimated distinct id for every feature. |
| batch_unique_indices_rate | unique indices rate in one batch |
| staleness | staleness of sparse model |
| nan_grad_skipped | nan gradient count |


2. Efficiency related

|  name   | implication  |
|  ----  | ----  |
| lookup_mixed_batch_time_cost | lookup embedding time cost on embedding server |
| num_pending_batches | num batches already sent to middleware but still waiting for trainer to trigger lookup |
| lookup_create_requests_time_cost | lookup preprocess time cost on middleware |
| lookup_rpc_time_cost | lookup embedding time cost on middleware server |
| update_gradient_time_cost | update gradient time cost on middleware server |
| summation_time_cost | lookup postprocess time cost on middleware |
| lookup_batched_time_cost | lookup and pre/post process time cost on middleware server |






[Prometheus]: https://prometheus.io/docs/introduction/overview/
[PushGateway]: https://github.com/prometheus/pushgateway
[docker-compose]: https://docs.docker.com/compose/
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config
[docker_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#docker_sd_config
[kubernetes_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#kubernetes_sd_config
[dockerswarm_sd_config]: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dockerswarm_sd_config