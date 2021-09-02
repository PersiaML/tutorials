# Troubleshooting

Debugging running task in distributed training can be difficult.We provide some general suggestions in this page.

## Investigate debug log
PersiaML print the log according to different `LOG_LEVEL`.Set corresponding launch environment `LOG_LEVEL` when launch the *trainer*, *middleware-server* and *embedding-server*. The value of *LOG_LEVEL* can be accepted include *debug*, *info*, *warn* and *error* can be accept, the default value is `info`.

## Investigate Grafana metrics
We use the `Prometheus` to collect the metrics in training phase.You can find out the information such as current embedding staleness, current embedding_size or the time cost of embedding backward. Read the [Monitoring cheapter](../monitoring/index.md) for more metric.


## Print intermediate results
Print the intermediate result is also necessary when meet some tough problems.When you need to print the intermediate results during the training phase, there are two solutions that you can do to after adding the intermediate log.The one is building a new docker image.And the second is install the persiaML python library manully. Read the [Installation cheapter](../installation/index.md) for more detail.
