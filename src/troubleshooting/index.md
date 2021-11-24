# Troubleshooting

Debugging running task in distributed training can be difficult. We provide
some general suggestions in this page.

## Debug Log

You can set the logging verbosity with `LOG_LEVEL` environment variable when
launching PERSIA. The value of *LOG_LEVEL* can be *debug*, *info*, *warn*, or
*error*, the default value is *info*.

## Grafana Metrics

PERSIA integrates [Prometheus](https://prometheus.io/) to report useful metrics
during training phase. This includes current embedding staleness, current total
embedding size, the time cost of each stage during an iteration, and more. See
[Monitoring](../monitoring/index.md) for more details.
