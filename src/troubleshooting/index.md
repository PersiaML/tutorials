# Troubleshooting

Debugging a distributed training system can be difficult. We provide some general suggestions in this page.

## Investigate debug log
PersiaML print the log according to different `LOG_LEVEL`.Set corresponding `LOG_LEVEL` when launch the PersiaML, `LOG_LEVEL` in `debug`, `info`, `warn`, `error`, the default `LOG_LEVEL` is `info`.

## Investigate Grafana metrics
We use the `Prometheus` to collect the metrics in training phase.You can find out the information such as current embedding staleness, current embedding_size or the time cost of embedding backward. Read the [Monitoring cheapter](https://github.com/PersiaML/tutorials/blob/main/src/monitoring/index.md) for more metric.


## Print intermediate results
Print the intermediate result is also necessary when meet some tough problem.When you need to print the intermediate results during training phase, the first step you need to do is build a new `perisaml/persia-cuda-runtime:latest` after adding the intermediate log into PersiaML source code.


## FAQ

### DDP launch failed

Most PyTorch DDP launch failed error comes from incorrect [NCCL](https://developer.nvidia.com/nccl) configuration. We suggest checking the followings
- Set `NCCL_SOCKET_IFNAME`to your physical network device name, and set `CUDA_VISIBLE_DEVICES` properly.
- Launch the training job according to the instructions in [PyTorch DDP documentation](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).

### `persia-core` ImportError

There are two versions of `persia-core`, a CPU version that is used for inference, and a GPU (CUDA) version used for GPU training. Make sure you installed the correct version.
