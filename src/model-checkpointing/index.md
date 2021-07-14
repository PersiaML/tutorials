Model Checkpointing
======

A PerisaML model contains two parts: dense and sparse.

Since pytorch is used in the calculation of the dense part, the pytorch api can be used directly for model saving, see [Saving and Loading Models].

For sparse part, there are apis for model checkpointing.

Here is an example. There is parameters called `initial_embedding_dir(str)` that indicate the directory of the checkpoint saved from last training. Once enter TrainCtx, embedding server will load this ckeckpoint and block training until load compelete.

```python
with TrainCtx(
    grad_scaler=scaler,
    sparse_optimizer=sparse_optimizer,
    rank_id=RANK_ID,
    world_size=WORLD_SIZE,
    initial_embedding_dir='/your/latest/checkpoint/'
) as ctx:
    if batch_idx % 10000 == 0:
        ctx.dump_embedding(f'{embedding_dir}/{datetime}_{batch_idx}', True)
```

There are configures in `global_config.yaml` about model checkpointing.

|  name   | implication  |
|  ----  | ----  |
| `storage` | dump or load embedding to ceph or hdfs.|
| `num_persistence_workers` | parallelism of dump or load embedding. |
| `num_signs_per_file` | num of indices dumped to a checkpoint file.(embedding dump split into files)  |


[Saving and Loading Models]: https://pytorch.org/tutorials/beginner/saving_loading_models.html