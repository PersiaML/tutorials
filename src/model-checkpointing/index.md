Model Checkpointing
======

A PerisaML model contains two parts: the dense part and the sparse part (embeddings).

Since PyTorch is used for defining the dense part, it can be used directly for saving the dense part, see [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

For the sparse part, we need to use PersiaML API to do model checkpointing.

There is an `initial_embedding_dir(str, default: None)` argument in the training context. It specifies the directory to load the sparse part checkpoint at the beginning of the training process. Once enter `TrainCtx`, PersiaML services will load the ckeckpoint.

During training, we can dump the sparse part checkpoint to a directory with the `dump_embedding(...)` method:

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

Relavant configurations in [`global_config.yaml`](../configuring/index.md) are `num_persistence_workers` and `num_signs_per_file`.
