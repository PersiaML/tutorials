Model Checkpointing
======

A PerisaML model contains two parts: the dense part and the sparse part (embeddings). A checkpoint contains both the dense part and the sparse part can be saved together through Persia api or manually saved separately.

## Checkpointing together

You can call `load_checkpoint` or `dump_checkpoint` in a persia context, both the dense part and the sparse part will be saved into `checkpoint_dir`.The model will be saved to the local path by default, when the path start with `hdfs://`, it will be saved to hdfs path.

```python
with TrainCtx(
    model=model,
    sparse_optimizer=sparse_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
    embedding_config=embedding_config,
) as ctx:
    ctx.load_checkpoint(checkpoint_dir)
    if batch_idx % 10000 == 0:
        ctx.dump_checkpoint(checkpoint_dir)
```

## Checkpointing separately

Since PyTorch is used for defining the dense part, it can be used directly for saving the dense part, see [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

For the sparse part, you need to use PERSIA API to do model checkpointing.

In a persia context, you can load or dump the sparse part checkpoint in a directory with the `load_embedding`, `dump_embedding` method:

```python
with TrainCtx(
    model=model,
    sparse_optimizer=sparse_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
    embedding_config=embedding_config,
) as ctx:
    ctx.load_embedding(checkpoint_dir, True)
    if batch_idx % 10000 == 0:
        ctx.dump_embedding(checkpoint_dir, True)
```

Relavant configurations in [`global_config.yaml`](../configuration/index.md) are `checkpointing_config`.
