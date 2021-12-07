Model Checkpointing
======

A PERSIA model contains two parts: the dense part and the sparse part (embeddings). When it comes to saving and loading the model, whether you want to save the dense part and sparse part together, or separately in different locations, PERSIA model checkpointing api provides handy solutions for both situations.

## Checkpointing Together

You can call `load_checkpoint` or `dump_checkpoint` in a PERSIA context. Both the dense part and the sparse part will be saved into `checkpoint_dir`. By default, the model will be saved to the local path. When the path start with `hdfs://`, the model will be saved to hdfs path.

```python
with TrainCtx(
    model=model,
    embedding_optimizer=embedding_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
) as ctx:
    ctx.load_checkpoint(checkpoint_dir)
    if batch_idx % 10000 == 0:
        ctx.dump_checkpoint(checkpoint_dir)
```

## Checkpointing Separately

Since PyTorch is used for defining the dense part, it can be used directly for saving the dense part. See [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

For the sparse part, you need to use PERSIA API to do model checkpointing.

In a persia context, you can load or dump the sparse part checkpoint in a directory with the `load_embedding`, `dump_embedding` method:

```python
with TrainCtx(
    model=model,
    embedding_optimizer=embedding_optimizer,
    dense_optimizer=dense_optimizer,
    device_id=device_id,
) as ctx:
    ctx.load_embedding(checkpoint_dir, True)
    if batch_idx % 10000 == 0:
        ctx.dump_embedding(checkpoint_dir, True)
```

Relavant configurations in [`global_config.yaml`](../configuration/index.md) are `checkpointing_config`.
