Model Checkpointing
======

A PERSIA model contains two parts: the dense part and the sparse part (embeddings). When it comes to saving and loading the model, whether you want to save the dense part and sparse part together or separately, PERSIA model checkpointing API provides handy solutions for both situations.

<!-- toc -->

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

Since the dense part of a PERSIA model is simply a `torch.nn.module`, you can use Pytorch API to checkpoint the dense part. See [Pytorch tutorials: Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) for guidance on how to save and load model in Pytorch.

For the sparse part, you need to use PERSIA API to do model checkpointing.

In a PERSIA context, you can load or dump the sparse part checkpoint in a directory with the `load_embedding`, `dump_embedding` method:

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
