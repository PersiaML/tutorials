Model Checkpointing
======

A PerisaML model contains two parts: dense and sparse.

Since pytorch is used directly in the calculation of the dense part, the pytorch api can be used directly for model saving, see [Saving and Loading Models].

For sparse part, 


[Saving and Loading Models]: https://pytorch.org/tutorials/beginner/saving_loading_models.html