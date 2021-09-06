Benchmark
======

## Setup

We use up to 4 GPU servers and 16 CPU servers for benchmarks. Each of GPU servers is equipped with 8 NVIDIA V100 32GB GPUs interconnected by NVLink, connected by 100 Gbps TCP/IP network. Each of CPU servers is equipped with 64 core `Intel(R) Xeon(R) Gold 5218 CPU@2.30GHz`, connected by 10 Gbps TCP/IP network.

We compare the performance of Persia with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and [x-deepleaning](https://github.com/alibaba/x-deeplearning) on a DNN model which is one of the most popular and basic model for Click-Through Rate prediction. The DNN model consists of four parts, embedding layer, concat layer, mlp layer, `cross_entropy` loss. The embedding size of all sparse features are set to 16. The mlp layer is 6 fully connect layers with hidden size `[4096, 2048, 1024, 512, 256]`.

We conducted experiments on three datasets, including [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge), [Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) and [Alibaba AD](https://www.kaggle.com/pavansanagapati/ad-displayclick-data-on-taobaocom).

## End-to-end performance

The following figure shows the end-to-end train performance(test auc) of three datasets when using 8GPU.

<img src="https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Criteo%20Dataset.png" width="320"><img src="https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Avazu%20CTR%20Dataset.png" width="320"><img src="https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Alibaba%20AD%20Dataset.png" width="320">

The figure above demonstrates Persia's hybrid training mode has almost the same performance as the x-deepleaning's synchronous training mode, better than all asynchronous training mode.

## Rank Scalability

The following figure shows the rank scalability of three datasets when using 1,2,4,8GPU.

<img src="https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Criteo%20Dataset.png" width="320"><img src="https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Avazu%20CTR%20Dataset.png" width="320"><img src="https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Alibaba%20AD%20Dataset.png" width="320">

Results show that Persia can achieve obviously speedup compared with other systems.

## Model Size Scalability

<img src="https://github.com/PersiaML/paper-experiments/blob/main/img/persia_model_scalability.png" width="400">

The figure above demonstrates the model size scalability persia system, The size of the sparse model hardly affects the training speed of Persia.