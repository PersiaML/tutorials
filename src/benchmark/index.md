Benchmark
======

## Setup

We use up to 4 GPU servers and 32 CPU servers for benchmarks. Each of GPU servers is equipped with 8 NVIDIA V100 32GB GPUs interconnected by NVLink, connected by 100 Gbps TCP/IP network. Each of CPU servers is equipped with 64 core `Intel(R) Xeon(R) Gold 5218 CPU@2.30GHz`, connected by 10 Gbps TCP/IP network.

We compare the performance of Persia with [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and [x-deepleaning](https://github.com/alibaba/x-deeplearning) on a DNN model which is one of the most popular and basic model for Click-Through Rate prediction. We conducted experiments on three datasets, including [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge), [Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) and [Alibaba AD](https://www.kaggle.com/pavansanagapati/ad-displayclick-data-on-taobaocom)
.

## End-to-end performance


![AUC on Criteo Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Criteo%20Dataset.png)
![AUC on Avazu CTR Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Avazu%20CTR%20Dataset.png)
![AUC on Alibaba AD Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/AUC%20on%20Alibaba%20AD%20Dataset.png)


The figure above demonstrates the end-to-end train performance of three datasets when using 8GPU.


## Rank Scalability

![Scalibility on Criteo Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Criteo%20Dataset.png)
![Scalibility on Avazu CTR Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Avazu%20CTR%20Dataset.png)
![Scalibility on Alibaba AD Dataset](https://github.com/PersiaML/paper-experiments/blob/main/img/Scalibility%20on%20Alibaba%20AD%20Dataset.png)

The figure above demonstrates the rank scalability of three datasets when using 1,2,4,8GPU.

## Model Size Scalability

![Persia Model Scalability](https://github.com/PersiaML/paper-experiments/blob/main/img/persia_model_scalability.png)

The figure above demonstrates the model size scalability persia system.