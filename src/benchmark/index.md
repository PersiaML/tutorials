Benchmark
======

## Setup

We evaluate Persia over three open-source benchmarks and one real-world production microvideo recommendation workflow at Kwai:

* [Taobao-Ad](https://www.kaggle.com/pavansanagapati/ad-displayclick-data-on-taobaocom): predict the advertisement CTR from Taobao’s website for 8 days with 26 million records.
* [Avazu-Ad](https://www.kaggle.com/c/avazu-ctr-prediction): predict the advertisement CTR of Avazu’s log for 11 days with 32 million records.
* [Criteo-Ad](https://www.kaggle.com/c/criteo-display-ad-challenge): predict the advertisement CTR of Criteo’s traffic for 24 days with 44 million records.
* Kwai-Video(confidential production dataset): predict the explicit behavior of Kwai’s active users about the microvideo recommendation in 7 days with 3 billion records.

For the three open source advertisement CTR benchmarks, we include 80% of the records as training set and the rest 20% of the records as test set, we consider a fully connected feed forward neural network (FFNN) as the deep learning model with five hidden layer dimensions of 4096, 2048, 1024, 512 and 256. For the Kwai production microvideo recommendation task, 85% of the data are included in the training set while the rest 15% are considered as the test set, we also use FFNN as the model to predict multiple user behaviors.

We include up to 64 Nvidia V100 GPUs, and 100 CPU instances (each with 52 cores and 480GB RAM). The instances are connected by a network with the bandwidth of 100 Gbps. The baseline systems (XDL and PaddlePaddle) are equipped with the same amount of computation resources for each individual setting.

## End-to-end performance

<img src="img/convergence.png" width="1200">

We see that the persia hybrid algorithm shows almost identical convergence when comparing with the fully synchronous mode. We see that test AUC gap between the hybrid mode and synchronous mode is always less than 0.1% in the three open-source benchmarks, and less than 0.001% in the production Kwai-video benchmark; by contrast, the gap between the asynchronous mode and the synchronous mode is much higher (from 0.5% to 1.0%); further, as we allow more aggressive asynchronicity in PaddlePaddle, the gap is more significant.


## Scalability: number of workers

<img src="img/scalability.png" width="1200">

Above figure illustrates significant performance improvements from Persia: e.g., for the Taobao-Ad benchmark, Persia is 7.12× and 8.4× faster than that of the synchronous and asynchronous modes of XDL, and 1.71× faster than PaddlePaddle–same level of speedup also appears in the Avazu-Ad and Criteo-Ad benchmark.


## Scalability: number of parameters

The intensive test of Persia’ capacity is conducted over Google cloud platform with a heterogeneous cluster including:
* 8 a2-highgpu-8g instances (each with 8 Nvidia A100 GPUs) as NN workers;
* 100 c2-standard-30 instances (each with 30vCPUs, 120GB RAM) as embedding workers;
* 30 m2-ultramem-416 instances (each with 416vCPUs, 12TB RAM) as embedding PS.

<img src="img/model_scalability.png" width="600">

We see that Persia shows stable training throughput when increasing the model size even up to 100 trillion parameters. For the 100 trillion-parameter model, Persia also achieves 2.6× higher throughput than the fully synchronous mode.