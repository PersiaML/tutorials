# Getting Started

<!-- - [Use Docker Images](#use-docker-images)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image locally](#building-the-image-locally)
- [Install Manually](#install-manually)
    - [Common Requirements](#common-requirements)
    - [Install from Pip](#install-from-pip)
    - [Install from source](#install-from-source) -->

## Run on Kubernetes with PERSIA Operator (Recommended)

**Requirements**

* `kubectl` command-line tool
* valid `kubeconfig` file (default located at `~/.kube/config`)

**Installation**

```bash
kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

> **NOTE** It will take a few minutes to start the `operator` due to container image pulling.

**Run**

To run a simple example training task ([adult income prediction](https://archive.ics.uci.edu/ml/datasets/census+income)), apply the following Kubernetes PERSIA task definition file:

```bash
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/adult-income-prediction.train.yml
```

This runs the adult income prediction training task defined by `adult-income-prediction.train.yml`. This file defines the system configuration (e.g. resources limit, volume mounts, and environment variables) of a PERSIA training task.

To run a customized training task on your own dataset and models, we can customize the following configuration files:

- **Embedding configuration file:** A file defining the embedding configurations (e.g. embedding dimension, and sum pooling). This file is named as `embedding_config.yaml` by default. For more details see [embedding config](../configuration/index.md#embedding-config).
- **Embedding PS configuration file:** Configuration of embedding parameter servers, e.g. max capacity of embedding parameter servers. This file is named as `global_config.yaml` by default. For more details see [global config](../configuration/index.md#global-configuration).
- **Model definition configuration file:** A file that defines the neural network (NN) using PyTorch. This file is named as `train.py` by default. For more details see [model definition](../customization/index.md#model-definition).
- **Data preprocessing configuration file:** A file that defines the data preprocessing. This file is named as `data_loader.py` by default. For more details see [train data](../customization/index.md#training-data).

To change the file name for these configuration files, we can remap the
`embeddingConfigPath`, `globalConfigPath`, `nnWorkerPyEntryPath`,
`dataLoaderPyEntryPath` in the Kubernetes PERSIA task definition file. For more
details on how to customize Kubernetes PERSIA task definitions, see
[k8s launcher customization](../customization/index.md#k8s-launcher).

## Run Manually

To launch the PERSIA adult income task manually, the first step is to download the corresponding dataset and preprocess the train data and test data.We already prepare the script to help you finish this step.

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd examples/src/getting_started/data && ./prepare_data.sh
```

After download the adult income data.You can choose below launcher to start your first PERSIA task.

### Using Docker-Compose
Docker-compose is a container manager tool that launch multiple service once time. By edit the `docker-compose.yml` file, you can configure the PERSIA training environment such as cuda-runtime-image or cpu-runtime-image, gpu_num, service_replica_num, etc. see docker-compose [configuration](../customization/index.md#docker-compose-launcher) for more detail.


**Requirements**

* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/)

**Run**

We provide an adult income example `docker-compose.yml` file. Try below command to start your `PERSIA` task after install the `docker-compose` tools.

```bash
cd examples/src/getting_started && make run
```

### Using Python Package

**Requirements**

* [PERSIA python package](https://pypi.org/project/persia/) 
* [honcho](https://github.com/nickstenning/honcho) 
* nats-server [release page](https://github.com/nats-io/nats-server/releases) or [installation page](https://docs.nats.io/running-a-nats-service/introduction/installation)

**Using Pre-compiled Wheels**

We provide pre-compiled wheels for linux platform.If your python3 version is greater than 3.6 and cuda version greater than cuda10.2.You can install the PERSIA package directly by specific cuda version. 

```bash
pip3 install persia-cuda102 # install cuda102
pip3 install persia-cuda111 # install cuda111
pip3 install persia-cuda113 # install cuda113
```

**From Source**

For example on Ubuntu 20.04:

```bash
apt update && apt-get install -y curl build-essential git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path

git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA 
USE_CUDA=1 NATIVE=1 pip3 install .
```

**Run**

After installing PERSIA Python package locally, you can launch the example adult income prediction task with:

```bash
 # TODO: use dataset name for example dir
git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA/examples/honcho  
EXAMPLE=getting_started make run -e
```

## Deploy Trained Model for Inference

See [Inference](../inference/index.md).
