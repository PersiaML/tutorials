# Getting Started


- [Run on Kubernetes with PERSIA Operator(Recommended)](#run-on-kubernetes-with-persia-operator)
- [Run Manually](#run-manually)
    - [Docker Compose](#using-docker-compose)
    - [Python Package](#using-python-package)

## Run on Kubernetes with PERSIA Operator

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

> **NOTE** It can take a few minutes to start the `operator` due to container image pulling.

**Run**

To run a simple example training task ([adult income prediction](https://archive.ics.uci.edu/ml/datasets/census+income)), apply the following Kubernetes PERSIA task definition file:

```bash
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/adult-income-prediction.train.yml
```

This runs the adult income prediction training task defined by `adult-income-prediction.train.yml`. This file defines the system configuration (e.g. resources limit, volume mounts, and environment variables) of a PERSIA training task.

To run a customized training task on your own dataset and models, we can customize the following configuration files:

- **Embedding configuration file:** A file defining the embedding configurations (e.g. embedding dimension, and sum pooling). This file is named as `embedding_config.yaml` by default. For more details see [embedding config](../configuration/index.md#embedding-config). TODO: @zhuxuefeng introduce each config file env variables
- **Embedding PS configuration file:** Configuration of embedding parameter servers, e.g. max capacity of embedding parameter servers. This file is named as `global_config.yaml` by default. For more details see [global config](../configuration/index.md#global-configuration).
- **Model definition configuration file:** A file that defines the neural network (NN) using PyTorch. This file is named as `train.py` by default. For more details see [model definition](../customization/index.md#model-definition).
- **Data preprocessing configuration file:** A file that defines the data preprocessing. This file is named as `data_loader.py` by default. For more details see [training data](../customization/index.md#training-data).

To change the file name for these configuration files, we can remap the
`embeddingConfigPath`, `globalConfigPath`, `nnWorkerPyEntryPath`,
`dataLoaderPyEntryPath` in the Kubernetes PERSIA task definition file. For more
details on how to customize Kubernetes PERSIA task definitions, see
[K8S launcher customization](../customization/index.md#k8s-launcher).

## Run Manually

To launch the PERSIA adult income task manually, the first step is to download the corresponding dataset and preprocess the train data and test data. We already prepare the script to help you finish this step.

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/src/adult-income/data && ./prepare_data.sh
```

After downloading the adult income dataset. You can choose from the following two methods to start your first PERSIA task.

### Using Docker-Compose

[Docker-compose](https://docs.docker.com/compose/) is a container management tool that can launch multiple services at once. By editing the `docker-compose.yml` file, you can customize the PERSIA training task (such as `docker-image`, `gpu_num`, and `service_replica_num`). See PERSIA docker-compose [configuration](../customization/index.md#docker-compose-launcher) for more detail.

TODO: fix content consistency @wangyulong

**Requirements**

* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/)

**Run**

We already provide the `docker-compose.yml` and `.docker.env` for adult income example. Try below command to start your `PERSIA` task after install the requirements.

```bash
cd examples/src/adult-income && make run
```

### Using Python Package

Alternatively, you can use PERSIA's Python packages directly to run a PERSIA task. In this way, users have the maximum flexibility (and you are free to modify source code to build and use your customized PERSIA Python packages).

**Requirements**

* [PERSIA python package](https://pypi.org/project/persia/) 
* [honcho](https://github.com/nickstenning/honcho) 
* [nats-server](https://docs.nats.io/running-a-nats-service/introduction/installation)

**Using Pre-compiled Wheels**
 
We provide pre-compiled wheels for linux platform. If your Python version is greater than 3.6. You can install pre-compiled PERSIA packages with:

TODO: use a table @wangyulong https://github.com/BaguaSys/bagua#installation

```bash
pip3 install persia-cuda102 # install cuda102
pip3 install persia-cuda111 # install cuda111
pip3 install persia-cuda113 # install cuda113
```

**From Source**

We provide the following instructions to build PERSIA Python packages from source (Ubuntu 20.04 & Windows 10. It should be similar on other OSes).

> **Note**: You need to provide environment variable `USE_CUDA=1` to add CUDA support (for GPU training). In this case, the CUDA runtime path should be already present in `LD_LIBRARY_PATH`.

**Ubuntu 20.04:**

```bash
apt update && apt-get install -y curl build-essential git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path

git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA 

# To install CUDA version
USE_CUDA=1 NATIVE=1 pip3 install .

# To install CPU version
NATIVE=1 pip3 install .
```

**Windows 10:**

Python3 and [Perl](https://strawberryperl.com/) are required.

TODO: @wangyulong

```bash

```

**Run**

After installing PERSIA Python package locally, you can launch the example adult income prediction training task with:

```bash
cd examples/src/adult-income
honcho start -e .honcho.env
```

For more configuration options see [Customization](../customization/index.md#honcho-launcher).

## Deploy Trained Model for Inference

See [Inference](../inference/index.md).
