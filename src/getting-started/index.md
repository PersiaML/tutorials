# Getting Started


- [Run on Kubernetes with PERSIA Operator (Recommended)](#run-on-kubernetes-with-persia-operator)
- [Run Manually](#run-manually)
    - [Docker Compose](#using-docker-compose)
    - [Python Package](#using-python-package)

## Run on Kubernetes with PERSIA Operator

**Requirements**

* `kubectl` command-line tool
* valid `kubeconfig` file (by efault located at `~/.kube/config`)

**Installation**

```bash
kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

> **NOTE:** It can take a few minutes to start the `operator` due to container image pulling.

**Run**

To run a simple example training task ([adult income prediction](https://archive.ics.uci.edu/ml/datasets/census+income)), apply the following Kubernetes PERSIA task definition file:

```bash
kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/adult-income-prediction.train.yml
```

This runs the adult income prediction training task defined by `adult-income-prediction.train.yml`. This file defines system configuration (e.g. resources limit, volume mounts) and environment variables (with paths to embedding, model and data configuration files) of a PERSIA training task.

To run a customized training task on your own dataset and models, you can edit the following configuration files:

- **Embedding configuration file:** A file defining the embedding configurations (e.g. embedding dimension, and sum pooling). This file is named as `embedding_config.yaml` by default. For more details see [embedding config](../configuration/index.md#embedding-config).
- **Embedding PS configuration file:** Configuration of embedding parameter servers, e.g. max capacity of embedding parameter servers. This file is named as `global_config.yaml` by default. For more details see [global config](../configuration/index.md#global-configuration).
- **Model definition configuration file:** A file that defines the neural network (NN) using PyTorch. This file is named as `train.py` by default. For more details see [model definition](../customization/index.md#model-definition).
- **Data preprocessing configuration file:** A file that defines the data preprocessing. This file is named as `data_loader.py` by default. For more details see [training data](../customization/index.md#training-data).

The location of these files can be specified using the environment variables `PERSIA_EMBEDDING_CONFIG`, `PERSIA_GLOBAL_CONFIG`, `PERSIA_NN_WORKER_ENTRY`, `PERSIA_DATALOADER_ENTRY` respectively. For more
details on how to customize these environment variables, see
[launcher configuration](../customization/index.md#launcher-configuration).

## Run Manually

The data of adult income should be downloaded and preprocessed before you get started to run the example PERSIA training task:

<!-- To launch the PERSIA adult income prediction task  manually, the first step is to download the corresponding dataset and preprocess the train data and test data. We already prepare the script to help you finish this step. -->

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/src/adult-income/data && ./prepare_data.sh
```

<!-- After downloading the adult income dataset. You can choose from the following two methods to start your first PERSIA task. -->

Now you can start your first PERSIA training task with one of the following methods.

### Using Docker-Compose

[Docker-compose](https://docs.docker.com/compose/) is a tool for defining and running multi-container docker applications. By modifying the `docker-compose.yml` file, you can customize the PERSIA training task (such as `image`, `replicas`). See PERSIA docker-compose [configuration](../customization/index.md#docker-compose-launcher) for more details.

**Requirements**

* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/)

**Run**

<!-- We already provide the `docker-compose.yml` and `.docker.env` for adult income example.  -->
Use the following instructions to start your PERSIA training task after installing the requirements.

```bash
cd examples/src/adult-income && make run
```

### Using Python Package

You are free to modify PERSIA source code and build your customized PERSIA Python package.

**Requirements**

* [PERSIA python package](https://pypi.org/project/persia/)
* [honcho](https://github.com/nickstenning/honcho)
* [nats-server](https://docs.nats.io/running-a-nats-service/introduction/installation)

**Using Pre-compiled Wheels**

Wheels (precompiled binary packages) are available for Linux (x86_64). Package names are different depending on your CUDA Toolkit version (CUDA Toolkit version is shown in` nvcc --version`). All of these precompiled binary packages need Python greater than 3.6.

|CUDA Toolkit version|Installation command|
|-|-|
|None (CPU version) |`pip3 install persia`|
|>= v10.2|`pip3 install persia-cuda102`|
|>= v11.1|`pip3 install persia-cuda111`|
|>= v11.3|`pip3 install persia-cuda113`|

**From Source**

Use following instructions to build PERSIA Python packages from source (Ubuntu 20.04 & Windows 10. It should be similar on other OSes).

> **Note**: You need to set environment variable `USE_CUDA=1` to add CUDA support (for GPU training). In this case, the CUDA runtime path should be already present in `LD_LIBRARY_PATH`.

**<center>Ubuntu 20.04</center>**

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


**<center>Windows 10</center>**


Python3, [Perl](https://strawberryperl.com/) and [Rust](https://www.rust-lang.org/tools/install) are required.

```bash
git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA

# To install CUDA version
USE_CUDA=1 NATIVE=1 pip3 install .

# To install CPU version
NATIVE=1 pip3 install .
```

**Run**

After installing the PERSIA Python package locally, you are able to launch the example adult income prediction training task with:

```bash
cd examples/src/adult-income
honcho start -e .honcho.env
```

For more configuration options see [Customization](../customization/index.md#honcho-launcher).

## Deploy Trained Model for Inference

See [Inference](../inference/index.md).
