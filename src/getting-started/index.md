# Getting Started


- [Run on Kubernetes with PERSIA Operator(Recommended)](#run-on-kubernetes-with-persia-operator)
- [Run Manually](#run-manually)
    - [docker-compose](#using-docker-compose)
    - [python-package](#using-python-package)

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
cd PERSIA/examples/src/adult-income/data && ./prepare_data.sh
```

After download the adult income data.You can choose below launcher to start your first PERSIA task.

### Using Docker-Compose
Docker-compose is a container manager tool that launch multiple services at once time. By edit the `docker-compose.yml` file, you can configure the PERSIA training environment such as docker-image, gpu_num, service_replica_num, etc. see PERSIA docker-compose [configuration](../customization/index.md#docker-compose-launcher) for more detail.


**Requirements**

* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/)

**Run**

We already provide the `docker-compose.yml` and `.docker.env` for adult income example. Try below command to start your `PERSIA` task after install the requirements.

```bash
cd examples/src/adult-income && make run
```

### Using Python Package
Using the PERSIA python package can help you improve development experience, you can explore the PERSIA docstring and modify the PERSIA python code to cover the debug requirement if currently log cannot satisfy your requirements.You can even to modify the PERSIA rust source code to contribute this project.


**Requirements**

* [PERSIA python package](https://pypi.org/project/persia/) 
* [honcho](https://github.com/nickstenning/honcho) 
* nats-server [release page](https://github.com/nats-io/nats-server/releases) or [installation page](https://docs.nats.io/running-a-nats-service/introduction/installation)

**Using Pre-compiled Wheels**

We provide pre-compiled wheels for linux platform.If your python version is greater than 3.6. You can install the PERSIA package with specific cuda version suffix. 

```bash
pip3 install persia-cuda102 # install cuda102
pip3 install persia-cuda111 # install cuda111
pip3 install persia-cuda113 # install cuda113
```

**From Source**

If you wanna build PERSIA cuda version, you should specific the environment `USE_CUDA=1` to include the cuda dependency.And don't forget to add system cuda runtime path into **LD_LIBRARY_PATH**.

For example on Ubuntu 20.04:


```bash
apt update && apt-get install -y curl build-essential git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path

git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA 

# for PERSIA cuda-version
USE_CUDA=1 NATIVE=1 pip3 install .

# for PERSIA cpu-version
NATIVE=1 pip3 install .
```

For Windows10:

Python3 and Perl are required.Download the [perl](https://strawberryperl.com/) requirements to avoid build failure.But

**Run**

After installing PERSIA Python package locally, you can launch the example adult income prediction task with:

```bash
examples/src/adult-income
honcho start -e .honcho.env
```
For more Python package configuration can review this [chapter](../customization/index.md#honcho-launcher) for more detail.
## Deploy Trained Model for Inference

See [Inference](../inference/index.md).
