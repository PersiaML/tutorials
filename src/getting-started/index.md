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
\$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
\$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
\$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
\$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

**Run**

To run a simple example training task (adult income prediction) TODO: @zhuxuefeng add hyperlink to the dataset, apply the following Kubernetes PERSIA task definition file:

```bash
\$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/adult-income-prediction.train.yml
```

This runs the adult income prediction training task defined by `adult-income-prediction.train.yml`. This file defines the system configuration (e.g. resources limit, volume mounts, and environment variables) of a PERSIA training task.

To run a customized training task on your own dataset and models, we can customize the following configuration files:

- **Embedding configuration file:** A file defining the embedding configurations (e.g. embedding dimension, and sum pooling). This file is named as `embedding_config.yaml` by default. For more details see #embedding config. TODO: @zhuxuefeng fix this link
- **Embedding PS configuration file:** Configuration of embedding parameter servers, e.g. max capacity of embedding parameter servers. This file is named as `global_config.yaml` by default. For more details see #global config. TODO: @zhuxuefeng fix this link
- **Model definition configuration file:** A file that defines the neural network (NN) using PyTorch. This file is named as `train.py` by default. For more details see #model definition. TODO: @zhuxuefeng fix this link
- **Data preprocessing configuration file:** A file that defines the data preprocessing. This file is named as `data_loader.py` by default. For more details see #train data. TODO: @zhuxuefeng fix this link

To change the file name for these configuration files, we can remap the `embeddingConfigPath`, `globalConfigPath`, `trainerPyEntryPath`, `dataLoaderPyEntryPath` in the Kubernetes PERSIA task definition file.

For more details. See #customization. TODO: @zhuxuefeng fix this link

## Run Manually

### Using Docker-Compose

**Requirements**

* [docker](https://docs.docker.com/engine/install/ubuntu/)
* [docker-compose](https://docs.docker.com/compose/)

**Installation**

TODO: @wangyulong  do we really need to pull images?

```bash
\$ $docker pull persiaml/persia-cuda-runtime:latest
```
> **NOTE** These docker images can be built from
> ```bash
> \$ git clone https://github.com/PersiaML/PERSIA.git
> # docker image name: persiaml/persia-cuda-runtime:dev
> \$ cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
> ```

**Run**

We provide an example `docker-compose.yml` file. Try below command to start your `PERSIA` task after install the `docker-compose` tools and `PERSIA` runtime image. # TODO: @wangyulong is this adult income ?

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/docker-compose
CODE_BASE=../src/getting_started/ make run -e
```

### Using Python Package

**Requirements**

* [honcho](xxx) TODO: @wangyulong fix the link
* [PERSIA python package](xxx) TODO: @wangyulong fix the link

**Using Pre-compiled Wheels**

TODO(wangyulong)

**From Source**

```bash
apt update && apt-get install -y curl git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path

git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA 
USE_CUDA=1 NATIVE=1 pip3 install persia
```

TODO: @wangyulong what's the difference?

> **NOTE** install from python setup.py
> ```bash
> pip3 install torch click colorlog colorama setuptools setuptools-rust setuptools_scm
> # install cpu version
> NATIVE=1 python3 setup.py install
> # install cuda version
> USE_CUDA=1 NATIVE=1 python3 setup.py install
> ```

**Run**

After installing PERSIA Python package locally, you can launch the example adult income prediction task by:

```bash
git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA/examples/honcho   # TODO: use dataset name for example dir
CODE_BASE=../src/getting_started/ honcho start
```

## Deployment

see #Deployment for inference  TODO: @zhuxuefeng  fix the link
