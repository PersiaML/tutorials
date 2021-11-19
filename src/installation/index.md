# Getting Started

<!-- - [Use Docker Images](#use-docker-images)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image locally](#building-the-image-locally)
- [Install Manually](#install-manually)
    - [Common Requirements](#common-requirements)
    - [Install from Pip](#install-from-pip)
    - [Install from source](#install-from-source) -->

## Run on Kubernetes with PERSIA Operator (Recommended)

TODO(@zhuxuefeng)

**Requirements:**

* Installed `kubectl` command-line tool.
* Have a `kubeconfig` file (default location is `~/.kube/config`).

**Installation:**

```bash
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

**Run:**

To run a basic example training task(adult income prediction), use following command.

```bash
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/k8s.train.yml
```

This runs a basic example training task by an operator definition file `k8s.train.yml`, which define the system configuration(e.g. resources limit, volume mounts, environment variables) of a persia task.

Generally, to run a customized training task, you need your own definition of models, configurations and data processing, and mount them to your training tasks. They are:

- Embedding configuration file: A file define the embedding parameters, e.g. embedding dim, whether to do summation, usually named as `embedding_config.yaml`. For more details see #embedding config.
- Server configuration file: Configuration of embedding servers, e.g. capacity of embedding servers, usually named as `global_config.yaml`. For more details see #global config.
- Model definition file: A file define the dense model with torch, usually named as `train.py`. For more details see #model definition.
- Data preprocessing file: A file define the preprocessing of data, usually named as `data_compose.py`. For more details see #train data.

There are fileds for these files in an operator definition file, you can change them to a specify path. They are: `embeddingConfigPath`, `globalConfigPath`, `trainerPyEntryPath`, `dataLoaderPyEntryPath`.

<!-- By default, there files are in the following locations in every container of the K8S application:

- configuration file: /data/configuration.yml
- xxxx file: .... -->

For more details. See #customization.


## Run Manually

### Using Docker-compose

**Requirements:**

* [docker](https://docs.docker.com/engine/install/ubuntu/) command line tools
* [dockerc-compose](https://docs.docker.com/compose/) command line tools

**Installation:**

```bash
docker pull persiaml/persia-cuda-runtime:latest
```
> Note: these docker images can be built from preset command after download source repo
> ```bash
> git clone https://github.com/PersiaML/PERSIA.git
> # docker image name: persiaml/persia-cuda-runtime:dev
> cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
> ```

**Run:**

We provide the preset `docker-compose.yml` file in our examples.Try below command to start your `PERSIA` task after install the `docker-compose` tools and `PERSIA` runtime image.

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/docker-compose
CODE_BASE=../src/getting_started/ make run -e
```

### Using Python Package

**Requirements**

```bash
pip3 install honcho
```

**Using Pre-compiled Wheels**:

TODO(wangyulong)
**From Source**:

```bash
apt update && apt-get install -y curl git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH
curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path

git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA 
USE_CUDA=1 NATIVE=1 pip3 install persia
```
> NOTE: install from python setup.py
> ```bash
> pip3 install torch click colorlog colorama setuptools setuptools-rust setuptools_scm
> # install cpu version
> NATIVE=1 python3 setup.py install
> # install cuda version
> USE_CUDA=1 NATIVE=1 python3 setup.py install
> ```

**Run:**

Launche the PERSIA and explore the all processes output log.
```bash
git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA/examples/honcho   # TODO: use dataset name for example dir
CODE_BASE=../src/getting_started/ honcho start
```

## Deployment

see #Deployment for inference
