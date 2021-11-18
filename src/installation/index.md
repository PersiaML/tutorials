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
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

**Run:**

To run PERSIA, xxxxx

```bash
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/k8s.train.yml
```

This runs a basic example training task (xxxxxx adult_income).

Generally, to run a customized training task, you need your own definition of models, configurations and data processing, and mount them to your training tasks.

By default, there files are in the following locations in every container of the K8S application:

- configuration file: /data/configuration.yml
- xxxx file: ....

For more details. See #customization.
<!-- 
 shared stroage, and put data, python entries and configurations to the shared stroage.

If you are using nfs, for example, if the NFS is mounted at `/nfs/`, then you can store data in `/nfs/general/data/adult_income/`, put python entries and configurations to `/nfs/general/PersiaML/e2e/adult_income/`, then you can run persia by following command.  -->



### Access the Control Panel

How to access the UI

## Run Manually

### Using Docker-compose

> Note: these docker images can be built from preset command after download source repo
> ```bash
> git clone https://github.com/PersiaML/PERSIA.git
> # docker image name: persiaml/persia-cuda-runtime:dev
> cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
> ```

Follow the running chapter after download the docker docker image.

### Docker Compose Launcher

Docker [compose](https://docs.docker.com/compose/) can launch multiple services under the swarm mode.Follow the [swarm mode](https://docs.docker.com/engine/swarm/) to adding multiple machines to swarm cluster to apply the distributed PersiaML training task.

We provide the preset `docker-compose.yml` file in our examples.Try below command to start your `PERSIA` task after install the `docker-compose` tools and `PERSIA` runtime image.

```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/docker-compose
CODE_BASE=../src/getting_started/ make run -e
```

### Using Python Package

#### Install Python Package

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
> git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA
> pip3 install torch click colorlog colorama setuptools setuptools-rust setuptools_scm
> # install cpu version
> NATIVE=1 python3 setup.py install
> # install cuda version
> USE_CUDA=1 NATIVE=1 python3 setup.py install
> ```

## How to Run?

We provided several examples and multiple type of launcher to help you quick start a *PersiaML* task.

### Kubernetes Launcher
TODO(@zhuxuefeng)

### Honcho Launcher
[Honcho](https://github.com/nickstenning/honcho) is a tool for managing multiple processes.Current honcho launcher only support launch the PersiaML Task in single node due to some distributed environments is hard to shared across multiple nodes.

Try below command to launch the training task, we already prepare the corresponding honcho preset env file.

```bash
pip3 install honcho
git clone https://github.com/PersiaML/PERSIA.git && cd PERSIA/examples/honcho
CODE_BASE=../src/getting_started/ honcho start
```

## Deployment

see xxxxxx.md