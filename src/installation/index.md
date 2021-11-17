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

refer to https://argo-cd.readthedocs.io/en/stable/getting_started/

Requirements

* Installed kubectl command-line tool.
* Have a kubeconfig file (default location is ~/.kube/config).

Installation

```bash
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/jobs.persia.com.yaml
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/resources/operator.persia.com.yaml
```

Run

```bash
$ kubectl apply -f https://raw.githubusercontent.com/PersiaML/PERSIA/main/k8s/example/k8s.train.yml
```

### Access the Control Panel

How to access the UI

## Run Manually

TODO(wangyulong):

### Using Docker-compose

TODO: provide docker-compose.yaml and how to run

> Note: these docker images can be built from xxxx with
> ```bash
> git clone https://github.com/PersiaML/PERSIA.git
> # docker image name: persiaml/persia-cuda-runtime:dev
> cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
> ```

TODO: provide example config, and link to docs

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
rustup install nightly-2021-06-01
```

```bash
USE_CUDA=1 NATIVE=1 pip3 install persia
```

```bash
git clone https://github.com/PersiaML/PERSIA.git
pip3 install torch click colorlog colorama setuptools setuptools-rust setuptools_scm

cd PERSIA
# install cpu version
NATIVE=1 python3 setup.py install
# install cuda version
USE_CUDA=1 NATIVE=1 python3 setup.py install
```

## How to Run?

We provided several examples and multiple type of launcher to help you quick start a *PersiaML* task.

### Kubernetes Launcher
TODO(@zhuxuefeng)

### Docker Compose Launcher

Docker [compose](https://docs.docker.com/compose/) can launch multiple services under the swarm mode.Follow the [swarm mode](https://docs.docker.com/engine/swarm/) to adding multiple machines to swarm cluster to apply the distributed PersiaML training task.

*launcher example below*
```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/docker-compose
CODE_BASE=../src/getting_started/ make run -e
```

### Honcho Launcher
[Honcho](https://github.com/nickstenning/honcho) is a tool for managing multiple processes.Current honcho launcher only support launch the PersiaML Task in single node due to some distributed environments is hard to shared across multiple nodes.

*launcher example below*
```bash
git clone https://github.com/PersiaML/PERSIA.git
cd PERSIA/examples/honcho
pip3 install honcho
CODE_BASE=../src/getting_started/ honcho start
```

