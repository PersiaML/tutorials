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

Requirements¶

    Installed kubectl command-line tool.
    Have a kubeconfig file (default location is ~/.kube/config).

Installation

kubectl create namespace persia
kubectl apply -n argocd -f https://raw.githubusercontent.com/xxxxxxxxx/install.yaml

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

#### TODO: How to Run?


