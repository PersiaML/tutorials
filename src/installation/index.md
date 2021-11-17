# Installation

- [Use Docker Images ](#use-docker-images)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image locally](#building-the-image-locally)
- [Install Manually](#install-manually)
    - [Common Requirements](#common-requirements)
    - [Install from Pip](#install-from-pip)
    - [Install from source](#install-from-source)


## Use Docker Images

The fastest way to training your first Persia task is using pre-built docker images.
### Using pre-built images
```bash
docker pull persiaml/persia-cuda-runtime:latest
```
### Building the image locally
```bash
git clone https://github.com/PersiaML/PERSIA.git
# docker image name: persiaml/persia-cuda-runtime:dev
cd PERSIA && IMAGE_TAG=dev make build_cuda_runtime_image -e
```
## Install Manually

You can also install PersiaML manually on your existing system.


### Common requirements
```bash
apt update && apt-get install -y curl git python3 python3-dev python3-pip 

export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH

curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path
rustup install nightly-2021-06-01

```

### Install From Pip 
```bash
USE_CUDA=1 NATIVE=1 pip3 install persia
```

### Install From Source
```bash
git clone https://github.com/PersiaML/PERSIA.git
pip3 install torch click colorlog colorama setuptools setuptools-rust setuptools_scm

cd PERSIA
# install cpu version
NATIVE=1 python3 setup.py install
# install cuda version
USE_CUDA=1 NATIVE=1 python3 setup.py install
```