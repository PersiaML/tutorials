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
git clone git@github.com:PersiaML/PersiaML.git 
# docker image name: persiaml/persia-cuda-runtime:dev
cd PersiaML && make build_dev_image 
```
## Install Manually

You can also install PersiaML manually on your existing system.


### Common requirements
```bash
export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH

curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y --profile default --no-modify-path
rustup install nightly-2021-06-01

sudo apt-get install -y python3.7 libpython3.7-dev
```

### Install From Pip 
```bash
USE_CUDA=1 pip3 install persia
```

### Install From Source
```bash
git clone git@github.com:PersiaML/PersiaML.git 
pip3 install click colorlog torch colorama setuptools setuptools-rust setuptools_scm

# install cpu version
python3 setup.py install
# install cuda version
USE_CUDA=1 python3 setup.py install
```