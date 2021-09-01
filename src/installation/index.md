# Installation

- [Use Docker Dmages ](#use-docker-images)
    - [Using pre-build images](using-prebuild-images)
    - [Building the image locally](build-the-image-locally)
- [Install Manually](#install-manually)
    - [Common Requirements](#common-requirements)
    - [Install from Pip](#install-from-pip)
    - [Install from source](#install-from-source)


## Docker Image

The fastest way to start using PersiaML is using prebuild docker images.
### Using pre-build image
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