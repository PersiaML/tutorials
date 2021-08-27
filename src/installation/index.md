# Installation

## Use docker images 

The fastest way to start using PersiaML is using docker images.

```bash
docker pull persiaml/persia-cuda-runtime:latest
```

## Install manually

You can also install PersiaML manually on your existing system.

First install PersiaML Python client library from PyPI:

```bash
pip3 install persia
```

Or install PersiaML locally from source:
```bash
# install rust compile denpendency
export RUSTUP_HOME=/rust
export CARGO_HOME=/cargo
export PATH=/cargo/bin:/rust/bin:$PATH

curl -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y --profile default --no-modify-path
chown -R 1000:1000 /rust /cargo 
rustup install nightly-2021-06-01

# install python package
git clone git@github.com:PersiaML/PersiaML.git
cd PersiaML && pip3 install . -v
```
