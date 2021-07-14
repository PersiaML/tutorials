# Installation

## Use docker images 

The fastest way to start using PersiaML is using PersiaML's docker images.

For training, use the following images:

```bash
docker pull persiaml/persia-cpu-runtime:latest
docker pull persiaml/persia-gpu-runtime:latest
```

For inference, use the following image:

```bash
docker pull persiaml/persia-inference:latest
```

## Install manually

You can also install PersiaML manually on your existing system.

First install PersiaML Python client library from PyPI:

```bash
pip3 install persia
```

Then install PersiaML server binaries with [cargo](https://rustup.rs/):

```bash
cargo install persia-middleware persia-server
```

