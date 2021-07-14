# Installation

## Use docker images 

The fastest way to start using PersiaML is using PersiaML's docker images.

For training, use the following images:

```bash
docker pull persiaml/persia-cpu-runtime:latest
docker pull persiaml/persia-gpu-runtime:latest
```

<!-- move this to contributing doc
### For development

persia-dev image can build all you need for dev enviroment

```bash
docker pull persiaml/persia-dev:latest
```
-->

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

<!-- move these to contributing doc
**build persia-core from source**
```bash
# build persia-core gpu version
git clone https://github.com/PersiaML/PersiaML-server
mkdir build
docker run -it --rm -v $(realpath PersiaML-server):/workspace/ \
    -e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
    --network=host -v $(realpath build):/build/ \
    persia-ci:master bash -c "cd /workspace && \
    cargo build --release --package persia-core --features cuda --target-dir /build && \
    cd /build/release && mv libpersia_core.so /build/persia_core.so && \
    rm -rf /build/release"

# build persia-core cpu version without features flag
docker run -it --rm -v $(realpath PersiaML-server):/workspace/ \
    -e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
    --network=host -v $(realpath build):/build/ \
    persia-ci:master bash -c "cd /workspace && \
    cargo build --release --package persia-core --target-dir /build && \
    cd /build/release && mv libpersia_core.so /build/persia_core.so && \
    rm -rf /build/release"
```
**build persia python from source**

```bash
# build persia python library
git clone https://github.com/PersiaML/PersiaML
cd PersiaML && pip3 install .
```


### Rust libary installation


*build from source*
```bash
git clone https://github.com/PersiaML/PersiaML-server
mkdir build
docker run -it --rm -v $(realpath PersiaML-server):/workspace/ \
    -e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
    --network=host -v $(realpath build):/build/ \
    persia-ci:master bash -c "cd /workspace && \
    cargo build --release --package persia-embedding-sharded-server --target-dir /build && \
    mv /build/release/persia-embedding-sharded-middleware /build/ && \
    mv /build/release/persia-embedding-sharded-server /build/ && \
    rm -rf /build/release"
```
-->
