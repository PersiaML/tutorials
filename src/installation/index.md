# Installation PersiaML
PersiaML System consist of serval components.
- Runtime image
- Python library
- Rust binary  articraft

## Pull the dependent images 
### for production scence
```bash
docker pull persiaml/persia-cpu-runtime:latest
docker pull persiaml/persia-gpu-runtime:latest
```

### for dev scence
persia-dev image can build all you need for dev enviroment
```bash
docker pull persiaml/persia-dev:latest
```

### for deploy scence
deploy enviroment should use the image with `torch serve`
```bash
docker pull persiaml/persia-inference:latest
```

### Python libary installation
*install from pip command directly*
```bash
pip3 install persia-core persia
```

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
*install from rust cargo package manager*
```bash
cargo install persia-middleware persia-server
```

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