# Troubleshooting

Debugging a distributed training system can be difficult. We provide some general suggestions in this page.

## Investigate debug log

...

## Investigate Grafana metrics

...

## Print intermediate results

<!-- this is not debugging.. 
Download the persia-dev and build the PersiaML-server and PersiaML-core as third party library.
```bash
docker pull persiaml/persia-ci:master # download the persia-dev docker image
docker run -it --rm -v $(realpath PersiaML-server):/workspace/ \
    -e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
    --network=host -v $(realpath examples/DLRM/third_party):/build/ \
    persia-ci:master bash -c "cd /workspace && \
    cargo build --release --package persia-core --features cuda --target-dir /build && \
    cargo build --release --package persia-embedding-sharded-server --target-dir /build && \
    cd /build/release && mv libpersia_core.so /build/persia_core.so && \
    mv /build/release/persia-embedding-sharded-middleware /build/ && \
    mv /build/release/persia-embedding-sharded-server /build/ && \
    rm -rf /build/release"
```

Mount the third party library into the container after build the PersiaML-server and PersiaML-core. Set the `THIRD_PARTY_PATH` to the path `/third_party`. 
`launch_middleware.sh` search the `middleware` or `server` binary articraft from `THIRD_PARTY_PATH` in high priority after setting `THIRD_PARTY_PATH`.Due to `ENV` config `THIRD_PARTY_PATH` as high priority part of `PATH` in runtime container.

*middleware and server config in docker-compose.yaml*
```yaml
middleware:
    env_file:
        - .env
    environment:
        THIRD_PARTY_PATH: /third_party
        REPLICA_INDEX: "{{.Task.Slot}}" 
    depends_on:
        - server
    image: persiaml/persia-gpu-runtime:latest
    command: bash -c "/workspace/launch_middleware.sh"
    deploy:
        endpoint_mode: dnsrr
        replicas: ${MIDDLEWARE_REPLICA}
    volumes:
        - ${CUR_DIR}:/workspace
        - ${THIRD_PARTY}:/third_party

server:
    env_file:
        - .env
    environment:
        THIRD_PARTY_PATH: /third_party
        REPLICA_INDEX: "{{.Task.Slot}}"
    image: persiaml/persia-gpu-runtime:latest
    command: bash -c "/workspace/launch_server.sh"
    deploy:
        endpoint_mode: dnsrr
        replicas: ${SERVER_REPLICA}
    volumes:
        - ${CUR_DIR}:/workspace
        - ${THIRD_PARTY}:/third_party

```

### Debug the PersiaML Python and PersiaML-core
PersiaML Python code can debug by mount the PersiaML python code path into container `PYTHONPATH`. Once there exist the requirement to debug the `persia-core` library, build the persia-core rust code as the step that debug in persia-server.But the difference between the `persia-server` is that `/third_party` path should add into `PYTHONPATH` and the origin `persia-core` should be uninstall by invoke `pip3 uninstall persia-core` as the part of `docker-compose` launch command.

*data compose and trainer config in docker-compose.yaml*
```yaml
data_compose:
    env_file:
        - .env
    environment:
        REPLICA_INDEX: "{{.Task.Slot}}"
        PYTHONPATH: /persia:/third_party
    depends_on:
        - trainer
        - middleware
        - persia_nats_service
    image: persiaml/persia-gpu-runtime:latest
    command: /workspace/launch_compose.sh /workspace/data_compose.py
    volumes:
        - ${CUR_DIR}:/workspace
        - ${DATA_DIR}:/data
        - ${PERSIA_DIR}:/persia
    deploy:
        endpoint_mode: dnsrr
        replicas: ${COMPOSE_REPLICA}

trainer:
    env_file:
        - .env
    environment:
        REPLICA_INDEX: "{{.Task.Slot}}"
        NCCL_SOCKET_IFNAME: eth0
        GPU_NUM: 1
        PYTHONPATH: /persia:/third_party
    image: persiaml/persia-gpu-runtime:latest
    command:  /workspace/launch_persia.sh /workspace/train.py
    volumes:
        - ${CUR_DIR}:/workspace
        - ${THIRD_PARTY}:/third_party
        - ${PERSIA_DIR}:/persia
    deploy:
        endpoint_mode: dnsrr
        replicas: ${TRAINER_REPLICA}
```

---> 

## FAQ

### DDP launch failed

Most PyTorch DDP launch failed error comes from incorrect [NCCL](https://developer.nvidia.com/nccl) configuration. We suggest checking the followings
- Set `NCCL_SOCKET_IFNAME`to your physical network device name, and set `CUDA_VISIBLE_DEVICES` properly.
- Launch the training job according to the instructions in [PyTorch DDP documentation](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py).

### `persia-core` ImportError

There are two versions of `persia-core`, a CPU version that is used for inference, and a GPU (CUDA) version used for GPU training. Make sure you installed the correct version.
