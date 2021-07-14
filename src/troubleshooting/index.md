# Troubleshooting in PersiaML

Debugging in distributed training system is complexity. So there is a general way for users to debug problem when they meet troubles.

## Debug in PersiaML
PersiaML consists of serval components that make debugging uneasy.The first step is build the dependent library according to persia-dev docker image. And then replace the origin perisa runtime by mount the third party library folder into the docker container.

### Debug PersiaML-sever and PersiaML-middleware

Download the persia-dev image and build the PersiaML-server and PersiaML-core as the third party library.
```bash
docker pull persiaml/persia-ci:master # download the persia-dev docker image
docker run -it --rm -v $(realpath PersiaML-server):/workspace/ \
    -e HTTP_PROXY=$http_proxy -e HTTPS_PROXY=$https_proxy \
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
`launch_middleware.sh` search the `middleware` and `server` binary articraft from `THIRD_PARTY_PATH` in high priority after setting `THIRD_PARTY_PATH`, due to `ENV` config `THIRD_PARTY_PATH` as high priority part of `PATH` in runtime container.

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
        replicas: \${MIDDLEWARE_REPLICA}
        
    volumes:
        - \${CUR_DIR}:/workspace
        - \${THIRD_PARTY}:/third_party

emb_server:
    env_file:
        - .env
    environment:
        THIRD_PARTY_PATH: /third_party
        REPLICA_INDEX: "{{.Task.Slot}}"
    image: persiaml/persia-gpu-runtime:latest
    command: bash -c "/workspace/launch_server.sh"
    deploy:
        endpoint_mode: dnsrr
        replicas: \${SERVER_REPLICA}
    volumes:
        - \${CUR_DIR}:/workspace
        - \${THIRD_PARTY}:/third_party

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
        - \${CUR_DIR}:/workspace
        - \${DATA_DIR}:/data
        - \${PERSIA_DIR}:/persia
    deploy:
        endpoint_mode: dnsrr
        replicas: \${COMPOSE_REPLICA}

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
        - \${CUR_DIR}:/workspace
        - \${THIRD_PARTY}:/third_party
        - \${PERSIA_DIR}:/persia
    deploy:
        endpoint_mode: dnsrr
        replicas: \${TRAINER_REPLICA}
```

## QA
### Q1: DDP launch failed
Pytorch ddp launch failed almost reason due to the ignorance of `NCCL` or `cuda` setting. There are some points that need notice when launch the trainer service in docker-compose
- set NCCL_SOCKET_IFNAME, CUDA_VISIBLE_DEVICES properly
- ensure master node launch in adavance than the other trainer node to avoid the hangout of `torch.distributed.init_process_group`

### Q2: Persia Core ImportError
Based on conditional compilation the `persia-core` can import without cuda environment if there not pass the rust args `--features cuda` in build phase. Once exists the `ImportError` of `persia_core.backward` or `persia_core.forward` from `persia-core`, make sure the `persia-core` build with `cuda` feature