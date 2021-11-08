Run Persia on Google Cloud
===

We assume that you already have a k8s cluster on Google Cloud, the following are the steps to deploy Persia to the k8s cluster.

1. Install NATS operator

[NATS Operator](https://github.com/nats-io/nats-operator) manages NATS clusters which is a dependency of Persia. You can install NATS operator with following command.

```bash
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/00-prereqs.yaml
$ kubectl apply -f https://github.com/nats-io/nats-operator/releases/latest/download/10-deployment.yaml
```

2. Installing NVIDIA GPU device drivers

Also see Google Could [docs](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers).

```bash
$ kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml
```

3. Run Persia

```bash
kubectl apply -f train.persia.yml
```

where `train.persia.yml` contains

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    service: trainer
  name: trainer
spec:
  serviceName: trainer
  replicas: 8
  selector:
    matchLabels:
      service: trainer
  template:
    metadata:
      labels:
        service: trainer
    spec:
      containers:
        - args:
            - /workspace/launch.sh
            - trainer
          image: persiaml/persia-cuda-runtime:latest
          resources:
            requests:
              memory: "600Gi"
              cpu: "80"
              nvidia.com/gpu: 8
            limits:
              memory: "600Gi"
              cpu: "80"
              nvidia.com/gpu: 8
          name: trainer
          env:
            - name: REPLICA_SIZE
              value: "8"
            - name: NPROC_PER_NODE
              value: "8"
            - name: PERSIA_NATS_IP
              value: nats://persia-nats-service:4222
          imagePullPolicy: Always

status:
  replicas: 8

---

apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    service: data-compose
  name: data-compose
spec:
  serviceName: data-compose
  replicas: 240
  selector:
    matchLabels:
      service: data-compose
  template:
    metadata:
      labels:
        service: data-compose
    spec:
      containers:
        - args:
            - /workspace/launch.sh
            - datacompose
          image: persiaml/persia-cuda-runtime:latest
          resources:
            requests:
              memory: "12Gi"
              cpu: "1"
            limits:
              memory: "12Gi"
              cpu: "1"
          name: data-compose
          env:
            - name: REPLICA_SIZE
              value: "240"
            - name: PERSIA_NATS_IP
              value: nats://persia-nats-service:4222
          imagePullPolicy: Always
status:
  replicas: 240

---

apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    service: middleware
  name: middleware
spec:
  serviceName: middleware
  replicas: 100
  selector:
    matchLabels:
      service: middleware
  template:
    metadata:
      labels:
        service: middleware
    spec:
      containers:
        - args:
            - /workspace/launch.sh
            - middleware
          image: persiaml/persia-cuda-runtime:latest
          resources:
            requests:
              memory: "100Gi"
              cpu: "28"
            limits:
              memory: "100Gi"
              cpu: "28"
          name: middleware
          env:
            - name: REPLICA_SIZE
              value: "100"
            - name: PERSIA_NATS_IP
              value: nats://persia-nats-service:4222
          imagePullPolicy: Always
status:
  replicas: 100

---

apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    service: emb-server
  name: emb-server
spec:
  serviceName: emb-server
  replicas: 120
  selector:
    matchLabels:
      service: emb-server
  template:
    metadata:
      labels:
        service: emb-server
    spec:
      containers:
        - args:
            - /workspace/launch.sh
            - embserver
          image: persiaml/persia-cuda-runtime:latest
          resources:
            requests:
              memory: "2500Gi"
              cpu: "80"
            limits:
              memory: "2500Gi"
              cpu: "80"
          name: emb-server
          env:
            - name: REPLICA_SIZE
              value: "120"
            - name: PERSIA_NATS_IP
              value: nats://persia-nats-service:4222
          imagePullPolicy: Always
status:
  replicas: 120

---

apiVersion: "nats.io/v1alpha2"
kind: "NatsCluster"
metadata:
  name: "persia-nats-service"
spec:
  size: 60
  natsConfig:
    maxPayload: 52428800

  resources:
    requests:
      memory: "32Gi"
      cpu: "4"
    limits:
      memory: "32Gi"
      cpu: "4" 
```

and you need to have `launch.sh` avaliable in the container at `/workspace/launch.sh` containing

```bash
#!/bin/bash
set -x
export REPLICA_INDEX=${HOSTNAME##*-}

export PERSIA_MODEL_CONFIG=/workspace/persia-exp/criteo_config.yml
export PERSIA_EMBEDDING_CONFIG=/workspace/persia-exp/criteo_embedding_config.yml
export PERSIA_GLOBAL_CONFIG=/workspace/persia-exp/global_config.yml
export LOG_DIR=/workspace/logs/

export NCCL_DEBUG=INFO

export PERSIA_PORT=23333

mkdir -p $LOG_DIR

export LOG_LEVEL=info
export RUST_BACKTRACE-full

export PERSIA_NATS_IP=nats://persia-nats-service:4222
export PERSIA_METRICS_GATEWAY_ADDR=metrics-gateway:9091

if [ $1 == "trainer" ];then

    export NODE_RANK=${REPLICA_INDEX}
    export WORLD_SIZE=$(($NPROC_PER_NODE * $REPLICA_SIZE))

    /opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$REPLICA_SIZE --node_rank=$NODE_RANK /workspace/persia-exp/train.py

fi

if [ $1 == "datacompose" ];then
    /opt/conda/bin/python3 /workspace/persia-exp/data_compose.py
fi

if [ $1 == "middleware" ];then
    /workspace/persia-exp/runtime/$CPU_TYPE/persia-middleware-server ---port $PERSIA_PORT --global-config $PERSIA_GLOBAL_CONFIG \
        --embedding-config $PERSIA_EMBEDDING_CONFIG --replica-index $REPLICA_INDEX --replica-size $REPLICA_SIZE
fi

if [ $1 == "embserver" ];then
    /workspace/persia-exp/runtime/$CPU_TYPE/persia-embedding-server ---port $PERSIA_PORT --global-config $PERSIA_GLOBAL_CONFIG \
        --embedding-config $PERSIA_EMBEDDING_CONFIG --replica-index $REPLICA_INDEX --replica-size $REPLICA_SIZE
fi


exit 0
```

NOTE: The scripts and configuration files required for training are not in the docker image, they need to be accessible by the container through shared storage.
