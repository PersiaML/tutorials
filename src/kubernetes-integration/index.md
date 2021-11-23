Run Persia on Kubernetes
===

The Persia Operator is a Kubernetes [custom resource definitions](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/). You can define your distributed persia task by an operator file. We have learned the structure of an operator definition file in the [Customize a PERSIA Job](../customize-a-persia-job/index.md) section. In following section, we will introduce more details about running PERSIA on a k8s cluster.

## Persia Job Name

For PERSIA operator, a job name is a unique identifier, it is important to keep job name different between PERSIA jobs.

```yaml
apiVersion: persia.com/v1
kind: PersiaJob
metadata:
  name: you-job-name
  namespace: default
...
```

## Configuring Environment Variables

You can set environment variables for all pods or for a PERSIA module. As the following example, setting of `PERSIA_NATS_IP` take effect for all pods in this job, while the `CUBLAS_WORKSPACE_CONFIG` only set on NN workers.

```yaml
...
spec:
  globalConfigPath: /workspace/global_config.yml
  embeddingConfigPath: /workspace/embedding_config.yml
  ...
  env:
    - name: GLOBAL_ENV
      value: "I will take effect on all pods"

  nnWorker:
    replicas: 1
    nprocPerNode: 1
    ...
    env:
      - name: MODULE_RNV
        value: "I will take effect on NN worker pods only"
...
```

## Configuring Resources

When you specify a PERSIA module, you can optionally specify how much of each resource a container of this module needs. The most common resources to specify are CPU, memory and GPUs. Refer to [k8s doc](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/).

```yaml
...
spec:
  globalConfigPath: /workspace/global_config.yml
  embeddingConfigPath: /workspace/embedding_config.yml
  ...

  nnWorker:
    replicas: 1
    nprocPerNode: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "12"
        nvidia.com/gpu: "1"
...
```

## Mount Volumes

Kubernetes supports many types of volumes (refer to [k8s doc](https://kubernetes.io/docs/concepts/storage/volumes/)), you can mount these volumes to your containers in a PERSIA job. Here is an example.

```yaml
...
spec:
  globalConfigPath: /workspace/global_config.yml
  embeddingConfigPath: /workspace/embedding_config.yml
  ...
  volumes:
    - name: data
      hostPath:
        path: /nfs/general/data/
        type: Directory

  nnWorker:
    replicas: 1
    nprocPerNode: 1
    volumeMounts:
      - name: data
        mountPath: /data/
        read_only: true
...
```

## Configuring PERSIA image

PERSIA operator support to specify a image for modules, here is an example.

```yaml
...
spec:
  globalConfigPath: /workspace/global_config.yml
  embeddingConfigPath: /workspace/embedding_config.yml
  ...

  nnWorker:
    replicas: 1
    nprocPerNode: 1
    image: persiaml/persia-cuda-runtime:dev
...
```

## Nats Operator

While starting a PERSIA training task, we usually need to start a nats service, which can be achieved through its [operator](https://github.com/nats-io/nats-operator). PERSIA transmits ID type feature through nats, so you need to ensure that its `maxPayload` is large enough.

```yaml
apiVersion: "nats.io/v1alpha2"
kind: "NatsCluster"
metadata:
  name: "persia-nats-service"
spec:
  size: 1
  natsConfig:
    maxPayload: 52428800
  resources:
    limits:
      memory: "8Gi"
      cpu: "2" 
```