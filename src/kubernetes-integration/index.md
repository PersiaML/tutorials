Kubernetes Integration
===

PERSIA is integrated to Kubernetes as a `PersiaJob` [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/). You can define your distributed PERSIA task by a [CustomResourceDefinition](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/) (CRD). We have learned the basic structure of a PERSIA CRD in the [Customize a PERSIA Job](../customization/index.md#K8s-launcher) section. In this section, we will introduce more details about running PERSIA on a K8s cluster.

## PERSIA Job Name

In a PERSIA CRD, the job name is a unique identifier of the current PERSIA training task. It is important to keep job names different between different PERSIA jobs.

```yaml
apiVersion: persia.com/v1
kind: PersiaJob
metadata:
  name: you-job-name
  namespace: default
...
```

## Configuring Environment Variables

You can set environment variables for all pods or for a PERSIA module. In the following example, the environment variable `GLOBAL_ENV` is set for all pods in this job, while the `MODULE_RNV` is only set on NN workers.

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

When you specify a PERSIA module, you can optionally specify how much of each resource a container of this module needs. The most common resources to specify are CPU, memory and GPUs. Refer to [K8s doc](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) for more details.

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

## Mounting Volumes

Kubernetes supports many types of volumes (see [K8s doc](https://kubernetes.io/docs/concepts/storage/volumes/)). You can mount these volumes to your containers in a PERSIA job. Here is an example:

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

## Configuring PERSIA Image

You can also specify a docker image for a PERSIA module. Here is an example:

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

## Configuring Nats Operator

While starting a PERSIA training task, we usually need to start a nats service, which can be achieved through [nats-operator](https://github.com/nats-io/nats-operator). PERSIA transmits ID type feature through nats, so you need to ensure that its `maxPayload` is large enough. Please note that global environment variable `PERSIA_NATS_IP` should be set to `nats://your-nats-operator-name:4222`, e.g. `nats://persia-nats-service:4222` for the following example.

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
