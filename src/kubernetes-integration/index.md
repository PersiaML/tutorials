Run Persia on Kubernetes
===

We assume that you already have a k8s cluster with [NATS Operator](https://github.com/nats-io/nats-operator#installing) installed, you can use Persia Operator or Schedule Server to deploy persia to k8s cluster. Command `persia_k8s_uitls` can be installed by `pip install persia`.

## Persia Operator

The Persia Operator is a Kubernetes [custom resource definitions](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/).

1. Apply Custom Resource Definitions

You can generate `jobs.persia.com.yaml` file and apply it to k8s cluster by following command.

```bash
$ persia_k8s_uitls gencrd
$ sudo kubectl apply -f jobs.persia.com.yaml
```

2. Run Persia Operator

```bash
$ export KUBECONFIG=/your/kubeconfig/path/k8s.yaml
$ persia_k8s_uitls operator
```

3. Deploy Persia Job

You can define a Persia Job by following yaml.

```yaml
apiVersion: persia.com/v1
kind: PersiaJob
metadata:
  name: adult-income  # persia job name, need to be globally unique
  namespace: default  # k8s namespace to deploy to this job
spec:
  # the following path are the path inside the container
  globalConfigPath: /workspace/config/global_config_train.yml
  embeddingConfigPath: /workspace/config/embedding_config.yml
  trainerPyEntryPath: /workspace/train.py
  dataLoaderPyEntryPath: /workspace/data_compose.py
  # k8s volumes, see https://kubernetes.io/docs/concepts/storage/volumes/
  volumes:
    - name: data
      hostPath:
        path: /nfs/general/data/adult_income/
        type: Directory
    - name: workspace
      hostPath:
        path: /nfs/general/PersiaML/e2e/adult_income/
        type: Directory
  # global env, it will apply to all containers.
  env:
    - name: EVAL_CHECKPOINT_DIR
      value: /workspace/eval_checkpoint/
    - name: INFER_CHECKPOINT_DIR
      value: /workspace/infer_checkpoint/
    - name: RESULT_FILE_PATH
      value: /workspace/result.json
    - name: PERSIA_NATS_IP
      value: nats://persia-nats-service:4222

  embeddingServer:
    replicas: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "4"
    volumeMounts:
      - name: workspace
        mountPath: /workspace/

  middlewareServer:
    replicas: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "4"
    volumeMounts:
      - name: workspace
        mountPath: /workspace/

  trainer:
    replicas: 1
    nprocPerNode: 1
    resources:
      limits:
        memory: "24Gi"
        cpu: "12"
        nvidia.com/gpu: "1"
    volumeMounts:
      - name: workspace
        mountPath: /workspace/
      - name: data
        mountPath: /data/
        read_only: true
    env:
      - name: CUBLAS_WORKSPACE_CONFIG
        value: :4096:8

  dataloader:
    replicas: 1
    resources:
      limits:
        memory: "8Gi"
        cpu: "1"
    volumeMounts:
      - name: workspace
        mountPath: /workspace/
      - name: data
        mountPath: /data/
        read_only: true

---

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

Then You can apply it to k8s cluster by following command.

```bash
$ sudo kubectl apply -f k8s.train.yaml
```


## Persia Schedule Server

You can also deploy Persia to k8s cluster through Persia Schedule Server. It is a http server with several k8s related apis, you can deploy Persia without knowing k8s.

1. Run Schedule Server

```bash
$ export KUBECONFIG=/your/kubeconfig/path/k8s.yaml
$ persia_k8s_uitls server
```

