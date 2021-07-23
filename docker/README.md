# Dockerfile for lanegcn image

This is dockerfile for lanegcn experiments environment, the base image is `horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1`

## How to Build

```shell
git clone git@github.com:argoai/argoverse-api.git
docker build -t zhaoyi/lanegcn:v3 .
```