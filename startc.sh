docker run \
    --runtime=nvidia \
    --name lanegcn \
    --rm \
    --shm-size="20g" \
    -d \
    -p 0.0.0.0:16006:6006 \
    -it \
    zhaone/lanegcn:v1

