docker run \
    --runtime=nvidia \
    --name lanegcn \
    --rm \
    --shm-size="20g" \
    -d \
    -v $1:/workspace \
    -p 0.0.0.0:16006:6006 \
    -it \
    zhaoyi/lanegcn:v3 \

