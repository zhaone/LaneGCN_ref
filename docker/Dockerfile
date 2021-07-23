FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

# apt
COPY source.list /etc/apt/sources.list
RUN apt-get update \
    && apt-get install -y curl git sudo\
    && apt-get install -y zsh tmux \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get install -y openssh-server \
    && rm -rf /var/lib/apt/lists/*

# add lgcn user
RUN useradd --create-home --no-log-init --shell /bin/bash lgcn \
    && adduser lgcn sudo \
    && echo 'lgcn:mypassword' | chpasswd

# copy argover-api
COPY argoverse-api /home/lgcn/argoverse-api
RUN chown -R lgcn:lgcn /home/lgcn/argoverse-api

# create workspace
RUN mkdir /workspace \
    && chown -R lgcn:lgcn /workspace

USER lgcn

# install all the python package
COPY pip.conf /home/lgcn/.pip/pip.conf
COPY requirement.txt /home/lgcn/requirement.txt
RUN pip install -r /home/lgcn/requirement.txt \
    && rm -rf .cache/pip/*

# install argo api
RUN pip install -e /home/lgcn/argoverse-api

WORKDIR /workspace

CMD ["/bin/bash"]
