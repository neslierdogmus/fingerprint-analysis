FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# CUDA_VERSION: 10.1.243
# CUDNN_VERSION: 7.6.5.32

LABEL maintainer="nesli <neslierdogmus@iyte.edu.tr>"

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update --fix-missing && \
    apt-get install -y ffmpeg libsm6 libxext6 wget bzip2 curl git locate && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV USER_NAME=csigb
ENV UID=1000
ENV GID=100
ENV HOME=/home/$USER_NAME

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER_NAME

COPY ./environment.yml /tmp/
RUN chown $UID:$GID /tmp/environment.yml

COPY ./entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

USER $USER_NAME

# install miniconda
ENV MINICONDA_VERSION=py37_4.8.3
ENV CONDA_DIR=$HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash

# create a project directory inside user home
ENV PROJECT_DIR=$HOME/foe
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

# add src folder to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:$HOME/foe/src"

# build the conda environment
ENV ENV_PREFIX=$PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
    conda env create --prefix $ENV_PREFIX --file /tmp/environment.yml --force && \
    conda clean --all --yes

# use an entrypoint script to insure conda environment is properly activated at runtime
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]

# default command
CMD [ "/bin/bash" ]
