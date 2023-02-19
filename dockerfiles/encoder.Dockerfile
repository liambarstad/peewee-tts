FROM ghcr.io/mlflow/mlflow:v2.1.1

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV REPO_PATH https://github.com/liambarstad/peewee-tts.git
ENV ENTRY train_encoder
ENV CONFIG_PATH config/speaker_recog_encoder_prod.yml
ENV GIT_PYTHON_REFRESH=quiet

RUN apt-get update && \
    apt-get install -y wget && \
    apt-get install -y libsndfile1-dev && \
    rm -rf /var/lib/apt/lists/*

# install conda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda --version

# add to local filesystem
ADD config/* opt/config/
ADD datasets/* opt/datasets/
ADD metrics/* opt/metrics/
ADD models/* opt/models/
ADD transforms/* opt/transforms/
ADD conda.yaml MLProject train_encoder.py utils.py /opt

WORKDIR opt

# add experiment ID

CMD mlflow run . -e $ENTRY \
    -P config_path=$CONFIG_PATH \
    -P save_model=true \
    --experiment-name encoder
