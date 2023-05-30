FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

EXPOSE 8080

ADD app/* opt/app/
ADD params.py predict.py opt/
ADD config/predict.yml opt/config/predict.yml
ADD transforms/transform.py opt/transforms/transform.py

RUN apt-get update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.8 python3-pip

WORKDIR /opt

RUN pip3 install -r app/requirements.txt

ENTRYPOINT python3 app/app.py
