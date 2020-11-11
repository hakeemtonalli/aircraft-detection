FROM ubuntu:18.04

# openBLAS config for cpu mode
ENV LANG C.UTF-8
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y vim python3.8 \
    python3-pip python3.8-dev gcc \
    libopenblas-dev ffmpeg libsm6 libxext6  -y \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip 


RUN apt-get install -y git wget curl zip unzip

WORKDIR /app 

RUN cd /app 

COPY . /app/aircraft-detection

WORKDIR /app/aircraft-detection 

# install dependencies from root 
RUN pip3 install pipenv \
    && pipenv install --deploy --ignore-pipfile 

# set up data and mlflow from root
RUN sh scripts/get_imagery.sh \
    && pipenv run sh scripts/setup_mlflow.sh 
    

ENTRYPOINT ["bash"]

LABEL maintainer="htfrank@cpp.edu"