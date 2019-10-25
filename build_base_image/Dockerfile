FROM continuumio/miniconda3

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN conda install pytorch-cpu torchvision-cpu -c pytorch
