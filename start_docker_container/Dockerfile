FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Keeps Python from generating .pyc files in the container 
ENV PYTHONDONTWRITEBYTECODE=1 
 
# Turns off buffering for easier container logging 
ENV PYTHONUNBUFFERED=1 

RUN apt-get update && \
  apt-get install -y git wget unzip bzip2 sudo build-essential && \
  apt-get install -y gcc g++ make && \
  apt-get install -y pkg-config && \
  apt-get install -y gnupg2 rsync && \
  rm -rf /var/lib/apt/lists/*

# Instal basic utilities
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN  pip install --upgrade pip 

COPY . /workspace

RUN pip install -r requirements.txt

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN rm -rf ~/.cache/pip

RUN apt-get update
#create a non root user to access the container 

RUN adduser -u 4444 --disabled-password --gecos “” vscode

