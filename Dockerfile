FROM nvidia/cuda:8.0-cudnn7-runtime-ubuntu16.04

COPY ./ /mnt/
WORKDIR /mnt/

#build the env and python packages	
RUN apt-get upgrade -y && apt-get dist-upgrade

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python-pip \
	curl \
	wget \	
	zip	\
	unzip \
	bzip2 \
	tar 

RUN pip install --upgrade pip

RUN pip install \
    h5py \
    keras==2.0.5 \
    numpy \
    scipy \
    scikit-learn \
    
##Install Tensorflow
## uncomment one or the other of the next lines depending on whether
## you want GPU
#RUN pip install --upgrade tensorflow==1.3
RUN pip install --upgrade tensorflow-gpu==1.3

ENV LD_LIBRARY_PATH=/cudadir/lib64
