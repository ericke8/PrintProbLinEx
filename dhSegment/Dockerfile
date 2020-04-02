FROM ubuntu:latest

RUN apt-get update && apt-get install -y \ 
    python3 \
    python3-pip \
    git \
    python3-setuptools \
    vim \
    wget \
    unzip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    freeglut3-dev \
    build-essential \
    libx11-dev \
    libxmu-dev \
    libxi-dev \
    libglu1-mesa \
    libglu1-mesa-dev

#install dhsegment dependencies
RUN pip3 install git+https://github.com/dhlab-epfl/dhSegment

#install tensorflow 1.13
RUN pip3 install tensorflow-gpu==1.13.1

#install opencv
RUN pip3 install opencv-python

#install CUDA 10.0
#system update
RUN apt-get update
RUN apt-get upgrade
#first get the PPA repository driver
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | tee /etc/apt/sources.list.d/cuda.list
#installing CUDA-10.0
RUN apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers
#setup your paths
RUN echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
RUN source ~/.bashrc
RUN ldconfig
#install cuDNN v7.5
RUN CUDNN_TAR_FILE="cudnn-10.1-linux-x64-v7.5.0.56.tgz"
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=118NrkAlBFzs0tgRgLcFzt0sSfkatllcn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=118NrkAlBFzs0tgRgLcFzt0sSfkatllcn" -O cudnn-10.1-linux-x64-v7.5.0.56.tgz && rm -rf /tmp/cookies.txt
RUN tar -xzvf ${CUDNN_TAR_FILE}
#copy the following files into the cuda toolkit directory.
RUN cp -P cuda/include/cudnn.h /usr/local/cuda-10.0/include
RUN cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/
RUN chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*

#install dhSegment
RUN git clone https://github.com/dhlab-epfl/dhSegment.git
