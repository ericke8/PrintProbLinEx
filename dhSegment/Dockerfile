FROM ubuntu:latest

RUN apt-get update && apt-get install -y \ 
    python3 \
    python3-pip \
    git \
    python3-setuptools

#install dhsegment dependencies
RUN pip3 install git+https://github.com/dhlab-epfl/dhSegment

#install tensorflow 1.13
RUN pip3 install tensorflow-gpu==1.13.1

#install dhSegment
RUN git clone https://github.com/dhlab-epfl/dhSegment.git
