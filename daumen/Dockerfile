FROM ubuntu:latest
MAINTAINER "Jason Vega" "jasonvega14@yahoo.com"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	apt-utils \
	pkg-config \
	python-numpy \
	python3-numpy \
	git \
	build-essential \
	cmake \
	libboost-all-dev \
	autoconf \
	automake \
	libtool \
	curl \
	make \
	g++ \
	unzip \
	libprotobuf-dev \
	wget \
	libgtest-dev \
	libfftw3-dev \
	libsoundtouch-dev \
	libsndfile-dev \
	libatlas-base-dev \
	liblapack-dev \
	libblas-dev \
	libgoogle-glog-dev \
	protobuf-compiler \
	libcaffe-cpu-dev \
	vim

# Install opencv3
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /opencv_contrib
RUN git checkout 3.4.9
WORKDIR /opencv
RUN git checkout tags/3.4.9
WORKDIR /opencv/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules -DBUILD_LIST=core,dnn,highgui,imgproc,plot ..
RUN make -j 12 && make install
WORKDIR /

# Install eigen3
RUN git clone https://gitlab.com/libeigen/eigen.git
WORKDIR /eigen/build
RUN cmake ..
RUN make -j 12 && make install
WORKDIR /

# Install daumen
RUN git clone https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen.git
WORKDIR /daumen
RUN git checkout d0587799
WORKDIR /daumen/build
RUN cmake -DDAUMEN_AUDIO=OFF -DCAFFE_CPU_ONLY=ON ..
RUN make -j 12 && make install
RUN /sbin/ldconfig -v
WORKDIR /daumen/examples/PageSegmentation
RUN wget https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen/-/raw/master/examples/PageSegmentation/model_ignore_background.json \
	&& wget https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen/-/raw/master/examples/PageSegmentation/net_ignore_background.prototxt \
	&& wget https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen/-/raw/master/examples/PageSegmentation/solver_ignore_background.prototxt \
	&& wget https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen/-/raw/master/examples/PageSegmentation/model_post.json \
	&& wget https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/daumen/-/raw/master/examples/PageSegmentation/run_model.py
WORKDIR /
