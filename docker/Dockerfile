FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

#This Dockerfile will attempt to build this repository with GPU support.
#NVIDIA-docker is required to run with GPU support enabled.
#To run easily use ./just run and ./just build to build easily

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libpng-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libhdf5-dev \
    git \
    wget \
    curl \
    libboost-all-dev \
    libopencv-dev \
    libtbb-dev \
    cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#TODO get this to automatically match the number of processors via just build.
#The numbers of processors to use while building
ARG PROC_NUM=8
ENV LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/stubs CUDA_PATH=/usr/local/cuda NPY_NUM_BUILD_JOBS=$PROC_NUM

# install chainer
RUN pip3 install -U setuptools && \
  pip3 install -U pip && \
   pip3 install \
    chainer  \
    Cython \
    h5py \
    lmdb \
    numpy==1.10.1 \
    tqdm

#Install GOSU
#GOSU is a very useful utility for when you mount directories as read and wirte with the same user than ran the program
ENV USER_ID=1 GROUP_ID=1

ARG GOSU_VERSION=1.9
RUN curl -Lo /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)" && \
    curl -Lo /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture).asc" && \
    export GNUPGHOME="$(mktemp -d)" && \
    gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 && \
    gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

#Installl lib booost
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev libopencv-dev libtbb-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
#Dump everything into opt because why not?!
WORKDIR /opt/

#Clone the repository for SSAI-CNN
RUN git clone https://github.com/mitmul/ssai-cnn.git

#Upgrades CMake to 3.2
RUN curl -sSL https://cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz | tar -xzC /opt

#Set some buildvariables
ENV PATH=/opt/cmake-3.5.2-Linux-x86_64/bin/:$PATH SSAI_HOME=/opt/ssai-cnn

#Install OpenCV
RUN wget https://github.com/Itseez/opencv/archive/3.0.0.zip && \
    unzip 3.0.0.zip && rm -rf 3.0.0.zip && \
    cd opencv-3.0.0 && mkdir build && cd build && \
    cmake \
    -DBUILD_DOCS=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_JPEG=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PACKAGE=ON \
    -DBUILD_PERF_TESTS=ON \
    -DBUILD_PNG=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TBB=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_TIFF=ON \
    -DBUILD_WITH_DEBUG_INFO=OFF \
    -DBUILD_ZLIB=OFF \
    -DENABLE_FAST_MATH=ON \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DBUILD_EXAMPLES=OFF \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.4 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.4m.so \
    -DENABLE_AVX=ON \
    -DENABLE_SSE=ON \
    -DENABLE_SSE2=ON \
    -DENABLE_SSE3=ON \
    -DENABLE_SSE41=ON \
    -DENABLE_SSE42=ON \
    -DENABLE_SSSE3=ON \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF \
    -DWITH_OPENCL_SVM=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_CUBLAS=OFF \
    -DWITH_CUDA=ON \
    -DCUDA_GENERATION=Auto \
    -DWITH_CUFFT=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_TBB=ON \
    -Wno-dev ../ && \
    make -j$PROC_NUM install

#Install Boost.NumPy
RUN git clone https://github.com/ndarray/Boost.NumPy.git && \
    cd Boost.NumPy && mkdir build && cd build && \
    ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py34.so /usr/lib/x86_64-linux-gnu/libboost_python3.so && \
    cmake \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.4 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.4m.so \
    ../ && \
    make install -j $PROC_NUM

#From build.sh
RUN cd ssai-cnn/utils && \
    cmake \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.4 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.4m.so \
    -DOpenCV_DIR=/usr/local/share/OpenCV \
    -Wno-dev \
    . && make

#Fixes the permissions
RUN chmod a+x /opt/ssai-cnn/shells/* /opt/ssai-cnn/scripts/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

#Does some necessary setup if you want to start it without gosu.
RUN echo "PYTHONPATH=/opt/ssai-cnn/scripts:/opt/ssai-cnn/utils/:/usr/local/lib64/:/opt/Boost.NumPy/build/lib/:/usr/local/lib/python3.4/dist-packages/" >> ~/.bashrc && \
    echo "ln -s /dev/null /dev/raw1394" >> ~/.bashrc && \
    echo "chmod 777 /usr/local/lib64/libboost_numpy.so && ln -s /usr/local/lib64/libboost_numpy.so /usr/local/lib/libboost_numpy.so" >> ~/.bashrc

WORKDIR /opt/ssai-cnn

#Black magic needed to get this working. (Libboost installs to wrong directory. Probably could fix this with Makefile)
RUN chmod 777 /usr/local/lib64/libboost_numpy.so && ln -s /usr/local/lib64/libboost_numpy.so /usr/local/lib/libboost_numpy.so

ADD docker_entrypoint.sh /

ENTRYPOINT ["/docker_entrypoint.sh"]
