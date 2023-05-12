# Installer
# FROM ubuntu:20.04
FROM tensorflow/tensorflow:latest-gpu
LABEL Author="Daniel Duclos-Cavalcanti"
LABEL Email="daniel.duclos-cavalcanti@gmail.com"
LABEL Maintainer="Daniel Duclos-Cavalcanti"

ARG USER_NAME="coral-docker"
ARG DEBIAN_FRONTEND="noninteractive"

RUN set -ex

# Base System
RUN apt-get update \
    && apt-get install -y \
    sudo build-essential locales wget curl gnupg curl zip unzip tar openjdk-11-jdk\
    python3 python3-pip \
    git vim nano \
    usbutils udev lshw \
    software-properties-common \
    && apt-get clean \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen && update-locale 

# CoralDev Apt Paackages
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key add -

RUN apt-get update
RUN apt-get install -y libedgetpu1-std edgetpu-compiler

# Wireshark/Tshark
RUN add-apt-repository -y ppa:wireshark-dev/stable
RUN apt-get install -y tshark

# TensorRT
RUN apt-get install -y tensorrt
# RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer.so.7
# RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7
RUN find / -name 'libnvinfer_plugin.so.8' -exec sh -c 'ln -sf "$0" "$(dirname "$0")/libnvinfer_plugin.so.7"' {} \;
RUN find / -name 'libnvinfer.so.8' -exec sh -c 'ln -sf "$0" "$(dirname "$0")/libnvinfer.so.7"' {} \;

# Python Libraries
#RUN pip3 install git+https://github.com/siliconlabs/mltk.git
RUN pip3 install silabs-mltk[full] --upgrade
RUN pip3 install pyshark pyserial pyftdi argparse metrics visualization
RUN pip3 install matplotlib scipy numpy pandas
# RUN pip3 install tensorflow
RUN pip3 install --upgrade pip
RUN apt-get install -y python3-ipdb python3-tabulate

# Pycoral
RUN apt-get install -y python3-pycoral
# RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# CUDA
RUN apt-get install -y nvidia-cuda-dev  nvidia-cuda-toolkit 
# RUN apt-get install -y libcudnn8

# CUBLAS
# RUN apt-get install -y libcublas10

WORKDIR /home/

# see the original bash script for detailed information
# https://github.com/terryky/tflite_gles_app/blob/master/tools/scripts/tf2.9/build_libtflite_r2.9.sh

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.sh
RUN chmod 755 cmake-3.22.1-linux-x86_64.sh
RUN ./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local

# bazel
RUN apt-get install -y unzip
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
RUN chmod 755 bazel-5.0.0-installer-linux-x86_64.sh
RUN ./bazel-5.0.0-installer-linux-x86_64.sh

# flatbuffers
RUN git clone https://github.com/Microsoft/vcpkg.git
RUN cd vcpkg
#RUN ./bootstrap-vcpkg.sh
#RUN ./vcpkg integrate install
#RUN ./vcpkg install flatbuffers

# NVIDIA Drivers:
RUN sudo add-apt-repository ppa:graphics-drivers/ppa
RUN sudo apt update
RUN sudo apt install nvidia-driver-530
RUN sudo apt install -y clinfo

# script to build delegates
RUN apt-get install -y libegl1-mesa-dev

# Tensorflow + GPU Delegate
COPY scripts/install_tf.sh /home/install_tf.sh
RUN chmod +x /home/install_tf.sh
RUN /home/install_tf.sh

# Ports
# EXPOSE 8080

# ENV Variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV MODE 0
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Work
RUN mkdir -p /home/tensorDSE/benchmarking
RUN mkdir /home/tensorDSE/deployment
WORKDIR /home/tensorDSE
ENTRYPOINT [ "/home/tensorDSE/benchmarking/scripts/entry.sh" ]
