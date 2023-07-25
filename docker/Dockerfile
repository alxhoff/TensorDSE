# Installer
# FROM ubuntu:20.04
FROM tensorflow/tensorflow:latest-gpu
LABEL Author="Daniel Duclos-Cavalcanti & Alex Hoffman"
LABEL Email="daniel.duclos-cavalcanti@gmail.com & alxhoff@gmail.com"
LABEL Maintainer="Alex Hoffman"

ARG WLSACCESSID
ARG WLSSECRET
ARG LICENSEID

SHELL ["/bin/bash", "-l", "-c"] 

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
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \ 
    && locale-gen && update-locale

# CoralDev APT Paackages
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key add - && \
apt-get update && \
apt-get install -y libedgetpu1-std edgetpu-compiler

# Wireshark/Tshark
RUN add-apt-repository -y ppa:wireshark-dev/stable && \
apt-get install -y tshark

# TensorRT
RUN apt-get install -y tensorrt && \
find / -name 'libnvinfer_plugin.so.8' -exec sh -c 'ln -sf "$0" "$(dirname "$0")/libnvinfer_plugin.so.7"' {} \; && \
find / -name 'libnvinfer.so.8' -exec sh -c 'ln -sf "$0" "$(dirname "$0")/libnvinfer.so.7"' {} \;

# Python Libraries
RUN pip3 install --upgrade pip && \
pip3 install silabs-mltk[full] --upgrade && \
pip3 install pyshark pyserial pyftdi argparse metrics visualization matplotlib scipy numpy pandas && \
apt-get install -y python3-ipdb python3-tabulate python3-pycoral

# CUDA
RUN apt-get install -y nvidia-cuda-dev  nvidia-cuda-toolkit

WORKDIR /home/

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh && \ 
wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.sh && \
chmod 755 cmake-3.22.1-linux-x86_64.sh && \
./cmake-3.22.1-linux-x86_64.sh --skip-license --prefix=/usr/local

# bazel
RUN apt-get install -y unzip && \
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh && \
chmod 755 bazel-5.0.0-installer-linux-x86_64.sh && \
./bazel-5.0.0-installer-linux-x86_64.sh

# NVIDIA Drivers:
RUN sudo add-apt-repository ppa:graphics-drivers/ppa && \
sudo apt update && sudo apt install -y clinfo libegl1-mesa-dev

# Flatbuffers
COPY scripts/install_flatbuffers.sh /home/install_flatbuffers.sh
RUN chmod +x /home/install_flatbuffers.sh && /home/install_flatbuffers.sh

# Sources
COPY scripts/pull_sources.sh /home/pull_sources.sh
RUN chmod +x /home/pull_sources.sh && /home/pull_sources.sh

# TF Deps
COPY scripts/download_tf_deps.sh /home/download_tf_deps.sh
RUN chmod +x /home/download_tf_deps.sh && /home/download_tf_deps.sh

# Tensorflow + GPU Delegate
COPY scripts/install_tf.sh /home/install_tf.sh
RUN chmod +x /home/install_tf.sh && /home/install_tf.sh

# Patches
COPY patches /home/patches
RUN chmod +x /home/patches/patch.sh && /home/patches/patch.sh

# Build and install backend
COPY scripts/build_backend.sh /home/build_backend.sh
RUN chmod +x /home/build_backend.sh && /home/build_backend.sh

# Gradle for DSE
RUN curl -s "https://get.sdkman.io" | bash && \
chmod +x /root/.sdkman/bin/sdkman-init.sh && \
source /root/.sdkman/bin/sdkman-init.sh && sdk install gradle 6.9.4 && \
ln -sf /root/.sdkman/candidates/gradle/6.9.4/bin/gradle /usr/bin/gradle6

# Gurobi
WORKDIR /opt
RUN echo $WLSACCESSID
RUN echo $WLSSECRET
RUN echo $LICENSEID
RUN wget https://packages.gurobi.com/9.5/gurobi9.5.2_linux64.tar.gz && \
tar -xzf gurobi9.5.2_linux64.tar.gz && \
export GUROBI_HOME=/opt/gurobi952/linux64 && \
echo "export GUROBI_HOME=/opt/gurobi952/linux64" >> /root/.bashrc && \
export PATH=$GUROBI_HOME/bin:$PATH && \
echo "export PATH=$GUROBI_HOME/bin:$PATH" >> /root/.bashrc && \
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH && \
echo "export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH" >> /root/.bashrc

RUN echo "WLSACCESSID=$WLSACCESSID" > /opt/gurobi952/gurobi.lic && \
echo "WLSSECRET=$WLSSECRET" >> /opt/gurobi952/gurobi.lic && \
echo "LICENSEID=$LICENSEID" >> /opt/gurobi952/gurobi.lic

# ENV Variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV MODE 0
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Work
RUN mkdir -p /home/tensorDSE/benchmarking && \
mkdir -p /home/tensorDSE/deployment
WORKDIR /home/tensorDSE
ENTRYPOINT ["bash", "-i", "/home/tensorDSE/docker/scripts/entry.sh" ]
