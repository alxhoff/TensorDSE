#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Update system's package list
sudo apt-get update

# Install the Newest Version of CMake
cd /home/
mkdir tools && cd tools
sudo apt install -y build-essential libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.27.2/cmake-3.27.2.tar.gz
tar -zxvf cmake-3.27.2.tar.gz
cd cmake-3.27.2
./bootstrap
make 
sudo make install
cd /home/

# Download Tensorflow Source Files
export TENSORFLOW_VER=r2.9
git clone https://github.com/alxhoff/TensorDSE.git
cd TensorDSE
mkdir tensorflow_env && cd tensorflow_env
git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git tensorflow_src
git clone https://github.com/google-coral/edgetpu.git


# Download and Install Flatbuffers
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles"
make
sudo ln -s $(pwd)/flatc /usr/local/bin/flatc
chmod +x $(pwd)/flatc

#Apply Patches
cd /home/TensorDSE/
chmod +x docker/patches/patch.sh && /home/TensoDSE/docker/patches/patch.sh

# Install Edge TPU Runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libedgetpu1-std

# Install EGL Headers
sudo apt-get install -y libegl1-mesa-dev

# Setup OpenCL
sudo apt-get install -y clinfo opencl-c-headers opencl-clhpp-headers ocl-icd-libopencl1 ocl-icd-dev ocl-icd-opencl-dev

# Install Vulkan
sudo apt install -y libvulkan-dev

# Install tshark
sudo apt install -y tshark

# Install Python Dependencies
sudo pip3 install cffi pyshark pandas scipy

# Install spdlog and fmt
sudo apt-get install -y libspdlog-dev libfmt-dev

#Build Backend (TODO)

echo "Installation completed successfully!"