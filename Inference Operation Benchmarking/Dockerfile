# Installer
FROM ubuntu:20.04
LABEL Author="Daniel Duclos-Cavalcanti"
LABEL Email="daniel.duclos-cavalcanti@gmail.com"
LABEL Maintainer="Daniel Duclos-Cavalcanti"

ARG USER_NAME="coral-docker"
ARG DEBIAN_FRONTEND=noninteractive

RUN set -ex

# Base System
RUN apt-get update \
    && apt-get install -y \
    build-essential sudo locales wget curl gnupg \
    python3 python3-pip \
    git vim nano \
    && apt-get clean \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen && update-locale

# CoralDev
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo apt-key add -

RUN sudo apt-get update
RUN sudo apt-get install -y libedgetpu1-std edgetpu-compiler

# Python
# Pillow?
RUN pip3 install pyshark tensorflow
RUN pip3 install --upgrade pip
RUN sudo apt-get install -y python3-pycoral python3-ipdb

# Ports
# EXPOSE 8080

# ENV Variables
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV DEBUG 0

# Work
RUN mkdir /home/tensorDSE
WORKDIR /home/tensorDSE
ENTRYPOINT [ "/home/tensorDSE/scripts/entry.sh" ]
