FROM tensorflow/tensorflow:2.13.0-gpu

SHELL ["/bin/bash", "-l", "-c"] 

RUN set -ex
    # update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 && \
    # update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
# Base System
RUN apt-get update && \ 
    apt-get install -y wget apt-utils && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py

RUN apt-get install -y \
    build-essential locales usbutils zip unzip tar kmod \
    git vim software-properties-common apt-utils python3-dev python3-scipy python-argparse \
    python3-numpy python3-matplotlib python3-tabulate python3-pandas python3-ipdb && \
    apt-get clean && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && update-locale

RUN wget https://github.com/alxhoff/TensorDSE/raw/master/resources/model_summaries/test_summarize_model.py

# Python Libraries
RUN python -m pip install pyshark pycoral metrics wget
RUN python -m pip install silabs-mltk --upgrade