#!/bin/bash

## Set ENV variables

## Set Git Config
# git config --global
# git config --global
# git config --global

## Exec shell
set +xe

coral_hello_world() {
    local coral_folder="/home/coral"
    mkdir ${coral_folder} && cd ${coral_folder}
    git clone https://github.com/google-coral/pycoral.git
    cd pycoral
    bash examples/install_requirements.sh classify_image.py

    python3 examples/classify_image.py \
    --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
    --labels test_data/inat_bird_labels.txt \
    --input test_data/parrot.jpg
}

run() {
    python3 main.py -m models/source/MNIST.tflite \
                    -c 1000
}

debug() {
    ipdb3 main.py -m models/source/MNIST.tflite \
                  -c 1000
}

main() {
    if [ "$DEBUG" -eq 1 ]; then
        debug
    else
        # coral_hello_world
        run
    fi
}

main "$@"
