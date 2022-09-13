#!/bin/bash

## Set ENV variables

## Set Git Config
# git config --global
# git config --global
# git config --global

## Exec shell
set +xe

DEBUG_MODE=1
TEST_MODE=2
HELLO_WORLD_MODE=3
SHELL_MODE=4

mode="$MODE"

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

debug() {
    ipdb3 main.py -m models/source/MNIST.tflite \
                  -c 1000
}

test() {
    ipdb3 scripts/test.py
}

run() {
    python3 main.py -m models/source/MNIST.tflite \
                    -c 1000
}

hello_world() {
    coral_hello_world
}

main() {
    if [ "$mode" -eq $DEBUG_MODE ]; then
        debug
    elif [ "$mode" -eq $TEST_MODE ]; then
        test
    elif [ "$mode" -eq $HELLO_WORLD_MODE ]; then
        hello_world
    elif [ "$mode" -eq $SHELL_MODE ]; then
        bash
    else
        run
    fi
}

main "$@"
