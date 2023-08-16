#!/bin/bash

source /root/.bashrc

## Exec shell
set +xe

for i in "$@"; do
  case $i in
    -m=*|--MODEL_SUMMARY=*)
      MODEL_SUMMARY="${i#*=}"
      shift # past argument=value
      ;;
    -a=*|--ARCHITECTURE_SUMMARY=*)
      ARCHITECTURE_SUMMARY="${i#*=}"
      shift # past argument=value
      ;;
    -b=*|--PROFILING_RESULTS=*)
      PROFILING_RESULTS="${i#*=}"
      shift # past argument=value
      ;;
    -o=*|--OUTPUT_FOLDER=*)
      OUTPUT_FOLDER="${i#*=}"
      shift # past argument=value
      ;;
    -i=*|--ILP_MAPPING=*)
      ILP_MAPPING="${i#*=}"
      shift # past argument=value
      ;;
    -r=*|--RUNS=*)
      RUNS="${i#*=}"
      shift # past argument=value
      ;;
    -c=*|--CROSSOVER=*)
      CROSSOVER="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--POPULATION_SIZE=*)
      POPULATION_SIZE="${i#*=}"
      shift # past argument=value
      ;;
    -n=*|--PARENTS_PER_GENERATION=*)
      PARENTS_PER_GENERATION="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--OFFSPRING_PER_GENERATION=*)
      OFFSPRING_PER_GENERATION="${i#*=}"
      shift # past argument=value
      ;;
    -g=*|--GENERATIONS=*)
      GENERATIONS="${i#*=}"
      shift # past argument=value
      ;;
    -v=*|--VERBOSE=*)
      VERBOSE="${i#*=}"
      shift # past argument=value
      ;;
    --default)
      DEFAULT=YES
      shift # past argument with no value
      ;;
    -*|--*)
      ;;
    *)
      ;;
  esac
done

DEBUG_MODE=1
TEST_MODE=2
SHELL_MODE=3

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
    ipdb3 docker/scripts/debug.py
}

test() {
    ipdb3 docker/scripts/test.py -c 2 \
                          -t 10
}

run() {
    python3 profile.py -m resources/models/example_models/MNIST_full_quanitization.tflite -c 10
    pushd DSE/TensorDSE
    gradle6 run --args="--modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingresults $PROFILING_RESULTS --outputfolder $OUTPUT_FOLDER --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    popd
    python3 deploy.py
}

main() {

    if [ "$mode" -eq $DEBUG_MODE ]; then
        debug
    elif [ "$mode" -eq $TEST_MODE ]; then
        test
    elif [ "$mode" -eq $SHELL_MODE ]; then
        bash
    else
        run
    fi
}

main "$@"
