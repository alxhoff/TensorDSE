#!/bin/bash

source /root/.bashrc

## Exec shell
set +xe

BRANCH=master

for i in "$@"; do
    case $i in
    -f=* | --GA_CONFIG=*)
        GA_CONFIG="${i#*=}"
        shift # past argument=value
        ;;
    -w=* | --MODEL_SUMMARY_W_MAPPINGS=*)
        MODEL_SUMMARY_W_MAPPINGS="${i#*=}"
        shift # past argument=value
        ;;
    -e=* | --DATASET=*)
        DATASET="${i#*=}"
        shift # past argument=value
        ;;
    -d=* | --MODEL=*)
        MODEL="${i#*=}"
        shift # past argument=value
        ;;
    -t=* | --COUNT=*)
        COUNT="${i#*=}"
        shift # past argument=value
        ;;
    -u=* | --USBMON=*)
        USBMON="${i#*=}"
        shift # past argument=value
        ;;
    -m=* | --MODEL_SUMMARY=*)
        MODEL_SUMMARY="${i#*=}"
        shift # past argument=value
        ;;
    -a=* | --ARCHITECTURE_SUMMARY=*)
        ARCHITECTURE_SUMMARY="${i#*=}"
        shift # past argument=value
        ;;
    -b=* | --PROFILING_COSTS=*)
        PROFILING_COSTS="${i#*=}"
        shift # past argument=value
        ;;
    -o=* | --OUTPUT_FOLDER=*)
        OUTPUT_FOLDER="${i#*=}"
        shift # past argument=value
        ;;
    -z=* | --OUTPUT_NAME=*)
        OUTPUT_NAME="${i#*=}"
        shift # past argument=value
        ;;
    -i=* | --ILP_MAPPING=*)
        ILP_MAPPING="${i#*=}"
        shift # past argument=value
        ;;
    -r=* | --RUNS=*)
        RUNS="${i#*=}"
        shift # past argument=value
        ;;
    -c=* | --CROSSOVER=*)
        CROSSOVER="${i#*=}"
        shift # past argument=value
        ;;
    -p=* | --POPULATION_SIZE=*)
        POPULATION_SIZE="${i#*=}"
        shift # past argument=value
        ;;
    -n=* | --PARENTS_PER_GENERATION=*)
        PARENTS_PER_GENERATION="${i#*=}"
        shift # past argument=value
        ;;
    -s=* | --OFFSPRING_PER_GENERATION=*)
        OFFSPRING_PER_GENERATION="${i#*=}"
        shift # past argument=value
        ;;
    -g=* | --GENERATIONS=*)
        GENERATIONS="${i#*=}"
        shift # past argument=value
        ;;
    -v=* | --VERBOSE=*)
        VERBOSE="${i#*=}"
        shift # past argument=value
        ;;
    -h=* | --BRANCH=*)
        BRANCH="${i#*=}"
        shift # past argument=value
        ;;
    -j=* | --OBJECTIVE=*)
        OBJECTIVE="${i#*=}"
        shift
        ;;
    --default)
        DEFAULT=YES
        shift # past argument with no value
        ;;
    -* | --*) ;;

    *) ;;

    esac
done

DEBUG_MODE=1
TEST_MODE=2
SHELL_MODE=3
DSE_ONLY_MODE=4
NO_DEPLOY_MODE=5
PROFILE=6

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
    ipdb3 docker/scripts/test.py -c 2 -t 10
}

run_full_flow() {
    python3 profiler.py -u $USBMON -m $MODEL -c $COUNT
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    pushd DSE/TensorDSE
    if [ -z ${GA_CONFIG+x} ];
    then
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    else
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --config $GA_CONFIG --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --config $GA_CONFIG --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
    python3 deploy.py -m $MODEL -s $MODEL_SUMMARY_W_MAPPINGS -d $DATASET
}

run_no_deploy() {
    python3 profiler.py -u $USBMON -m $MODEL -c $COUNT
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    pushd DSE/TensorDSE
    if [ -z ${GA_CONFIG+x} ];
    then
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    else
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --config $GA_CONFIG --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --config $GA_CONFIG --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
}

run_profile_only() {
    python3 profiler.py -u $USBMON -m $MODEL -c $COUNT
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
}

run_just_dse() {
    pushd DSE/TensorDSE
    echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    gradle6 run --args="--model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
}

main() {
    git fetch origin
	git reset --hard origin/$BRANCH
    MODULE="usbmon"
    if lsmod | grep -wq "$MODULE"; then
    echo "$MODULE is loaded!"
    else
    echo "$MODULE is not loaded!"
    exit 1
    fi
    if [ "$mode" -eq $DEBUG_MODE ]; then
        debug
    elif [ "$mode" -eq $TEST_MODE ]; then
        test
    elif [ "$mode" -eq $SHELL_MODE ]; then
        bash
    elif [ "$mode" -eq $DSE_ONLY_MODE ]; then
        echo "RUNNING JUST DSE"
        run_just_dse
    elif [ "$mode" -eq $NO_DEPLOY_MODE ]; then
        echo "RUNNING NO DEPLOY"
        run_no_deploy
    elif [ "$mode" -eq $PROFILE ]; then
        echo "RUNNING PROFILE ONLY"
        run_profile_only
    else
        echo "RUNNING FULL FLOW"
        run_full_flow
    fi
}

main "$@"
