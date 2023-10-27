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
    -k=* | --PLATFORM=*)
        PLATFORM="${i#*=}"
        shift # past argument=value
        ;;
    -j=* | --OBJECTIVE=*)
        OBJECTIVE="${i#*=}"
        shift
        ;;
    -q=* | --MULTI_MODEL=*)
        MULTI_MODEL="${i#*=}"
        shift # past argument=value
        ;;
    -l=* | --WORKLOAD_DIR=*)
        WORKLOAD_DIR="${i#*=}"
        shift # past argument=value
        ;;
    -x=* | --MODEL_NAME=*)
        MODEL_NAME="${i#*=}"
        shift # past argument=value
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
PROFILE_MODE=6
DEPLOY_MODE=7
SETUP_MODE=8

mode="$MODE"
model_name="$MODEL_NAME"
workload_name=$(basename "$WORKLOAD_DIR")

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
    run_profile_only
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    pushd DSE/TensorDSE
    if [ -z ${GA_CONFIG+x} ];
    then
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    else
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --config $GA_CONFIG --modelsummary $MODEL_SUMMARY --architecturesummary $ARCHITECTURE_SUMMARY --profilingcosts $PROFILING_COSTS --outputfolder $OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --config $GA_CONFIG --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
    run_deploy_only
}

run_no_deploy() {
    run_profile_only
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    pushd DSE/TensorDSE
    if [ -z ${GA_CONFIG+x} ];
    then
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    else
      echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --config $GA_CONFIG --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
      gradle6 run --args="--model $MODEL --config $GA_CONFIG --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
}

run_profile_only() {
    export PYTHONPATH=$(pwd):$PYTHONPATH
    mkdir resources/artifacts && mkdir resources/artifacts/model_summaries
    if [ "$MULTI_MODEL" == "true" ]; then
        for FILE in "$WORKLOAD_DIR"/*; do
            if [ -f "$FILE" ]; then
                file_with_extension=$(basename "$FILE")
                file_name="${file_with_extension%.*}"
                python3 resources/model_summaries/CreateModelSummary.py --model $"$FILE" --outputname "$file_name"_summary --outputdir resources/artifacts/model_summaries
            fi
        done
        python3 resources/model_summaries/MergeSummaries.py -s resources/artifacts/model_summaries -o $WORKLOAD_DIR -w "$workload_name"
        find resources/artifacts/model_summaries -type f -name "*.json" -exec rm {} \;
        model_summary_dir=$WORKLOAD_DIR
        model_summary="$WORKLOAD_DIR/${workload_name}_multi_model.json"

    elif [ "$MULTI_MODEL" == "false" ]; then
        python3 resources/model_summaries/CreateModelSummary.py --model $MODEL --outputname "$model_name"_summary --outputdir resources/artifacts/model_summaries
        model_summary_dir=resources/artifacts/model_summaries
        model_summary=resources/artifacts/model_summaries/"$model_name"_summary.json

    else
        echo "Cannot Create Summary. Unknown MULTI_MODEL $MULTI_MODEL"
    fi

    if [ "$PLATFORM" == "desktop" ]; then
        python3 profiler.py -s "$model_summary" -u $USBMON -c $COUNT -p $PLATFORM 
        echo "Profiling for Desktop Environment successfully completed!"

    elif [ "$PLATFORM" == "coral" ]; then
        echo "Profiling for Coral Dev Board"
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
        source ~/.bash_profile
        python3 -m utils.splitter.split -s "$model_summary"
        mdt exec "cd /media/afUSB/TensorDSE/resources && mkdir atrifacts && mkdir artifacts/model_summaries"
        mdt push "$model_summary" /media/afUSB/TensorDSE/"$model_summary_dir"
        mdt push utils/splitter/models /media/afUSB/TensorDSE/utils/splitter/
        rm -rf utils/splitter/models
        mdt exec "cd /media/afUSB/TensorDSE && python3 profiler.py -s "$model_summary" -p coral -c "$COUNT""
        mdt exec "cd /media/afUSB/TensorDSE/utils/splitter && rm -rf models"
        mdt pull /media/afUSB/TensorDSE/resources/profiling_results/coral/* resources/profiling_results/coral/
        mdt pull /media/afUSB/TensorDSE/resources/logs/* resources/logs/
        echo "Profiling for Coral Dev Board successfully completed!"
    
    elif [ "$PLATFORM" == "rpi" ]; then
        echo "Profiling for Raspberry Pi"
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
        source ~/.bash_profile
        ssh starkaf@192.168.0.6 "cd /home/starkaf/TensorDSE && mkdir resources/artifacts && mkdir resources/artifacts/models_summaries"
        scp "$model_summary" starkaf@192.168.0.6:/home/starkaf/TensorDSE/"$model_summary_dir"/
        python3 -m utils.splitter.split -s "$model_summary"
        scp -r utils/splitter/models starkaf@192.168.0.6:/home/starkaf/TensorDSE/utils/splitter/
        rm -rf utils/splitter/models
        ssh starkaf@192.168.0.6 "sudo modprobe usbmon"
        ssh starkaf@192.168.0.6 "cd /home/starkaf/TensorDSE && sudo python3 profiler.py -s "$model_summary" -p rpi -c "$COUNT" -u "$USBMON""
        ssh starkaf@192.168.0.6 "cd /home/starkaf/TensorDSE/utils/splitter && rm -rf models"
        scp starkaf@192.168.0.6:/home/starkaf/TensorDSE/resources/profiling_results/rpi/* resources/profiling_results/rpi/
        scp starkaf@192.168.0.6:/home/starkaf/TensorDSE/resources/logs/* resources/logs/
        echo "Profiling for Raspberry Pi successfully completed!"

    else
        echo "Cannot Profile. Unknown PLATFORM $PLATFORM"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
}

run_deploy_only() {
    export PYTHONPATH=$(pwd):$PYTHONPATH
    if [ "$PLATFORM" == "desktop" ]; then
        python3 deploy.py -s $MODEL_SUMMARY_W_MAPPINGS -p desktop
        echo "Deployment for Desktop Environment successfully completed!"

    elif [ "$PLATFORM" == "coral" ]; then
        echo "Deployment for Coral Dev Board"
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
        source ~/.bash_profile
        python3 -m utils.splitter.split -s $MODEL_SUMMARY_W_MAPPINGS -q True
        mdt push utils/splitter/models /media/afUSB/TensorDSE/utils/splitter/
        mdt exec "cd /media/afUSB/TensorDSE && python3 deploy.py -s '$MODEL_SUMMARY_W_MAPPINGS' -p coral"
        mdt pull /media/afUSB/TensorDSE/resources/deployment_results/coral/* resources/deployment_results/
        mdt pull /media/afUSB/TensorDSE/resources/logs/* resources/logs/
        echo "Deployment for Coral Dev Board successfully completed!"
    
    elif [ "$PLATFORM" == "rpi" ]; then
        echo "Deployment for Raspberry Pi"
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
        source ~/.bash_profile
        python3 -m utils.splitter.split -s $MODEL_SUMMARY_W_MAPPINGS -q True
        scp -r utils/splitter/models starkaf@192.168.0.6:/home/starkaf/TensorDSE/utils/splitter/
        ssh starkaf@192.168.0.6 "sudo modprobe usbmon"
        ssh starkaf@192.168.0.6 "cd /home/starkaf/TensorDSE && sudo deploy.py -s '$MODEL_SUMMARY_W_MAPPINGS' -p rpi"
        scp starkaf@192.168.0.6:/home/starkaf/TensorDSE/resources/deployment_results/rpi/* resources/deployment_results/rpi/
        scp starkaf@192.168.0.6:/home/starkaf/TensorDSE/resources/logs/* resources/logs/
        echo "Deployment for Raspberry Pi successfully completed!"

    else
        echo "Cannot Deploy. Unknown PLATFORM $PLATFORM"
    fi
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
}

setup_board() {
    if [ "$PLATFORM" -eq "CORAL" ]; then
    echo "Setting Up Coral Dev Board"
    mdt push docker/scripts/setup.sh /media/afUSB/
    mdt exec "chmod +x /media/afUSB/setup.sh && ./media/afUSB/setup.sh"
    echo "Setting Up Coral Dev Board successfully completed!"

    elif [ "$PLATFORM" -eq "RPI" ]; then
    echo "Setting Up Raspberry Pi"
    scp docker/scripts/setup.sh starkaf@192.168.0.7:/home/starkaf/
    ssh starkaf@192.168.0.7 "chmod +x /home/starkaf/setup.sh && ./home/starkaf/setup.sh"
    echo "Setting Up Raspberry Pi successfully completed!"
    
    else
        echo "Cannot Setup. Unknown PLATFORM $PLATFORM"
    fi

}

run_just_dse() {
    export PYTHONPATH=$(pwd):$PYTHONPATH
    python3 resources/model_summaries/CreateModelSummary.py --model $MODEL --outputname "$model_name" --outputdir resources/artifacts/model_summaries
    pushd DSE/TensorDSE
    echo gradle6 run --args="--objective $OBJECTIVE --model $MODEL --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    gradle6 run --args="--model $MODEL --modelsummary ../../$MODEL_SUMMARY --architecturesummary ../../$ARCHITECTURE_SUMMARY --profilingcosts ../../$PROFILING_COSTS --outputfolder ../../$OUTPUT_FOLDER --resultsfile $OUTPUT_NAME --ilpmapping $ILP_MAPPING --runs $RUNS --crossover $CROSSOVER --populationsize $POPULATION_SIZE --parentspergeneration $PARENTS_PER_GENERATION --offspringspergeneration $OFFSPRING_PER_GENERATION --generations $GENERATIONS --verbose $VERBOSE"
    cp -r /home/sources/TensorDSE/resources/* /home/tensorDSE/resources
    popd
    ls -ll resources/GA_results
}

main() {
    #git fetch origin
	#git reset --hard origin/$BRANCH
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
    elif [ "$mode" -eq $PROFILE_MODE ]; then
        echo "RUNNING PROFILE ONLY"
        run_profile_only
    elif [ "$mode" -eq $DEPLOY_MODE ]; then
        echo "RUNNING DEPLOY ONLY"
        run_deploy_only
    elif [ "$mode" -eq $SETUP_MODE ]; then
        echo "SETTING UP BOARD"
        setup_board
    else
        echo "RUNNING FULL FLOW"
        run_full_flow
    fi
}

main "$@"
