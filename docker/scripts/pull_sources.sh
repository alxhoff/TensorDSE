#!/bin/bash

BRANCH=master

for i in "$@"; do
    case $i in
    -b=* | --BRANCH=*)
        BRANCH="${i#*=}"
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

set -e

[ -d /home/sources ] || mkdir /home/sources
pushd /home/sources

export TENSORFLOW_VER=r2.9
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {
    [ -d TensorDSE ] || git clone https://github.com/alxhoff/TensorDSE.git
    pushd TensorDSE
    git checkout ${BRANCH}
    popd
    [ -d tensorflow_env ] || mkdir tensorflow_env
    cd tensorflow_env
    [ -d tensorflow_src ] || git clone -b ${TENSORFLOW_VER} --depth 1 https://github.com/tensorflow/tensorflow.git tensorflow_src
    [ -d edgetpu ] || git clone https://github.com/google-coral/edgetpu.git
    [ -d flatbuffers ] || git clone https://github.com/google/flatbuffers
}

main "$@"
popd

exit 0
