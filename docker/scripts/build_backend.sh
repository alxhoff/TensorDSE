#!/bin/bash

set -e

if ! command -v cmake &>/dev/null; then
    echo "cmake is needed for this script!"
    exit 1
fi

[ -d /home/tensorDSE ] || mkdir /home/tensorDSE
pushd /home/tensorDSE

export TENSORFLOW_VER=r2.9
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {
    echo "Working Dir: $(pwd)"
    pushd TensorDSE
    git fetch origin
    git reset --hard origin/master
    pushd backend
    mkdir -p build
    rm -rf build/*
    pushd build
    cmake -DCMAKE_BUILD_TYPE=Release -DTF_ENV=/home/tensorDSE/tensorflow_env ..
    echo "Building backend"
    make
    echo "Instsalling backend"
    make install
    ls /usr/lib | grep .so
    popd #backend
    echo "Compiling python"
    python3 compile.py
    popd #TensorDSE
    popd #tensorDSE
}

main "$@"
popd

exit 0
