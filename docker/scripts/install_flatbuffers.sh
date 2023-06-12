#!/bin/bash

set -e

if ! command -v bazel &>/dev/null; then
    echo "bazel is needed for this script!"
    exit 1
fi

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
    # Flatbuffers
    [ -d vcpkg ] || git clone https://github.com/Microsoft/vcpkg.git
    ./vcpkg/bootstrap-vcpkg.sh
    ./vcpkg/vcpkg integrate install
    ./vcpkg/vcpkg install flatbuffers
    [ -f /usr/local/bin/flatc ] || ln -s /home/tensorDSE/vcpkg/installed/x64-linux/tools/flatbuffers/flatc /usr/local/bin/flatc
    chmod +x /home/tensorDSE/vcpkg/installed/x64-linux/tools/flatbuffers/flatc
}

main "$@"
popd
