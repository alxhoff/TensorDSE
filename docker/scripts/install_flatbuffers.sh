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

[ -d /home/sources ] || mkdir /home/sources
pushd /home/sources

export TENSORFLOW_VER=r2.9
export TENSORFLOW_SRC=`pwd`/tensorflow_env/tensorflow_src

main() {
    # Flatbuffers
    [ -d vcpkg ] || git clone https://github.com/Microsoft/vcpkg.git
    ./vcpkg/bootstrap-vcpkg.sh
    ./vcpkg/vcpkg integrate install
    ./vcpkg/vcpkg install flatbuffers
    [ -f /usr/local/bin/flatc ] || ln -s /home/sources/vcpkg/installed/x64-linux/tools/flatbuffers/flatc /usr/local/bin/flatc
    chmod +x /home/sources/vcpkg/installed/x64-linux/tools/flatbuffers/flatc
}

main "$@"
popd
