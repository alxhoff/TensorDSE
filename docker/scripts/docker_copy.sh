#!/bin/sh

S_FILE=$1

PWD=$(pwd)
S_FILE="${PWD}/${S_FILE}"

echo -n "Docker landing location: "
read -r D_DIR

docker cp ${S_FILE} exp-docker:/home/deb/${D_DIR}

