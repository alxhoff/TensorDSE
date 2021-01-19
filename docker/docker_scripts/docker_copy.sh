#!/usr/bin/env bash

source_file=$1

cwd=$(pwd)
S_FILE="${cwd}/${source_file}"

echo -n "Docker landing location: "
read -r docker_location

docker cp ${source_file} exp-docker:/home/deb/${docker_location}

