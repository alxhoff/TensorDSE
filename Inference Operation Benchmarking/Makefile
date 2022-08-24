SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

TAG := foobar-template
REPO := foobar-ubuntu-20.04
NAME := foobar-example
REGEX := foobar-example
DISPLAY := ${DISPLAY}
XAUTH := ${HOME}/.Xauthority

exist-docker-image = $(shell docker image ls | grep ${TAG} | tr -s ' ' | cut -f2 -d ' ')
exist-docker-running = $(shell docker ps -a | grep ${TAG} | tr -s ' ' | cut -f2 -d ' ')
ip-address = $(shell hostname -i | cut -d ' ' -f1)

all:

.PHONY: build
build:
	@echo "## Building Docker ##"
	@docker build . -t ${TAG}:${REPO}

.PHONY: run
run: $(if $(call exist-docker-image),,build)
	@echo "## Running Docker ##"
	@if [ -n "$(call exist-docker-running)" ]; then \
		echo "Docker already exists, run make clean to run new docker instance"; \
	else \
		if ! [ -e ${XAUTH} ]; then \
			touch ${XAUTH}; \
		fi; \
		docker run --name ${NAME} \
				   --rm \
				   --network=host \
				   -e DISPLAY=${DISPLAY} \
				   -v ${XAUTH}:/root/.Xauthority \
				   --detach-keys="ctrl-@" \
				   -it ${TAG}:${REPO}; \
	fi
	@ # docker option -p 8080:8080  is not needed when using host network

.PHONY: clean
clean:
	@echo "## Removing Docker ##"
	@if [ -n "$(call exist-docker-running)" ]; then \
		docker rm ${NAME}; \
	fi

.PHONY: clean-img
clean-img:
	@echo "## Removing Docker Image ##"
	@if [ -n "$(call exist-docker-image)" ]; then \
		docker rmi $(shell docker images --filter=reference="*${REGEX}*" -q); \
	fi

.PHONY: reset
reset: clean clean-img
	@echo "## Finished Resetting Docker Environment ##"
