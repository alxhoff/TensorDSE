SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

TAG := latest-gpu
REPO := tensorflow/tensorflow
NAME := coral-environment
REGEX := coral

# VOLUME := $(shell pwd)
VOLUME := $(shell cd ../ && pwd)


MODE 				:= 0
DEBUG_MODE 			:= 1
TEST_MODE 			:= 2
SHELL_MODE 			:= 3

exist-docker-image = $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ')
is-docker-running = $(shell docker ps -a | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ')
filter-docker = $(shell docker images --filter=reference="*${REGEX}*" -q)

all:

info:
	@printf "%s\n" "DOCKER IMAGES:"
	@docker image ls
	@printf "\n"
	@printf "%s\n" "RUNNING DOCKERS:"
	@docker ps -a
	@echo "Volume"
	@echo $(VOLUME)

build:
	@$(info "Grab a coffee, building will take some time...")
	@sleep 1s
	@docker build . -t ${REPO}:${TAG}

run: $(if $(call exist-docker-image),,build)
	@if [ -n "$(call is-docker-running)" ]; then \
		echo "Docker already exists, run make clean to run new docker instance"; \
	else \
		docker run --name ${NAME} \
				   --rm \
				   --privileged \
				   --net host \
				   --cap-add SYS_ADMIN \
				   -v /dev/usbmon0:/dev/usbmon0 \
				   -v /dev/bus/usb:/dev/bus/usb \
				   -v "${VOLUME}":/home/tensorDSE \
				   --detach-keys="ctrl-@" \
				   --env MODE=${MODE} \
				   -it ${REPO}:${TAG} \
				   --gpus 'all,capabilities=utility,compute'; \
	fi

debug:
	$(MAKE) -f ${FILE} run MODE=${DEBUG_MODE}

test:
	$(MAKE) -f ${FILE} run MODE=${TEST_MODE}

shell:
	$(MAKE) -f ${FILE} run MODE=${SHELL_MODE}

capture:
	ipdb3 scripts/capture.py

stop:
	@if [ -n "$(call is-docker-running)" ]; then \
		docker stop ${NAME}; \
		docker rm ${NAME}; \
	fi

rm:
	@if [ -n "$(call exist-docker-image)" ]; then \
		docker rmi $(shell docker images --filter=reference="*${REGEX}*" -q); \
	fi

# for debugging purposes only
stop-all:
	@docker ps -a | \
	awk '{print $$1}'| \
	tail -n+2 | \
	xargs -I % docker rm %

# for debugging purposes only
rm-all:
	@docker images | \
	tail -n+2 | \
	grep none | \
	awk '{print $$3}' | \
	xargs -I % docker rmi %


clean:
	@mv models/layers/.gitignore tmp
	@sudo rm -rvf models/layers/*
	@mv tmp models/layers/.gitignore
	@touch models/layers/.gitkeep
	@sudo rm -rvf models/compiled/*
	@touch models/compiled/.gitkeep
	@sudo rm -rvf results/*
	@touch  results/.gitkeep
	@sudo rm -rvf __pycache__/

purge: stop rm

purge-all: stop-all rm-all purge

.PHONY: info \
		build \
		run \
		debug \
		test \
		shell \
		capture \
		stop \
		rm \
		stop-all \
		rm-all \
		purge \
		purge-all \
		clean
