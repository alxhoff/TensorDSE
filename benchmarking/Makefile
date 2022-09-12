SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

TAG := coral-docker
REPO := coral-ubuntu-20.04
NAME := coral-environment
REGEX := coral

VOLUME := $(shell pwd)

MODE := 0
DEBUG_MODE := 1
TEST_MODE := 2

exist-docker-image = $(shell docker image ls | grep ${TAG} | tr -s ' ' | cut -f2 -d ' ')
is-docker-running = $(shell docker ps -a | grep ${TAG} | tr -s ' ' | cut -f2 -d ' ')
filter-docker = $(shell docker images --filter=reference="*${REGEX}*" -q)

all: build

info:
	@printf "%s\n" "DOCKER IMAGES:"
	@docker image ls
	@printf "\n"
	@printf "%s\n" "RUNNING DOCKERS:"
	@docker ps -a

build:
	@$(info "Grab a coffee, building will take some time...")
	@sleep 1s
	@docker build . -t ${TAG}:${REPO}

run: $(if $(call exist-docker-image),,build)
	@if [ -n "$(call is-docker-running)" ]; then \
		echo "Docker already exists, run make clean to run new docker instance"; \
	else \
		docker run --name ${NAME} \
				   --rm \
				   --privileged \
				   --env MODE=${MODE} \
				   -v /sys/bus/usb/devices:/sys/bus/usb/devices \
				   -v /dev/bus/usb:/dev/bus/usb \
				   -v "${VOLUME}":/home/tensorDSE \
				   --detach-keys="ctrl-@" \
				   -it ${TAG}:${REPO}; \
	fi

debug:
	$(MAKE) -f ${FILE} run MODE=${DEBUG_MODE}

test:
	$(MAKE) -f ${FILE} run MODE=${TEST_MODE}

stop:
	@if [ -n "$(call is-docker-running)" ]; then \
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
		stop \
		rm \
		stop-all \
		rm-all \
		purge \
		purge-all \
		clean
