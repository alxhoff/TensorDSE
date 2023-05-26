SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})

exist-docker-image = $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ')

.PHONY: all
all:

.PHONY: clean
clean:
	rm -rf __pycache__
	${MAKE} -C deployment/ clean
	${MAKE} -C benchmarking/ clean

.PHONY: shell
shell:
	${MAKE} -C docker shell

.PHONY: build
build: $(if $(call exist-docker-image),,${MAKE} -C docker build)
	${MAKE} -C deployment build

.PHONY: run
run:
	${MAKE} -C docker run	

.PHONY: info
info:
	${MAKE} -C docker info

.PHONY: stop
stop:
	${MAKE} -C docker stop

