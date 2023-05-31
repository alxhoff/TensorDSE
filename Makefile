SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})
REPO := tensorflow/tensorflow

.PHONY: all
all:

.PHONY: clean
clean:
	rm -rf __pycache__
	${MAKE} -C backend clean

.PHONY: shell
shell:
	${MAKE} -C docker shell

.PHONY: build
build:
	@if ! [ $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ') ]; then\
		${MAKE} -C docker build;\
	fi

.PHONY: run
run:
	${MAKE} -C docker run

.PHONY: info
info:
	${MAKE} -C docker info

.PHONY: stop
stop:
	${MAKE} -C docker stop
