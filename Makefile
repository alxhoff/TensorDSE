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
	${MAKE} -C backend/ clean

.PHONY: shell
shell:
	${MAKE} -C docker shell

.PHONY: build
<<<<<<< HEAD
build: 
	@if ! [ $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ') ]; then\
		${MAKE} -C docker build;\
	fi
	${MAKE} -C deployment build
=======
build: $(if $(call exist-docker-image),,${MAKE} -C docker build)
	${MAKE} -C backend build
>>>>>>> b11e40ed6365b5aa104fbe58b0374fa8de20dc3a

.PHONY: run
run:
	${MAKE} -C docker run	

.PHONY: info
info:
	${MAKE} -C docker info

.PHONY: stop
stop:
	${MAKE} -C docker stop

