SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})

.PHONY: all
all:

.PHONY: clean
clean:
	rm -rf __pycache__
	${MAKE} -C backend/ clean


.PHONY: build
build:
	${MAKE} -C backend/ build

.PHONY: run
run:
	python3 main.py
