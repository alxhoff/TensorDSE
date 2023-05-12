SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})

.PHONY: all
all:

.PHONY: clean
clean:
	rm -rf __pycache__
	${MAKE} -C cpp_backend/ clean
	${MAKE} -C py_backend/ clean


.PHONY: compile
compile:
	${MAKE} -C cpp_backend/ build
	${MAKE} -C py_backend/ build

.PHONY: run
run:
	python3 main.py