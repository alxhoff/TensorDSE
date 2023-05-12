SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

.PHONY: all
all:

.PHONY: clean
clean:
	rm -rf include lib __pycache__
	rm -f *.cpp
	rm -f *.so
	rm -f *.o


.PHONY: build
build: 
	python3 compile.py