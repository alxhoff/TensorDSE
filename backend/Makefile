SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

isbuildEmpty = $(shell ls build/)

.PHONY: all
all:

.PHONY: clean
clean:
	if [ -n "$(call isbuildEmpty)" ]; then \
		rm -rf build/*; \
	fi
	touch build/.gitkeep

.PHONY: build
build: clean
	cd build/ && cmake -DCMAKE_BUILD_TYPE=Release ..
	$(MAKE) -C build/
	$(MAKE) -C build/ install
	python3 compile.py
