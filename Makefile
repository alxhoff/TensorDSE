SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})
REPO := tensorflow/tensorflow

ifndef MODEL_SUMMARY
override MODEL_SUMMARY = "../../resources/model_summaries/example_summaries/MNIST_multi_1.json"	
$(info Using default MODEL_SUMMARY: $(MODEL_SUMMARY))
endif
ifndef ARCHITECTURE_SUMMARY
override ARCHITECTURE_SUMMARY = "../../resources/architecture_summaries/example_output_architecture_summary.json"
$(info Using default ARCHITECTURE_SUMMARY: $(ARCHITECTURE_SUMMARY))
endif
ifndef BENCHMARKING_RESULTS
override BENCHMARKING_RESULTS = "../../resources/benchmarking_results/example_benchmark_results.json"	
$(info Using default BENCHMARKING_RESULTS: $(BENCHMARKING_RESULTS))
endif
ifndef OUTPUT_FOLDER
override OUTPUT_FOLDER = "src/main/resources/output"
$(info Using default OUTPUT_FOLDER: $(OUTPUT_FOLDER))
endif
ifndef ILP_MAPPING
override ILP_MAPPING = true
$(info Using default ILP_MAPPING: $(ILP_MAPPING))
endif
ifndef RUNS
override RUNS = 1
$(info Using default RUNS: $(RUNS))
endif
ifndef CROSSOVER
override CROSSOVER = 0.95
$(info Using default CROSSOVER: $(CROSSOVER))
endif
ifndef POPULATION_SIZE
override POPULATION_SIZE = 100
$(info Using default POPULATION_SIZE: $(POPULATION_SIZE))
endif
ifndef PARENTS_PER_GENERATION
override PARENTS_PER_GENERATION = 50
$(info Using default PARENTS_PER_GENERATION: $(PARENTS_PER_GENERATION))
endif
ifndef OFFSPRING_PER_GENERATION
override OFFSPRING_PER_GENERATION = 50
$(info Using default OFFSPRING_PER_GENERATION: $(OFFSPRING_PER_GENERATION))
endif
ifndef GENERATIONS
override GENERATIONS = 25
$(info Using default GENERATIONS: $(GENERATIONS))
endif
ifndef VERBOSE
override VERBOSE = false
$(info Using default VERBOSE: $(VERBOSE))
endif

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

.PHONY: forcebuild
forcebuild:
	${MAKE} -C docker build;

.PHONY: run
run:
	@${MAKE} -C docker run MODEL_SUMMARY=$(MODEL_SUMMARY) ARCHITECTURE_SUMMARY=$(ARCHITECTURE_SUMMARY) BENCHMARKING_RESULTS=$(BENCHMARKING_RESULTS) OUTPUT_FOLDER=$(OUTPUT_FOLDER) ILP_MAPPING=$(ILP_MAPPING) RUNS=$(RUNS) CROSSOVER=$(CROSSOVER) POPULATION_SIZE=$(POPULATION_SIZE) PARENTS_PER_GENERATION=$(PARENTS_PER_GENERATION) OFFSPRING_PER_GENERATION=$(OFFSPRING_PER_GENERATION) GENERATIONS=$(GENERATIONS) VERBOSE=$(VERBOSE)

.PHONY: info
info:
	${MAKE} -C docker info

.PHONY: stop
stop:
	${MAKE} -C docker stop
