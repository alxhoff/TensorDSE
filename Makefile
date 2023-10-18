SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))
CWD := $(shell pwd)
FOLDER := $(shell basename ${CWD})
REPO := tensorflow/tensorflow

ifndef BRANCH
override BRANCH = "master"
$(info Running branch $(BRANCH))
endif
ifndef USBMON
$(eval USBMON=$(shell python3 utils/usb/detect_tpu_bus.py 2>&1))
$(info Finding USBMON as $(USBMON))
endif
ifndef MODEL
override MODEL =resources/models/example_models/MNIST_extended_full_quanitization.tflite
$(info Using default MODEL: $(MODEL))
endif
ifndef MODEL_NAME
override MODEL_NAME = $(notdir $(basename $(MODEL)))
$(info Using default MODEL_NAME: $(MODEL_NAME))
endif
ifndef DATASET
override DATASET = "utils.datasets.MNIST"
$(info Using default DATASET: $(DATASET))
endif
ifndef COUNT
override COUNT = 2
$(info Using default COUNT: $(COUNT))
endif
ifndef MODEL_SUMMARY
override MODEL_SUMMARY = "../../resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization.json"
$(info Using default MODEL_SUMMARY: $(MODEL_SUMMARY))
endif
ifndef MODEL_SUMMARY_W_MAPPINGS
override MODEL_SUMMARY_W_MAPPINGS = "resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization_w_mappings.json"
$(info Using default MODEL_SUMMARY_W_MAPPINGS: $(MODEL_SUMMARY_W_MAPPINGS))
endif
ifndef ARCHITECTURE_SUMMARY
override ARCHITECTURE_SUMMARY = "../../resources/architecture_summaries/example_output_architecture_summary.json"
$(info Using default ARCHITECTURE_SUMMARY: $(ARCHITECTURE_SUMMARY))
endif
ifndef PLATFORM
override PLATFORM = "DESKTOP"
$(info Using default PLATFORM: $(PLATFORM))
endif
ifndef PROFILING_COSTS
override PROFILING_COSTS = "../../resources/profiling_results"
$(info Using default PROFILING_COSTS: $(PROFILING_COSTS))
endif
ifndef OUTPUT_FOLDER
override OUTPUT_FOLDER = "../../resources/GA_results"
$(info Using default OUTPUT_FOLDER: $(OUTPUT_FOLDER))
endif
ifndef OUTPUT_NAME
override OUTPUT_NAME = "results.csv"
$(info Using default OUTPUT_NAME: $(OUTPUT_NAME))
endif
ifndef ILP_MAPPING
override ILP_MAPPING = false
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
ifndef OBJECTIVE
override OBJECTIVE = "average_finish_time"
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

.PHONY: profile
profile:
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	${MAKE} -C docker profile USBMON=$(USBMON) MODEL=$(MODEL) COUNT=$(COUNT)

.PHONY: dse
dse:
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	${MAKE} -C docker dse OBJECTIVE=$(OBJECTIVE) USBMON=$(USBMON) DATASET=$(DATASET) COUNT=$(COUNT) MODEL_SUMMARY_W_MAPPINGS=$(MODEL_SUMMARY_W_MAPPINGS) MODEL=$(MODEL) MODEL_SUMMARY=$(MODEL_SUMMARY) ARCHITECTURE_SUMMARY=$(ARCHITECTURE_SUMMARY) PROFILING_COSTS=$(PROFILING_COSTS) OUTPUT_FOLDER=$(OUTPUT_FOLDER) OUTPUT_NAME=$(OUTPUT_NAME) ILP_MAPPING=$(ILP_MAPPING) RUNS=$(RUNS) CROSSOVER=$(CROSSOVER) POPULATION_SIZE=$(POPULATION_SIZE) PARENTS_PER_GENERATION=$(PARENTS_PER_GENERATION) OFFSPRING_PER_GENERATION=$(OFFSPRING_PER_GENERATION) GENERATIONS=$(GENERATIONS) VERBOSE=$(VERBOSE) BRANCH=$(BRANCH)

.PHONY: nodeploy
nodeploy: 
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	${MAKE} -C docker nodeploy OBJECTIVE=$(OBJECTIVE) USBMON=$(USBMON) DATASET=$(DATASET) COUNT=$(COUNT) MODEL_SUMMARY_W_MAPPINGS=$(MODEL_SUMMARY_W_MAPPINGS) MODEL=$(MODEL) MODEL_SUMMARY=$(MODEL_SUMMARY) ARCHITECTURE_SUMMARY=$(ARCHITECTURE_SUMMARY) PROFILING_COSTS=$(PROFILING_COSTS) OUTPUT_FOLDER=$(OUTPUT_FOLDER) OUTPUT_NAME=$(OUTPUT_NAME) ILP_MAPPING=$(ILP_MAPPING) RUNS=$(RUNS) CROSSOVER=$(CROSSOVER) POPULATION_SIZE=$(POPULATION_SIZE) PARENTS_PER_GENERATION=$(PARENTS_PER_GENERATION) OFFSPRING_PER_GENERATION=$(OFFSPRING_PER_GENERATION) GENERATIONS=$(GENERATIONS) VERBOSE=$(VERBOSE) BRANCH=$(BRANCH)

.PHONY: build
build:
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	@if ! [ $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ') ]; then\
		${MAKE} -C docker build BRANCH=$(BRANCH);\
	fi

.PHONY: forcebuild
forcebuild:
	${MAKE} -C docker build;

.PHONY: run
ifdef GA_CONFIG
run:
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	@${MAKE} -C docker run OBJECTIVE=$(OBJECTIVE) GA_CONFIG=$(GA_CONFIG) USBMON=$(USBMON) DATASET=$(DATASET) COUNT=$(COUNT) MODEL_SUMMARY_W_MAPPINGS=$(MODEL_SUMMARY_W_MAPPINGS) MODEL=$(MODEL) MODEL_SUMMARY=$(MODEL_SUMMARY) ARCHITECTURE_SUMMARY=$(ARCHITECTURE_SUMMARY) PROFILING_COSTS=$(PROFILING_COSTS) OUTPUT_FOLDER=$(OUTPUT_FOLDER) OUTPUT_NAME=$(OUTPUT_NAME) ILP_MAPPING=$(ILP_MAPPING) RUNS=$(RUNS) CROSSOVER=$(CROSSOVER) POPULATION_SIZE=$(POPULATION_SIZE) PARENTS_PER_GENERATION=$(PARENTS_PER_GENERATION) OFFSPRING_PER_GENERATION=$(OFFSPRING_PER_GENERATION) GENERATIONS=$(GENERATIONS) VERBOSE=$(VERBOSE) BRANCH=$(BRANCH)
else
run:
	git fetch https://git@github.com/alxhoff/TensorDSE.git
	git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	@${MAKE} -C docker run USBMON=$(USBMON) DATASET=$(DATASET) COUNT=$(COUNT) MODEL_SUMMARY_W_MAPPINGS=$(MODEL_SUMMARY_W_MAPPINGS) MODEL=$(MODEL) MODEL_SUMMARY=$(MODEL_SUMMARY) ARCHITECTURE_SUMMARY=$(ARCHITECTURE_SUMMARY) PROFILING_COSTS=$(PROFILING_COSTS) OUTPUT_FOLDER=$(OUTPUT_FOLDER) OUTPUT_NAME=$(OUTPUT_NAME) ILP_MAPPING=$(ILP_MAPPING) RUNS=$(RUNS) CROSSOVER=$(CROSSOVER) POPULATION_SIZE=$(POPULATION_SIZE) PARENTS_PER_GENERATION=$(PARENTS_PER_GENERATION) OFFSPRING_PER_GENERATION=$(OFFSPRING_PER_GENERATION) GENERATIONS=$(GENERATIONS) VERBOSE=$(VERBOSE) BRANCH=$(BRANCH)
endif

.PHONY: info
info:
	${MAKE} -C docker info

.PHONY: profile
profile:
	@echo "Deployment for $(PLATFORM) Environment"
	#git fetch https://git@github.com/alxhoff/TensorDSE.git
	#git reset --hard origin/$(BRANCH)
	$(info USBMON is $(USBMON))
	${MAKE} -C docker profile  USBMON=$(USBMON) MODEL=$(MODEL) COUNT=$(COUNT) PLATFORM=$(PLATFORM)
	@echo "Profiling for $(PLATFORM) environment successfully completed!"

.PHONY: stop
stop:
	${MAKE} -C docker stop
