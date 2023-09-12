SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

TAG := latest-gpu
REPO := tensorflow/tensorflow
NAME := coral-environment
REGEX := coral

# VOLUME := $(shell pwd)
VOLUME := $(shell cd ../ && pwd)

MODE 				:= 0
DEBUG_MODE 			:= 1
TEST_MODE 			:= 2
SHELL_MODE 			:= 3
DSE_ONLY_MODE		:= 4
NO_DEPLOY_MODE		:= 5 

exist-docker-image = $(shell docker image ls | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ')
is-docker-running = $(shell docker ps -a | grep ${REPO} | tr -s ' ' | cut -f2 -d ' ')
filter-docker = $(shell docker images --filter=reference="*${REGEX}*" -q)

.PHONY: info \
		build \
		run \
		debug \
		dse \
		nodeploy \
		test \
		shell \
		capture \
		stop \
		rm \
		stop-all \
		rm-all \
		purge \
		purge-all \
		clean

all:

info:
	@printf "%s\n" "DOCKER IMAGES:"
	@docker image ls
	@printf "\n"
	@printf "%s\n" "RUNNING DOCKERS:"
	@docker ps -a
	@echo "Volume"
	@echo $(VOLUME)

.ONESHELL:
build:
	@echo "Grab a coffee, building will take some time..."
	@sleep 1s
	@echo "Docker requires a Gurobi WLS Compute Server licence."
	@echo "These can be requested at 'https://portal.gurobi.com/iam/licenses/list/' under 'WLS Academic'."
	@echo "Then going to 'https://license.gurobi.com/manager/keys' request an API key and download the generated licence."
	@echo "The licence file will contain three values: 'WLSACCESSID', 'WLSSECRET', and 'LICENSEID', please enter them now."
	read -p "WLSACCESSID:" GUROBI_ACCESSID
	read -p "WLSSECRET:" GUROBI_SECRET
	read -p "LICENSEID:" GUROBI_LICENCEID
	@echo $$GUROBI_ACCESSID
	@echo $$GUROBI_SECRET
	@echo $$GUROBI_LICENCEID
	docker build -t ${REPO}:${TAG} --build-arg WLSACCESSID=$$GUROBI_ACCESSID \
		--build-arg WLSSECRET=$$GUROBI_SECRET --build-arg LICENSEID=$$GUROBI_LICENCEID .

ifdef GA_CONFIG
run: $(if $(call exist-docker-image),,build)
	@if [ -n "$(call is-docker-running)" ]; then \
		docker stop ${NAME}; \
		docker rm ${NAME}; \
	fi; \
			docker run --name ${NAME} \
					--privileged \
					--net host \
					--cap-add SYS_ADMIN \
					-v /dev/usbmon$$USBMON:/dev/usbmon$$USBMON \
					-v /dev/bus/usb:/dev/bus/usb \
					-v "${VOLUME}":/home/tensorDSE \
					--detach-keys="ctrl-@" \
					--env MODE=${MODE} \
					-it ${REPO}:${TAG} \
					--gpus 'all,capabilities=utility,compute' \
					--USBMON=$$USBMON \
					--DATASET=$$DATASET \
					--COUNT=$$COUNT \
					--MODEL=$$MODEL \
					--MODEL_SUMMARY=$$MODEL_SUMMARY \
					--MODEL_SUMMARY_W_MAPPINGS=$$MODEL_SUMMARY_W_MAPPINGS \
					--ARCHITECTURE_SUMMARY=$$ARCHITECTURE_SUMMARY \
					--PROFILING_COSTS=$$PROFILING_COSTS \
					--OUTPUT_FOLDER=$$OUTPUT_FOLDER \
					--OUTPUT_NAME=$$OUTPUT_NAME \
					--ILP_MAPPING=$$ILP_MAPPING \
					--GA_CONFIG=$$GA_CONFIG \
					--RUNS=$$RUNS \
					--CROSSOVER=$$CROSSOVER \
					--POPULATION_SIZE=$$POPULATION_SIZE \
					--PARENTS_PER_GENERATION=$$PARENTS_PER_GENERATION \
					--OFFSPRING_PER_GENERATION=$$OFFSPRING_PER_GENERATION \
					--GENERATIONS=$$GENERATIONS \
					--VERBOSE=$$VERBOSE \
					;
else
run: $(if $(call exist-docker-image),,build)
	@if [ -n "$(call is-docker-running)" ]; then \
		docker stop ${NAME}; \
		docker rm ${NAME}; \
	fi; \
			docker run --name ${NAME} \
					--privileged \
					--net host \
					--cap-add SYS_ADMIN \
					-v /dev/usbmon$$USBMON:/dev/usbmon$$USBMON \
					-v /dev/bus/usb:/dev/bus/usb \
					-v "${VOLUME}":/home/tensorDSE \
					--detach-keys="ctrl-@" \
					--env MODE=${MODE} \
					-it ${REPO}:${TAG} \
					--gpus 'all,capabilities=utility,compute' \
					--USBMON=$$USBMON \
					--DATASET=$$DATASET \
					--COUNT=$$COUNT \
					--MODEL=$$MODEL \
					--MODEL_SUMMARY=$$MODEL_SUMMARY \
					--MODEL_SUMMARY_W_MAPPINGS=$$MODEL_SUMMARY_W_MAPPINGS \
					--ARCHITECTURE_SUMMARY=$$ARCHITECTURE_SUMMARY \
					--PROFILING_COSTS=$$PROFILING_COSTS \
					--OUTPUT_FOLDER=$$OUTPUT_FOLDER \
					--OUTPUT_NAME=$$OUTPUT_NAME \
					--ILP_MAPPING=$$ILP_MAPPING \
					--RUNS=$$RUNS \
					--CROSSOVER=$$CROSSOVER \
					--POPULATION_SIZE=$$POPULATION_SIZE \
					--PARENTS_PER_GENERATION=$$PARENTS_PER_GENERATION \
					--OFFSPRING_PER_GENERATION=$$OFFSPRING_PER_GENERATION \
					--GENERATIONS=$$GENERATIONS \
					--VERBOSE=$$VERBOSE \
					;
endif

dse: 
	$(MAKE) -f ${FILE} run MODE=${DSE_ONLY_MODE}

nodeploy: 
	$(MAKE) -f ${FILE} run MODE=${NO_DEPLOY_MODE}

debug:
	$(MAKE) -f ${FILE} run MODE=${DEBUG_MODE}

test:
	$(MAKE) -f ${FILE} run MODE=${TEST_MODE}

shell:
	$(MAKE) -f ${FILE} run MODE=${SHELL_MODE}

capture:
	ipdb3 scripts/capture.py

stop:
	@if [ -n "$(call is-docker-running)" ]; then \
		docker stop ${NAME}; \
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

purge: stop rm

purge-all: stop-all rm-all purge
