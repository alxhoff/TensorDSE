SHELL := /bin/bash
FILE := $(lastword $(MAKEFILE_LIST))

clean:
	@mv models/layers/.gitignore tmp
	@sudo rm -rvf models/layers/*
	@mv tmp models/layers/.gitignore
	@touch models/layers/.gitkeep
	@sudo rm -rvf models/compiled/*
	@touch models/compiled/.gitkeep
	@sudo rm -rvf results/*
	@touch  results/.gitkeep
	@sudo rm -rvf __pycache__/

.PHONY: clean
