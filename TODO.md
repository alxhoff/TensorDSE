# ToDo's

## Benchmarking
1. Tflite Convertsion

2. Merging/Finishing Benchmarking + Analysis
* [ ] Merge/Clean double effort benchmarking done by me and `ines`
  * [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
    * [ ] make adjustments to code accordingly
  * [ ] change output to `JSON`

3. Deployment
- [ ] Implement GPU benchmarking as well
  - [ ] verify ssh connection
  - [ ] implement feature
  - [ ] test it out on remote machine

### Questions :question:
1. Currently the python packages installed are `tensorflow`, `pyshark` and `pycoral`, maybe look at a way
   to read in the requirements within a file automagically and pass it to the `Dockerfile` as it is being
   built. But seems to be enough for now.

1. Activation functions for tflite conversions? TODO?

## USB Analysis
* [ ] Remove docker code, bring changes (Dockerfile, Makefile) from `Benchmarking` over

- [ ] Reevaluate USB logic, reassure correctness
  - [ ] change output to `JSON`

### Questions :question:
1. Should usb analysis be just a module callable from within `Benchmarking`?
  - That way, docker workflow is the same for everything python related
  - Maybe consider the same for Ala's OP splitting


## Integration of python parts
- [ ] Just do it

## Java DSE
* [ ] Clean up/Make Generic Ine‘s java code that generates mapping
  * [ ] Add in comm costs (USB) to architecture graph

