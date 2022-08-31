# ToDo's

### 1. Tflite Convertsion
#### Questions :question:
1. Activation functions for tflite conversions? TODO?
2. I changed a flag concerning `experimental_new_converter`, double check Alex

### 2. Merging/Finishing Benchmarking + Analysis
* [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
* [X] make adjustments to code according to new docker workflow
* [ ] change output to `JSON`
* [ ] implement **NEW ANALYSIS** as merger between yours and Ines


### 3. Implement GPU Deployment
- [ ] Implement Function
- [ ] Test it ssh server

#### Questions :question:
1. Maybe GPU can only be deployed onto mobile GPU's like Android and IOS

links that support this statement:
[1](https://www.tensorflow.org/lite/performance/delegates)
[2](https://github.com/tensorflow/tensorflow/issues/40706#issuecomment-648456999)
[3](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906)
[4](https://github.com/tensorflow/tensorflow/issues/31377#issuecomment-519331496)

possible hacky workaround:
[1](https://github.com/tensorflow/tensorflow/issues/52155#issuecomment-931498450)

### 4. USB
- [ ] Reevaluate USB logic, reassure correctness
- [ ] change output to `JSON`
- [ ] merge with analysis

### Questions :question:

### 5. Java DSE
* [ ] Clean up/Make Generic Ine‘s java code that generates mapping
* [ ] Add in comm costs (USB) to architecture graph

