# ToDo's

### 1. Tflite Conversion
#### Questions :question:
1. Activation functions for tflite conversions? TODO?
2. I changed a flag concerning `experimental_new_converter`, double check Alex

### 2. Merging/Finishing Benchmarking + Analysis
* [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
* [X] make adjustments to code according to new docker workflow
* [X] change output to `JSON`
* [ ] implement **NEW ANALYSIS** as merger between yours and Ines,
      ines uses linear regression to obtain cost parameters

Future Ideas:
- [ ] suppress stdout of unnecessary commands like the conversion itself, pollutes stdout
- [ ] at some point, instead of pip installing libraries in dockerfile, maybe use requirements.txt

#### Questions :question:
1. Should the parent model (entire model) also be benchmarked or only its ops?

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
- [X] change output to `JSON`
- [ ] merge with analysis

### Questions :question:

### 5. Java DSE
* [ ] Clean up/Make Generic Ine‘s java code that generates mapping
* [ ] Add in comm costs (USB) to architecture graph

