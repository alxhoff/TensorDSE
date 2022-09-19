# ToDo's

### 1. Benchmarking
* [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
* [X] make adjustments to code according to new docker workflow
* [X] change output to `JSON`
* [X] fix issue with checking TPU usb address properly
* [ ] fix issue using tflite_runtime on conjunction with tflite
  - [Link](https://github.com/ultralytics/yolov5/issues/5709)

<details open>
<summary>
<b> Questions </b>
</summary>

- [ ] isCPUAvailable, isTPUAvailable and isGPUAvailable, is solution good? CPU is still WIP
- [ ] how should `sudo modprobe usbmon` be implemented? Makefile? or isTPUAvailable()? Show used solution!
- [ ] Is the cost model final?

</details>

<details open>
<summary>
<b> Ideas </b>
</summary>

- [ ] suppress stdout of unnecessary commands like the conversion itself
- [ ] use requirements.txt instead pip install
</details>

### 2. Implement GPU Deployment
* Pro:
[1](https://github.com/tensorflow/tensorflow/issues/52155#issuecomment-931498450)
* Cons:
[1](https://www.tensorflow.org/lite/performance/delegates)
[2](https://github.com/tensorflow/tensorflow/issues/40706#issuecomment-648456999)
[3](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906)
[4](https://github.com/tensorflow/tensorflow/issues/31377#issuecomment-519331496)

- [X] Implement Function
- [X] Adapt Dockerfile (custom GPU delegate build through bazel + git)
- [ ] Test it out ssh server
  - sudo rights docker

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>


### 3. USB
- [X] change output to `JSON`
- [X] merge with analysis/cost calculations done in benchmarking
- [ ] reevaluate USB logic, reassure correctness

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>

### 4. DSE
* [ ] Clean up/Make Generic Ine‘s java code that generates mapping
* [ ] Add in comm costs (USB) to architecture graph
* [ ] Adapt to new JSON format

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>

### 5. EXTRA
* [ ] Setup `Doxygen` documentation and "website"
