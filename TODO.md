# ToDo's

### 1. Benchmarking
* [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
* [X] make adjustments to code according to new docker workflow
* [X] change output to `JSON`
* [X] fix issue with checking TPU usb address properly
* [X] Add CPU count to JSON Format
* [ ] GPU Deployment
  - [X] Implement Function
  - Pro:
  [1](https://github.com/tensorflow/tensorflow/issues/52155#issuecomment-931498450)
  - Cons:
  [1](https://www.tensorflow.org/lite/performance/delegates)
  [2](https://github.com/tensorflow/tensorflow/issues/40706#issuecomment-648456999)
  [3](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906)
  [4](https://github.com/tensorflow/tensorflow/issues/31377#issuecomment-519331496)

- [X] fix issue using tflite_runtime on conjunction with tflite [currently hacky]
  * [Link](https://github.com/ultralytics/yolov5/issues/5709)

<details open>
<summary>
<b> Questions </b>
</summary>

- [ ] how should `sudo modprobe usbmon` be implemented? Makefile?
- [ ] Is the cost model final?

</details>

<details open>
<summary>
<b> Ideas </b>
</summary>

- [ ] suppress stdout of unnecessary commands like the conversion itself
</details>

### 2. USB
- [X] change output to `JSON`
- [X] merge with analysis/cost calculations done in benchmarking
- [ ] reevaluate USB logic, reassure correctness

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>

### 3. EXTRA
* [ ] Setup `Doxygen` documentation and "website"
* [ ] Remove docs folder from benchmarking maybe
