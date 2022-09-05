# ToDo's

### 1. Benchmarking
* [X] create and implement working Dockerfile (w Makefile) for a reproducible containerized environment
* [X] make adjustments to code according to new docker workflow
* [X] change output to `JSON`
* [ ] implement **NEW ANALYSIS** as merger between yours and Ines, cost values?

<details closed>
<summary>
<b> Questions </b>
</summary>

- [X] Should the parent model (entire model) also be benchmarked or only its ops?
</details>

<details closed>
<summary>
<b> Ideas </b>
</summary>

- [ ] suppress stdout of unnecessary commands like the conversion itself
- [ ] use requirements.txt instead pip install
</details>

### 2. Implement GPU Deployment
- [X] Implement Function
- [ ] Adapt Dockerfile
- [ ] Test it out ssh server

<details closed>
<summary>
<b> Issues </b>
</summary>

* Pro:
[1](https://github.com/tensorflow/tensorflow/issues/52155#issuecomment-931498450)
* Cons:
[1](https://www.tensorflow.org/lite/performance/delegates)
[2](https://github.com/tensorflow/tensorflow/issues/40706#issuecomment-648456999)
[3](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906)
[4](https://github.com/tensorflow/tensorflow/issues/31377#issuecomment-519331496)
</details>

<details closed>
<summary>
<b> Questions </b>
</summary>

- [X] Should the parent model (entire model) also be benchmarked or only its ops?
</details>

<details closed>
<summary>
<b> Ideas </b>
</summary>
</details>


### 3. USB
- [ ] Reevaluate USB logic, reassure correctness
- [X] change output to `JSON`
- [ ] merge with analysis/cost calculations done in benchmarking

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>

<details closed>
<summary>
<b> Ideas </b>
</summary>
</details>

### 5. DSE
* [ ] Clean up/Make Generic Ine‘s java code that generates mapping
* [ ] Add in comm costs (USB) to architecture graph
* [ ] Adapt to new JSON format

<details closed>
<summary>
<b> Questions </b>
</summary>
</details>

<details closed>
<summary>
<b> Ideas </b>
</summary>
</details>
