|scripts

|results

|models_to_benchmark 
  ||generic_models
  ||ops_models
  
Running the benchmark tool requires some configurations:

	Modifying the source code of the tensorflow lite benchmark to include the Coral, namely the "benchmark_tflite_model.cc" and the bazel build file and adding the edge tpu closed library
	
	Building tensorflow from source (requires a list of configurations and installations following this tutorial https://www.tensorflow.org/install/source)
	
	Configuring the NCS and the chosen SDK (In the  case of the thesis the ncsdk2)
	
	In the case of the thesis a docker image is created with all the configurations and reused when the user needs the benchmark
	
	Running the benchmark is then reduced to running inside the benchmark_tool folder the command line tool and precising the different labels:
		--bench_folder=benchmark folder that is needed to be precised
		--cpugpu_benchmark= run benchmark for cpu and gpu if True 
		--ncs_benchmark= run benchmark for NCS if True
		--coral_benchmark= run benchmark for coral if True
		--all_benchmark= run for all if True
		--accumulate=new tests or accumulate new test results with old results
