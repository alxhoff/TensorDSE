package net.sf.opendse.TensorDSE;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import org.opt4j.core.Individual;
import org.opt4j.core.optimizer.Archive;
import org.opt4j.core.start.Opt4JModule;
import org.opt4j.core.start.Opt4JTask;
import org.opt4j.optimizers.ea.EvolutionaryAlgorithmModule;

import com.google.inject.Module;
import com.google.inject.multibindings.Multibinder;

import net.sf.opendse.io.SpecificationWriter;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.optimization.ImplementationEvaluator;
import net.sf.opendse.optimization.ImplementationWrapper;
import net.sf.opendse.optimization.OptimizationModule;
import net.sf.opendse.optimization.SpecificationWrapper;
import net.sf.opendse.optimization.io.SpecificationWrapperInstance;
import net.sf.opendse.visualization.SpecificationViewer;

/**
 * Entry point for this project
 * 
 * @author Ines Ben Hmida
 * @author Alex Hoffman
 *
 */

public class TensorDSE {


	/**
	 * @param args
	 * @return Namespace
	 */
	private static Namespace GetArgNamespace(String[] args) {
		ArgumentParser parser = ArgumentParsers.newFor("TensorDSE").build().defaultHelp(true)
				.description("Find Y-graph mapping");

		// EA Parameters
		parser.addArgument("-n", "--config").type(String.class)
				.help("Location of config file to be used for running multiple tests");
		parser.addArgument("-c", "--crossover").setDefault(0.9).type(Double.class)
				.help("Cross over rate of the EA");
		parser.addArgument("-s", "--populationsize").setDefault(400).type(int.class)
				.help("Pupulation size for the EA");
		parser.addArgument("-p", "--parentspergeneration").setDefault(100).type(int.class)
				.help("Number of parents per generation in the EA");
		parser.addArgument("-g", "--generations").setDefault(300).type(int.class)
				.help("Number of generations in the EA");
		parser.addArgument("-o", "--offspringspergeneration").setDefault(100).type(int.class)
				.help("Number of offsprings per generation");
		parser.addArgument("-v", "--verbose").setDefault(true).type(Boolean.class)
				.help("Enables verbose output messages");
		parser.addArgument("-u", "--visualise").setDefault(false).type(Boolean.class)
				.help("If set, OpenDSE will visualise all specificatons");

		// Other
		parser.addArgument("-r", "--runs").setDefault(1).type(int.class).help("Number of runs");

		// Input Files
		parser.addArgument("-m", "--modelsummary")
				.setDefault("src/main/resources/modelsummaries/MNIST_multi_3.json")
				.type(String.class).help("Location of model summary CSV");
		parser.addArgument("-a", "--architecturesummary")
				.setDefault(
						"src/main/resources/architecturesummaries/outputarchitecturesummary.json")
				.type(String.class).help("Location of architecture summary JSON");
		parser.addArgument("-d", "--costfile")
				.setDefault("src/main/resources/benchmarkingresults/examplebenchmarkresults.json")
				.type(String.class).help("Directory containing cost files");

		// Output Files
		parser.addArgument("-f", "--resultsfile").setDefault("results.csv").type(String.class)
				.help("Results file name");
		parser.addArgument("-t", "--outputfolder").setDefault("src/main/resources/exampleoutput")
				.type(String.class);

		// ILP
		parser.addArgument("-i", "--ilpmapping").type(Boolean.class).setDefault(false)
				.help("If the ILP should be run instead of the DSE for finding task mappings");
		parser.addArgument("-k", "--deactivationnumber").type(Double.class).setDefault(100.0).help(
				"The large integer value used for deactivating pair-wise resource mapping constraints");
		parser.addArgument("-e", "--demo").type(Boolean.class).setDefault(false)
				.help("Run Demo instead of solving input specification");

		Namespace ns = null;

		try {
			ns = parser.parseArgs(args);
		} catch (ArgumentParserException e) {
			parser.handleError(e);
		}

		return ns;
	}


	/**
	 * @param args_namespace
	 * @return EvolutionaryAlgorithmModule
	 */
	private static EvolutionaryAlgorithmModule GetEAModule(Double crossover_rate,
			int population_size, int parents_per_generation, int generations,
			int offsprings_per_generation) {

		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(generations);
		ea.setPopulationSize(population_size);
		ea.setParentsPerGeneration(parents_per_generation);
		ea.setOffspringsPerGeneration(offsprings_per_generation);
		ea.setCrossoverRate(crossover_rate);

		return ea;
	}


	/**
	 * @brief The specification module binds an evaluator to the problem's specification
	 * @param specification_definition
	 * @return Module
	 */
	private static Module GetSpecificationModule(SpecificationDefinition specification_definition,
			Boolean verbose, Boolean visualise) {

		Module specification_module = new Opt4JModule() {

			@Override
			protected void config() {
				SpecificationWrapperInstance specification_wrapper =
						new SpecificationWrapperInstance(
								specification_definition.getSpecification());
				bind(SpecificationWrapper.class).toInstance(specification_wrapper);

				EvaluatorMinimizeCost evaluator = new EvaluatorMinimizeCost("cost_of_mapping",
						specification_definition, verbose, visualise);

				Multibinder<ImplementationEvaluator> multibinder =
						Multibinder.newSetBinder(binder(), ImplementationEvaluator.class);
				multibinder.addBinding().toInstance(evaluator);

			}
		};

		return specification_module;
	}


	/**
	 * @param args_namespace
	 * @return String
	 */
	private static String GetModelSummaryPath(Namespace args_namespace) {
		String model_summary_loc = args_namespace.getString("modelsummary");
		if (model_summary_loc == null) {
			System.out.println("You need to provide the model summary file");
			System.exit(0);
		}
		System.out.printf("Model Summary: %s\n", model_summary_loc);
		return model_summary_loc;
	}


	/**
	 * @param args_namespace
	 * @return String
	 */
	private static String GetArchitectureSummaryPath(Namespace args_namespace) {
		String architecture_summary_loc = args_namespace.getString("architecturesummary");
		if (architecture_summary_loc == null) {
			System.out.println("You need to provide the architecture summary file");
			System.exit(0);
		}
		System.out.printf("Architecture Summary: %s\n", architecture_summary_loc);
		return architecture_summary_loc;
	}


	/**
	 * @param args_namespace
	 * @return String
	 */
	private static String GetBenchmarkingResultsPath(Namespace args_namespace) {
		String cost_file = args_namespace.getString("costfile");
		if (cost_file == null) {
			System.out.println("You need to provide the cost files directory");
			System.exit(0);
		}
		System.out.printf("Cost Directory: %s\n", cost_file);
		return cost_file;
	}


	/**
	 * @param args_namespace
	 * @return FileWriter
	 */
	private static FileWriter GetResultsWriter(Namespace args_namespace) {
		String results_file = String.format("%s/%s/%s", System.getProperty("user.dir"),
				args_namespace.getString("outputfolder"), args_namespace.getString("resultsfile"));
		System.out.printf("Results File: %s\n", results_file);
		FileWriter csvWriter = null;
		try {
			csvWriter = new FileWriter(results_file, true);
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("Could not create results file");
			System.exit(0);
		}
		return csvWriter;
	}


	/**
	 * @param args_namespace
	 * @return File
	 */
	private static File GetOutputFolder(Namespace args_namespace) {
		String output_folder = args_namespace.getString("outputfolder");
		System.out.printf("Output Directory: %s\n", output_folder);
		File file = new File(output_folder);
		file.mkdir();
		return file;
	}


	/**
	 * @param ea
	 * @param spec_module
	 * @param opt
	 * @return Collection<Module>
	 */
	private static Collection<Module> GetModulesCollection(EvolutionaryAlgorithmModule ea,
			Module spec_module, OptimizationModule opt) {
		Collection<Module> ret = new ArrayList<Module>();
		ret.add(ea);
		ret.add(spec_module);
		ret.add(opt);

		return ret;
	}


	/**
	 * @param modules
	 * @return Opt4JTask
	 */
	private static Opt4JTask GetOpt4JTask(Collection<Module> modules) {
		Opt4JTask task = new Opt4JTask(false);
		task.init(modules);

		return task;
	}


	/**
	 * @param args_namespace
	 */
	private static void PrintEAParameters(Double crossover_rate, int population_size,
			int parents_per_generation, int generations, int offsprings_per_generation) {
		System.out.printf("Crossover Rate: %.2f\n", crossover_rate);
		System.out.printf("Population Size: %d\n", population_size);
		System.out.printf("Parents Per Generation: %d\n", parents_per_generation);
		System.out.printf("Generations: %d\n", generations);
		System.out.printf("Offsprings Per Generations: %d\n", offsprings_per_generation);
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {

		Namespace args_namespace = GetArgNamespace(args);

		int test_runs = args_namespace.getInt("runs");
		double[] objective_values = new double[test_runs];
		File output_directory = GetOutputFolder(args_namespace);
		FileWriter csv_writer = GetResultsWriter(args_namespace);
		String models_description_path = GetModelSummaryPath(args_namespace);
		String benchmark_results_path = GetBenchmarkingResultsPath(args_namespace);
		String hardware_description_path = GetArchitectureSummaryPath(args_namespace);

		System.out.println("Working Directory: " + System.getProperty("user.dir"));
		System.out.printf("Runs: %d\n", test_runs);


		ArrayList<Double> crossover_rates = new ArrayList<Double>();
		ArrayList<Integer> population_sizes = new ArrayList<Integer>();
		ArrayList<Integer> generations = new ArrayList<Integer>();
		ArrayList<Integer> parents_per_generation = new ArrayList<Integer>();
		ArrayList<Integer> offsprings_per_generation = new ArrayList<Integer>();

		String config_file = args_namespace.getString("config");

		if (config_file != null) {
			FileInputStream propsInput = null;
			try {
				propsInput = new FileInputStream(config_file);
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			Properties prop = new Properties();
			try {
				prop.load(propsInput);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			ArrayList<Double> crs = new ArrayList<Double>(
					Arrays.asList(prop.getProperty("CROSSOVER").toString().split(",")).stream()
							.map(Double::parseDouble).collect(Collectors.toList()));
			ArrayList<Integer> pop_sizes = new ArrayList<Integer>(
					Arrays.asList(prop.getProperty("POPULATION_SIZE").toString().split(","))
							.stream().map(Integer::parseInt).collect(Collectors.toList()));
			ArrayList<Integer> gens = new ArrayList<Integer>(
					Arrays.asList(prop.getProperty("GENERATIONS").toString().split(",")).stream()
							.map(Integer::parseInt).collect(Collectors.toList()));
			ArrayList<Double> parent_ratios = new ArrayList<Double>(
					Arrays.asList(prop.getProperty("PARENTS_PER_GENERATION").toString().split(","))
							.stream().map(Double::parseDouble).collect(Collectors.toList()));
			ArrayList<Double> offspring_ratios = new ArrayList<Double>(Arrays
					.asList(prop.getProperty("OFFSPRING_PER_GENERATION").toString().split(","))
					.stream().map(Double::parseDouble).collect(Collectors.toList()));

			for (Double cr : crs)
				for (Integer ps : pop_sizes)
					for (Integer gen : gens)
						for (Double pr : parent_ratios)
							for (Double or : offspring_ratios) {
								crossover_rates.add(cr);
								population_sizes.add(ps);
								generations.add(gen);
								parents_per_generation.add((int) (pr * ps));
								offsprings_per_generation.add((int) (or * ps));
							}


			System.out.println();
		} else {
			crossover_rates.add(args_namespace.getDouble("crossover"));
			population_sizes.add(args_namespace.getInt("populationsize"));
			generations.add(args_namespace.getInt("generations"));
			parents_per_generation.add(args_namespace.getInt("parentspergeneration"));
			offsprings_per_generation.add(args_namespace.getInt("offspringspergeneration"));
		}

		for (Double crossover_rate : crossover_rates)
			for (Integer population_size : population_sizes)
				for (Integer generation_size : generations)
					for (Integer parents : parents_per_generation)
						for (Integer offspring : offsprings_per_generation)
							for (int i = 0; i < test_runs; i++) {

								System.out.println(String.format("Run %d/%d\n", i + 1, test_runs));

								// Specification contains, architecture and application graph as
								// well as a generated set
								// of possible mappings
								SpecificationDefinition specification_definition =
										new SpecificationDefinition(models_description_path,
												benchmark_results_path, hardware_description_path);


								if (args_namespace.getBoolean("ilpmapping") == true) {

									// Solve for mappings and schedule using only ILP
									if (args_namespace.getBoolean("demo") == true) {
										// Run an ILP demo
										ILPFormuation ilp_formulation = new ILPFormuation();
										ilp_formulation.gurobiDSEExampleSixTask();
									} else {
										// Solver contains the application, architecture, and
										// possible mapping graphs as
										// well as the list of starting tasks and the operation
										// costs
										ScheduleSolver schedule_solver = new ScheduleSolver(
												specification_definition.getSpecification(),
												specification_definition.getStarting_tasks(),
												specification_definition.getOperation_costs(),
												args_namespace.getDouble("deactivationnumber"),
												args_namespace.getBoolean("verbose"));
										long startILP = System.currentTimeMillis();
										schedule_solver.solveILPMappingAndSchedule();
										System.out.println(String.format("ILP Exec time: %dms",
												System.currentTimeMillis() - startILP));
									}

									// Solve mappings using the DSE
								} else {
									// Solve for mappings using heuristic and schedule using ILP
									PrintEAParameters(crossover_rate, population_size,
											generation_size, parents, offspring);

									// Opt4J Modules
									EvolutionaryAlgorithmModule ea_module =
											GetEAModule(crossover_rate, population_size, parents,
													generation_size, offspring);
									// Bind the evaluator to the specification
									Module specification_module =
											GetSpecificationModule(specification_definition,
													args_namespace.getBoolean("verbose"),
													args_namespace.getBoolean("visualise"));
									OptimizationModule optimization_module =
											new OptimizationModule();
									Collection<Module> modules = GetModulesCollection(ea_module,
											specification_module, optimization_module);

									Opt4JTask opt4j_task = GetOpt4JTask(modules);

									try {
										long startDSE = System.currentTimeMillis();
										opt4j_task.execute();
										System.out.println(String.format("DSE Exec time: %dms",
												System.currentTimeMillis() - startDSE));
										Archive archive = opt4j_task.getInstance(Archive.class);

										for (Individual individual : archive) {

											Specification implementation =
													((ImplementationWrapper) individual
															.getPhenotype()).getImplementation();

											if (args_namespace.getBoolean("visualise"))
												SpecificationViewer.view(implementation);

											SpecificationWriter writer = new SpecificationWriter();
											String time_string =
													new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss")
															.format(new Date());

											writer.write(implementation, output_directory + "/"
													+ time_string + "_solution.xml");

											objective_values[i] = individual.getObjectives()
													.getValues().iterator().next().getDouble();
											System.out.println(objective_values[i]);

											csv_writer.append("\n");
											csv_writer.append(String.join(",", Integer.toString(i),
													time_string,
													Integer.toString(ea_module.getGenerations()),
													Integer.toString(ea_module.getPopulationSize()),
													Integer.toString(
															ea_module.getParentsPerGeneration()),
													Integer.toString(
															ea_module.getOffspringsPerGeneration()),
													Double.toString(ea_module.getCrossoverRate()),
													Double.toString(objective_values[i])));

											for (Mapping<Task, Resource> m : implementation
													.getMappings()) {
												System.out.println(m.getSource().getId() + " type "
														+ m.getSource().getAttribute("type")
														+ " HW " + m.getTarget().getId());
											}

										}

									} catch (Exception ex) {
										ex.printStackTrace();
									} finally {
										opt4j_task.close();
									}
								}
							}

		try {
			csv_writer.flush();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		try {
			csv_writer.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		System.out.println("Done!");
		System.exit(1);
	}
}
