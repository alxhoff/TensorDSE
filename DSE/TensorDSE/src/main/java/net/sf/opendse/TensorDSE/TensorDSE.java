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
import java.util.Properties;
import java.util.stream.Collectors;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import org.javatuples.Pair;
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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
		parser.addArgument("-s", "--populationsize").setDefault(100).type(int.class)
				.help("Pupulation size for the EA");
		parser.addArgument("-p", "--parentspergeneration").setDefault(50).type(int.class)
				.help("Perfecntage of parents per generation in the EA");
		parser.addArgument("-o", "--offspringspergeneration").setDefault(50).type(int.class)
				.help("Perfecntage of offsprings per generation");
		parser.addArgument("-g", "--generations").setDefault(25).type(int.class)
				.help("Number of generations in the EA");

		parser.addArgument("-v", "--verbose").setDefault(true).type(Boolean.class)
				.help("Enables verbose output messages");
		parser.addArgument("-u", "--visualise").setDefault(false).type(Boolean.class)
				.help("If set, OpenDSE will visualise all specificatons");

		// Other
		parser.addArgument("-r", "--runs").setDefault(1).type(int.class).help("Number of runs");

		// Input Files
		parser.addArgument("-m", "--modelsummary").setDefault(
				"../../resources/model_summaries/example_summaries/MNIST/MNIST_full_quanitization_summary.json")
				.type(String.class).help("Location of model summary CSV");
		parser.addArgument("-a", "--architecturesummary").setDefault(
				"../../resources/architecture_summaries/example_output_architecture_summary.json")
				.type(String.class).help("Location of architecture summary JSON");
		parser.addArgument("-d", "--profilingcosts")
				.setDefault("../../resources/profiling_results/MNIST_full_quanitization.json")
				.type(String.class).help("Directory containing cost files");

		// Output Files
		parser.addArgument("-f", "--resultsfile").setDefault("results.csv").type(String.class)
				.help("Results file name");
		parser.addArgument("-t", "--outputfolder").setDefault("../../resources/GA_results")
				.type(String.class);

		// ILP
		parser.addArgument("-i", "--ilpmapping").type(Boolean.class).setDefault(true)
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
			int population_size, int generations, int parents_per_generation,
			int offsprings_per_generation) {

		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(generations);
		ea.setPopulationSize(population_size);
		ea.setParentsPerGeneration(population_size * parents_per_generation / 100);
		ea.setOffspringsPerGeneration(population_size * offsprings_per_generation / 100);
		ea.setCrossoverRate(crossover_rate);

		return ea;
	}

	/**
	 * @brief The specification module binds an evaluator to the problem's
	 *        specification
	 * @param specification_definition
	 * @return Module
	 */
	private static Module GetSpecificationModule(SpecificationDefinition specification_definition,
			Boolean verbose, Boolean visualise) {

		Module specification_module = new Opt4JModule() {

			@Override
			protected void config() {
				SpecificationWrapperInstance specification_wrapper = new SpecificationWrapperInstance(
						specification_definition.getSpecification());
				bind(SpecificationWrapper.class).toInstance(specification_wrapper);

				EvaluatorMinimizeCost evaluator = new EvaluatorMinimizeCost("cost_of_mapping",
						specification_definition, verbose, visualise);

				Multibinder<ImplementationEvaluator> multibinder = Multibinder.newSetBinder(binder(),
						ImplementationEvaluator.class);
				multibinder.addBinding().toInstance(evaluator);

			}
		};

		return specification_module;
	}

	/**
	 * @param args_namespace
	 * @return String
	 */
	private static String GetModelSummaryFilePath(Namespace args_namespace) {
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
	private static String GetArchitectureSummaryFilePath(Namespace args_namespace) {
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
	private static String GetProfilingCostsFilePath(Namespace args_namespace) {
		String cost_file = args_namespace.getString("profilingcosts");
		if (cost_file == null) {
			System.out.println("You need to provide the cost files directory");
			System.exit(0);
		}
		System.out.printf("Profiling costs: %s\n", cost_file);
		return cost_file;
	}

	/**
	 * @param args_namespace
	 * @return FileWriter
	 */
	private static FileWriter GetResultsWriter(String results_file) {
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
		System.out.printf("Generations: %d\n", generations);
		System.out.printf("Population Size: %d\n", population_size);
		System.out.printf("Crossover Rate: %.2f\n", crossover_rate);
		System.out.printf("Parents Per Generation: %d%%\n", parents_per_generation);
		System.out.printf("Offsprings Per Generations: %d%%\n", offsprings_per_generation);
	}

	public static void main(String echo) {
		System.out.println(echo);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		Namespace args_namespace = GetArgNamespace(args);

		int test_runs = args_namespace.getInt("runs");
		double[] objective_values = new double[test_runs];
		File output_directory = GetOutputFolder(args_namespace);
		String results_file = String.format("%s/%s/%s", System.getProperty("user.dir"),
				output_directory, args_namespace.getString("resultsfile"));
		FileWriter csv_writer = GetResultsWriter(results_file);

		String models_description_file_path = GetModelSummaryFilePath(args_namespace);
		String profiling_costs_file_path = GetProfilingCostsFilePath(args_namespace);
		String hardware_description_file_path = GetArchitectureSummaryFilePath(args_namespace);

		System.out.println("Working Directory: " + System.getProperty("user.dir"));
		System.out.printf("Runs: %d\n", test_runs);

		String config_file = args_namespace.getString("config");
		System.out.println(String.format("CONFIG FILE: %s", config_file));

		System.out.println(String.format("Binary loc: %s",
				TensorDSE.class.getProtectionDomain().getCodeSource().getLocation().getPath()));

		// Specification contains, architecture and application graph as well as a
		// generated
		// set of possible mappings
		SpecificationDefinition specification_definition = new SpecificationDefinition(models_description_file_path,
				profiling_costs_file_path,
				hardware_description_file_path);

		String time_string = new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date());

		if (args_namespace.getBoolean("ilpmapping") == true) {
			try {
				csv_writer.append(String.join(",", "Test", "Time", "Objective", "Exec Time", "Application"));
				csv_writer.write("\n");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			Double obj_val = 0.0;

			for (int i = 0; i < test_runs; i++) {
				System.out.println(String.format("Run %d/%d\n", i + 1, test_runs));
				long exec_time = 0;

				// Solve for mappings and schedule using only ILP
				if (args_namespace.getBoolean("demo") == true) {
					// Run an ILP demo
					ILPFormuation ilp_formulation = new ILPFormuation();
					ilp_formulation.gurobiDSEExampleSixTask();
				} else {
					// Solver contains the application, architecture, and possible mapping
					// graphs as well as the list of starting tasks and the operation costs
					ScheduleSolver schedule_solver = new ScheduleSolver(specification_definition,
							args_namespace.getDouble("deactivationnumber"),
							args_namespace.getBoolean("verbose"));
					long startILP = System.currentTimeMillis();
					Pair<Double, ArrayList<ArrayList<ILPTask>>> ret = schedule_solver.solveILPMappingAndSchedule();
					// Incremental average for all tests
					obj_val = obj_val + ((ret.getValue0() - obj_val) / (test_runs + 1));
					ArrayList<ArrayList<ILPTask>> models = ret.getValue1();
					exec_time = System.currentTimeMillis() - startILP;
					System.out.println(String.format("ILP Exec time: %dms", exec_time));

					// Populate model summary with mapping information
					for (ArrayList<ILPTask> model : models) {
						for (ILPTask task : model) {
							Pattern pat = Pattern.compile("([a-z0-9_]+)-index([0-9]+)_model([0-9]+)");
							Matcher mat = pat.matcher(task.getID());
							if (mat.matches()) {
								String layer_index = mat.group(2);
								String model_index = mat.group(3);

								specification_definition.json_models
										.get(Integer.parseInt(model_index)).getLayers()
										.get(Integer.parseInt(layer_index))
										.setMapping(task.getTarget_resource_string());
								;
							}
						}
					}
				}

				try {
					csv_writer.append(String.join(",", Integer.toString(i + 1), time_string,
							Double.toString(obj_val),
							Double.toString(exec_time),
							models_description_file_path));
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		} else {
			try {
				csv_writer.append(String.join(",", "Test", "Time", "Generations", "Population Size",
						"Parents per Generation", "Offspring per Generation", "Crossover Rate",
						"Objective", "Exec Time", "Application"));
				csv_writer.write("\n");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// Solve for mappings using heuristic and schedule using ILP
			ArrayList<Double> crossover_rates = new ArrayList<Double>();
			ArrayList<Integer> population_sizes = new ArrayList<Integer>();
			ArrayList<Integer> generations = new ArrayList<Integer>();
			ArrayList<Integer> parents_per_generation = new ArrayList<Integer>();
			ArrayList<Integer> offsprings_per_generation = new ArrayList<Integer>();

			if (config_file != null) {
				System.out.println(String.format("Running GA config: %s", config_file));
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
						Arrays.asList(prop.getProperty("GENERATIONS").toString().split(","))
								.stream().map(Integer::parseInt).collect(Collectors.toList()));
				ArrayList<Double> parent_ratios = new ArrayList<Double>(Arrays
						.asList(prop.getProperty("PARENTS_PER_GENERATION").toString().split(","))
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

									System.out.println(String.format(
											"Configuration created -> gens: %d, pop size: %d, cross: %f, parents: %f = %f, offspring: %f = %f",
											gen, ps, cr, pr, pr * ps, or, or * ps));
								}

				System.out.println();

			} else {
				crossover_rates.add(args_namespace.getDouble("crossover"));
				population_sizes.add(args_namespace.getInt("populationsize"));
				generations.add(args_namespace.getInt("generations"));
				parents_per_generation.add(args_namespace.getInt("parentspergeneration"));
				offsprings_per_generation.add(args_namespace.getInt("offspringspergeneration"));
			}

			for (int c = 0; c < crossover_rates.size(); c++) {
				System.out.println(String.format("Test %d/%d", c + 1, crossover_rates.size()));
				for (int i = 0; i < test_runs; i++) {
					System.out.println(String.format("Run %d/%d\n", i + 1, test_runs));

					PrintEAParameters(crossover_rates.get(c), population_sizes.get(c),
							parents_per_generation.get(c), generations.get(c),
							offsprings_per_generation.get(c));

					// Opt4J Modules
					EvolutionaryAlgorithmModule ea_module = GetEAModule(crossover_rates.get(c),
							population_sizes.get(c), generations.get(c),
							parents_per_generation.get(c), offsprings_per_generation.get(c));

					// Bind the evaluator to the specification
					Module specification_module = GetSpecificationModule(specification_definition,
							args_namespace.getBoolean("verbose"),
							args_namespace.getBoolean("visualise"));

					OptimizationModule optimization_module = new OptimizationModule();
					optimization_module.setUsePreprocessing(false);
					optimization_module.setUseVariableOrder(false);

					Collection<Module> modules = GetModulesCollection(ea_module,
							specification_module, optimization_module);

					Opt4JTask opt4j_task = GetOpt4JTask(modules);

					try {
						long startDSE = System.currentTimeMillis();
						opt4j_task.execute();
						long exec_time = System.currentTimeMillis() - startDSE;
						System.out.println(String.format("DSE Exec time: %dms", exec_time));
						Archive archive = opt4j_task.getInstance(Archive.class);

						for (Individual individual : archive) {

							Specification implementation = ((ImplementationWrapper) individual.getPhenotype())
									.getImplementation();

							if (args_namespace.getBoolean("visualise"))
								SpecificationViewer.view(implementation);

							// Write solution to JSON
							for (Mapping mapping : implementation.getMappings().getAll()) {
								Pattern pat = Pattern.compile(
										"([a-z0-9_]+)-index([0-9]+)_model([0-9]+):([a-z0-9]+)");
								Matcher mat = pat.matcher(mapping.getId());
								if (mat.matches()) {
									String layer_type = mat.group(1);
									String layer_index = mat.group(2);
									String model_index = mat.group(3);
									String mapped_device = mat.group(4);

									specification_definition.json_models
											.get(Integer.parseInt(model_index)).getLayers()
											.get(Integer.parseInt(layer_index))
											.setMapping(mapped_device);
									;
								}
							}

							SpecificationWriter writer = new SpecificationWriter();

							writer.write(implementation,
									output_directory + "/" + time_string + "_solution.xml");

							objective_values[i] = individual.getObjectives().getValues().iterator()
									.next().getDouble();
							System.out.println(String.format("Objective: %f", objective_values[i]));
							System.out.println();

							System.out.println(
									String.format("Writing GA results to: %s", results_file));

							csv_writer.append(String.join(",", Integer.toString(i + 1), time_string,
									Integer.toString(ea_module.getGenerations()),
									Integer.toString(ea_module.getPopulationSize()),
									Integer.toString(ea_module.getParentsPerGeneration()),
									Integer.toString(ea_module.getOffspringsPerGeneration()),
									Double.toString(ea_module.getCrossoverRate()),
									Double.toString(objective_values[i]),
									Double.toString(exec_time),
									models_description_file_path));
							csv_writer.append("\n");
							csv_writer.flush();

							if (args_namespace.getBoolean("verbose"))
								for (Mapping<Task, Resource> m : implementation.getMappings()) {
									System.out.println(m.getSource().getId() + " type "
											+ m.getSource().getAttribute("type") + " HW "
											+ m.getTarget().getId());
								}
						}
					} catch (Exception ex) {
						ex.printStackTrace();
					} finally {
						opt4j_task.close();
					}
				}
			}
		}

		specification_definition.WriteJSONModelsToFile(models_description_file_path
				.substring(0, models_description_file_path.lastIndexOf('.'))
				.concat("_with_mappings.json"));
		specification_definition
				.WriteJSONModelsToFile(output_directory.getPath().concat("/mappings.json"));

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
		System.exit(0);
	}
}
