package net.sf.opendse.TensorDSE;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;

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

	private static Namespace GetArgNamespace(String[] args) {
		ArgumentParser parser = ArgumentParsers.newFor("TensorDSE").build().defaultHelp(true)
				.description("Find Y-graph mapping");

		// EA Parameters
		parser.addArgument("-c", "--crossover").setDefault(0.9).type(Double.class)
				.help("Cross over rate of the EA");
		parser.addArgument("-s", "--populationsize").setDefault(100).type(int.class)
				.help("Pupulation size for the EA");
		parser.addArgument("-p", "--parentspergeneration").setDefault(50).type(int.class)
				.help("Number of parents per generation in the EA");
		parser.addArgument("-g", "--generations").setDefault(500).type(int.class)
				.help("Number of generations in the EA");
		parser.addArgument("-o", "--offspringspergeneration").setDefault(50).type(int.class)
				.help("Number of offsprings per generation");

		// Other
		parser.addArgument("-r", "--runs").setDefault(50).type(int.class).help("Number of runs");

		// Input Files
		parser.addArgument("-m", "--modelsummary")
				.setDefault("src/main/resources/modelsummaries/MNIST_multi.json").type(String.class)
				.help("Location of model summary CSV");
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
		parser.addArgument("-i", "--ilp").type(Boolean.class).setDefault(true)
				.help("If the ILP should be run instead of the DSE");
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

	private static EvolutionaryAlgorithmModule GetEAModule(Namespace args_namespace) {
		Double crossover_rate = args_namespace.getDouble("crossover");
		int population_size = args_namespace.getInt("populationsize");
		int parents_per_generation = args_namespace.getInt("parentspergeneration");
		int generations = args_namespace.getInt("generations");
		int offsprings_per_generation = args_namespace.getInt("offspringspergeneration");

		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(generations);
		ea.setPopulationSize(population_size);
		ea.setParentsPerGeneration(parents_per_generation);
		ea.setOffspringsPerGeneration(offsprings_per_generation);
		ea.setCrossoverRate(crossover_rate);

		return ea;
	}

	private static Module GetSpecificationModule(SpecificationDefinition specification_definition) {

		Module specification_module = new Opt4JModule() {

			@Override
			protected void config() {
				SpecificationWrapperInstance specification_wrapper =
						new SpecificationWrapperInstance(
								specification_definition.getSpecification());
				bind(SpecificationWrapper.class).toInstance(specification_wrapper);

				EvaluatorMinimizeCost evaluator =
						new EvaluatorMinimizeCost("cost_of_mapping", specification_definition);

				Multibinder<ImplementationEvaluator> multibinder =
						Multibinder.newSetBinder(binder(), ImplementationEvaluator.class);
				multibinder.addBinding().toInstance(evaluator);

			}
		};

		return specification_module;
	}

	private static String GetModelSummaryPath(Namespace args_namespace) {
		String model_summary_loc = args_namespace.getString("modelsummary");
		if (model_summary_loc == null) {
			System.out.println("You need to provide the model summary file");
			System.exit(0);
		}
		System.out.printf("Model Summary: %s\n", model_summary_loc);
		return model_summary_loc;
	}

	private static String GetArchitectureSummaryPath(Namespace args_namespace) {
		String architecture_summary_loc = args_namespace.getString("architecturesummary");
		if (architecture_summary_loc == null) {
			System.out.println("You need to provide the architecture summary file");
			System.exit(0);
		}
		System.out.printf("Architecture Summary: %s\n", architecture_summary_loc);
		return architecture_summary_loc;
	}

	private static String GetBenchmarkingResultsPath(Namespace args_namespace) {
		String cost_file = args_namespace.getString("costfile");
		if (cost_file == null) {
			System.out.println("You need to provide the cost files directory");
			System.exit(0);
		}
		System.out.printf("Cost Directory: %s\n", cost_file);
		return cost_file;
	}

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

	private static File GetOutputFolder(Namespace args_namespace) {
		String output_folder = args_namespace.getString("outputfolder");
		System.out.printf("Output Directory: %s\n", output_folder);
		File file = new File(output_folder);
		file.mkdir();
		return file;
	}

	private static Collection<Module> GetModulesCollection(EvolutionaryAlgorithmModule ea,
			Module spec_module, OptimizationModule opt) {
		Collection<Module> ret = new ArrayList<Module>();
		ret.add(ea);
		ret.add(spec_module);
		ret.add(opt);

		return ret;
	}

	private static Opt4JTask GetOpt4JTask(Collection<Module> modules) {
		Opt4JTask task = new Opt4JTask(false);
		task.init(modules);

		return task;
	}

	private static void PrintEAParams(Namespace args_namespace) {
		Double crossover_rate = args_namespace.getDouble("crossover");
		int population_size = args_namespace.getInt("populationsize");
		int parents_per_generation = args_namespace.getInt("parentspergeneration");
		int generations = args_namespace.getInt("generations");
		int offsprings_per_generation = args_namespace.getInt("offspringspergeneration");
		System.out.printf("Crossover Rate: %.2f\n", crossover_rate);
		System.out.printf("Population Size: %d\n", population_size);
		System.out.printf("Parents Per Generation: %d\n", parents_per_generation);
		System.out.printf("Generations: %d\n", generations);
		System.out.printf("Offsprings Per Generations: %d\n", offsprings_per_generation);
	}

	public static void main(String[] args) {

		System.out.println("Working Directory: " + System.getProperty("user.dir"));

		Namespace args_namespace = GetArgNamespace(args);

		int test_runs = args_namespace.getInt("runs");
		System.out.printf("Runs: %d\n", test_runs);

		double[] objective_values = new double[test_runs];

		File output_directory = GetOutputFolder(args_namespace);
		FileWriter csv_writer = GetResultsWriter(args_namespace);

		PrintEAParams(args_namespace);

		String model_summary_path = GetModelSummaryPath(args_namespace);
		String benchmark_results_path = GetBenchmarkingResultsPath(args_namespace);
		String architecture_summary_path = GetArchitectureSummaryPath(args_namespace);

		for (int i = 0; i < test_runs; i++) {

			System.out.println(String.format("Run %d/%d\n", i + 1, test_runs));

			// Specification contains, architecture and application graph as well as a generated set
			// of possible mappings
			SpecificationDefinition specification = new SpecificationDefinition(model_summary_path,
					benchmark_results_path, architecture_summary_path);

			if (args_namespace.getBoolean("ilp") == true) {
				// Solve for mappings and schedule using only ILP

				if (args_namespace.getBoolean("demo") == true) {
					// Run an ILP demo
					ILPSolver ilps = new ILPSolver();
					ilps.gurobiDSEExampleSixTask();
				} else {
					// Solve the given input

					// Solver contains the application, architecture, and possible mapping graphs as
					// well as the list of starting tasks and the operation costs
					Solver solver = new Solver(specification.specification, specification.application_graphs,
							specification.starting_tasks, specification.GetOperationCosts(),
							args_namespace.getDouble("deactivationnumber"));
					solver.solveILP();
				}
			} else {
				// Solve for mappings using heuristic and schedule using ILP

				// Opt4J Modules
				EvolutionaryAlgorithmModule ea_module = GetEAModule(args_namespace);
				Module specification_module = GetSpecificationModule(specification);
				OptimizationModule optimization_module = new OptimizationModule();
				Collection<Module> modules =
						GetModulesCollection(ea_module, specification_module, optimization_module);

				Opt4JTask opt4j_task = GetOpt4JTask(modules);

				try {
					opt4j_task.execute();
					Archive archive = opt4j_task.getInstance(Archive.class);

					for (Individual individual : archive) {

						Specification implementation =
								((ImplementationWrapper) individual.getPhenotype())
										.getImplementation();

						SpecificationViewer.view(implementation);

						SpecificationWriter writer = new SpecificationWriter();
						String time_string =
								new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date());

						writer.write(implementation,
								output_directory + "/" + time_string + "_solution.xml");

						objective_values[i] = individual.getObjectives().getValues().iterator()
								.next().getDouble();
						System.out.println(objective_values[i]);

						csv_writer.append("\n");
						csv_writer.append(String.join(",", Integer.toString(i), time_string,
								Integer.toString(ea_module.getGenerations()),
								Integer.toString(ea_module.getPopulationSize()),
								Integer.toString(ea_module.getParentsPerGeneration()),
								Integer.toString(ea_module.getOffspringsPerGeneration()),
								Double.toString(ea_module.getCrossoverRate()),
								Double.toString(objective_values[i])));

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
