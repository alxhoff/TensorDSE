package net.sf.opendse.TensorDSE;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.Argument;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import org.opt4j.core.Individual;
import org.opt4j.core.optimizer.Archive;
import org.opt4j.core.start.Opt4JModule;
import org.opt4j.core.start.Opt4JTask;
import org.opt4j.optimizers.ea.EvolutionaryAlgorithmModule;
import org.opt4j.optimizers.ea.SelectorDefault;
import org.opt4j.viewer.ViewerModule;
import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;

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
import net.sf.opendse.optimization.evaluator.SumEvaluator;
import net.sf.opendse.optimization.evaluator.SumEvaluatorModule.Type;
import net.sf.opendse.optimization.io.SpecificationWrapperInstance;
import net.sf.opendse.optimization.ImplementationWrapper;

/**
 * Entry point for this project
 * 
 * @author Ines Ben Hmida
 * @author Alex Hoffman
 *
 */

public class TensorDSE {

	private static Namespace GetArgNamespace(String[] args) {
		ArgumentParser parser = ArgumentParsers
				.newFor("TensorDSE")
				.build()
				.defaultHelp(true)
				.description("Find Y-graph mapping");

		// EA Parameters
		parser.addArgument("-c", "--crossover").setDefault(0.9).type(Double.class).help("Cross over rate of the EA");
		parser.addArgument("-s", "--populationsize").setDefault(100).type(int.class).help("Pupulation size for the EA");
		parser.addArgument("-p", "--parentspergeneration").setDefault(50).type(int.class)
				.help("Number of parents per generation in the EA");
		parser.addArgument("-g", "--generations").setDefault(500).type(int.class)
				.help("Number of generations in the EA");
		parser.addArgument("-o", "--offspringspergeneration").setDefault(50).type(int.class)
				.help("Number of offsprings per generation");

		// Other
		parser.addArgument("-r", "--runs").setDefault(500).type(int.class).help("Number of runs");

		// Input Files
		parser.addArgument("-m", "--modelsummary")
				.setDefault("src/main/resources/model_summaries/example_summary.csv")
				.type(String.class).help("Location of model summary CSV");
		parser.addArgument("-d", "--costfile").setDefault("src/main/resources/costfiles/examplecosts.csv")
				.type(String.class).help("Directory containing cost files");

		// Output Files
		parser.addArgument("-f", "--resultsfile").setDefault("results.csv").type(String.class)
				.help("Results file name");
		parser.addArgument("-t", "--outputfolder").setDefault("src/main/resources/example_output")
				.type(String.class);

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

		System.out.printf("Crossover Rate: %.2f\n", crossover_rate);
		System.out.printf("Population Size: %d\n", population_size);
		System.out.printf("Parents Per Generation: %d\n", parents_per_generation);
		System.out.printf("Generations: %d\n", generations);
		System.out.printf("Offsprings Per Generations: %d\n", offsprings_per_generation);

		EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
		ea.setGenerations(generations);
		ea.setPopulationSize(population_size);
		ea.setParentsPerGeneration(parents_per_generation);
		ea.setOffspringsPerGeneration(offsprings_per_generation);
		ea.setCrossoverRate(crossover_rate);

		return ea;
	}

	private static Module GetSpecModule(FullSpecDef fullspecdef) {

		Module specModule = new Opt4JModule() {

			@Override
			protected void config() {
				SpecificationWrapperInstance sw = new SpecificationWrapperInstance(fullspecdef.getSpecification());
				bind(SpecificationWrapper.class).toInstance(sw);
				String objectives_s = "cost_of_mapping";
				ExternalEvaluator evaluator = new ExternalEvaluator(objectives_s, fullspecdef);

				Multibinder<ImplementationEvaluator> multibinder = Multibinder.newSetBinder(binder(),
						ImplementationEvaluator.class);
				multibinder.addBinding().toInstance(evaluator);

			}
		};

		return specModule;
	}

	private static FileInputStream GetModelSummary(Namespace args_namespace) {
		String model_summary_loc = args_namespace.getString("modelsummary");
		System.out.printf("Model Summary: %s\n", model_summary_loc);
		if (model_summary_loc == null) {
			System.out.println("You need to provide the model summary file");
			System.exit(0);
		}
		FileInputStream model_summary = null;
		try {
			model_summary = new FileInputStream(model_summary_loc);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}

		return model_summary;
	}

	private static String GetCostFile(Namespace args_namespace) {
		String cost_file = args_namespace.getString("costfile");
		System.out.printf("Cost Directory: %s\n", cost_file);
		if (cost_file == null) {
			System.out.println("You need to provide the cost files directory");
			System.exit(0);
		}
		return cost_file;
	}

	private static FileWriter GetResultsWriter(Namespace args_namespace) {
		String results_file = args_namespace.getString("resultsfile");
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

	private static Collection<Module> GetModulesCollection(EvolutionaryAlgorithmModule ea, Module spec_module,
			OptimizationModule opt) {
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

	public static void main(String[] args) {

		System.out.println("Working Directory: " + System.getProperty("user.dir"));

		Namespace args_namespace = GetArgNamespace(args);

		int test_runs = args_namespace.getInt("runs");
		System.out.printf("Runs: %d\n", test_runs);

		double[] objectiveVals = new double[test_runs];

		File output_directory = GetOutputFolder(args_namespace);
		// File output_file = GetOutputFolder(args_namespace);
		FileWriter csvWriter = GetResultsWriter(args_namespace);
		FileInputStream model_summary = GetModelSummary(args_namespace);

		for (int itRun = 0; itRun < test_runs; itRun++) {

			FullSpecDef fullspecdef = new FullSpecDef(model_summary, GetCostFile(args_namespace));

			// Opt4J Modules
			EvolutionaryAlgorithmModule ea = GetEAModule(args_namespace);
			Module specModule = GetSpecModule(fullspecdef);
			OptimizationModule opt = new OptimizationModule();
			Collection<Module> modules = GetModulesCollection(ea, specModule, opt);

			Opt4JTask task = GetOpt4JTask(modules);

			try {
				task.execute();
				Archive archive = task.getInstance(Archive.class);

				for (Individual individual : archive) {

					Specification impl = ((ImplementationWrapper) individual.getPhenotype()).getImplementation();
					SpecificationWriter writer = new SpecificationWriter();
					String nameSolution = new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date());

					writer.write(impl, output_directory + "/" + nameSolution + "_solution.xml");
					objectiveVals[itRun] = individual.getObjectives().getValues().iterator().next().getDouble();
					System.out.println(objectiveVals[itRun]);
					csvWriter.append("\n");
					csvWriter.append(String.join(",", Integer.toString(itRun), nameSolution,
							Integer.toString(ea.getGenerations()), Integer.toString(ea.getPopulationSize()),
							Integer.toString(ea.getParentsPerGeneration()),
							Integer.toString(ea.getOffspringsPerGeneration()),
							Double.toString(ea.getCrossoverRate()), Double.toString(objectiveVals[itRun])));

					for (Mapping<Task, Resource> m : impl.getMappings()) {
						System.out.println(m.getSource().getId() + " type " +
								m.getSource().getAttribute("type")
								+ " HW " + m.getTarget().getId() + " number of shaves : "
								+ m.getTarget().getAttribute("num_of_shaves"));
					}

				}

			} catch (Exception ex) {
				ex.printStackTrace();
			} finally {
				task.close();
			}

		}

		try {
			csvWriter.flush();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
		try {
			csvWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}
}
