package net.sf.opendse.TensorDSE;

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
 * @author z0040rwx Ines Ben Hmida
 *
 */

public class TensorDSE {

	public static void main(String[] args) throws IOException {
		ArgumentParser parser = ArgumentParsers
				.newFor("TensorDSE")
				.build()
				.defaultHelp(true)
				.description("Find Y-graph mapping");

		parser.addArgument("-c", "--crossover").setDefault(0.9).type(Double.class).help("Cross over rate of the EA");
		parser.addArgument("-s", "--populationsize").setDefault(100).type(int.class).help("Pupulation size for the EA");
		parser.addArgument("-p", "--parentspergeneration").setDefault(50).type(int.class)
				.help("Number of parents per generation in the EA");
		parser.addArgument("-g", "--generations").setDefault(500).type(int.class)
				.help("Number of generations in the EA");
		parser.addArgument("-r", "--runs").setDefault(1000).type(int.class).help("Number of runs");
		parser.addArgument("-o", "--offspringspergeneration").setDefault(50).type(int.class)
				.help("Number of offsprings per generation");
		parser.addArgument("-m", "--modelsummary").setDefault("TensorDSE/src/main/resources/model_summaries/example_summary.csv")
				.type(String.class).help("Location of model summary CSV");
		parser.addArgument("-d", "--costfile").setDefault("TensorDSE/src/main/resources/costfiles/examplecosts.csv")
				.type(String.class).help("Directory containing cost files");
		parser.addArgument("-f", "--resultsfile").setDefault("results.csv").type(String.class)
				.help("Results file name");

		Namespace ns = null;

		try {
			ns = parser.parseArgs(args);
		} catch (ArgumentParserException e) {
			parser.handleError(e);
		}

		Double crossover_rate = ns.getDouble("crossover");
		int population_size = ns.getInt("populationsize");
		int parents_per_generation = ns.getInt("parentspergeneration");
		int generations = ns.getInt("generations");
		int test_runs = ns.getInt("runs");
		int offsprings_per_generation = ns.getInt("offspringspergeneration");
		String model_summary_loc = ns.getString("modelsummary");
		String cost_directory = ns.getString("costfile");
		String results_file = ns.getString("resultsfile");

		System.out.printf("Crossover Rate: %.2f\n", crossover_rate);
		System.out.printf("Population Size: %d\n", population_size);
		System.out.printf("Parents Per Generation: %d\n", parents_per_generation);
		System.out.printf("Generations: %d\n", generations);
		System.out.printf("Runs: %d\n", test_runs);
		System.out.printf("Offsprings Per Generations: %d\n", offsprings_per_generation);
		System.out.printf("Model Summary: %s\n", model_summary_loc);
		System.out.printf("Cost Directory: %s\n", cost_directory);
		System.out.printf("Results File: %s\n", results_file);

		if (model_summary_loc == null) {
			System.out.println("You need to provide the model summary file");
			return;
		}
		if (cost_directory == null) {
			System.out.println("You need to provide the cost files directory");
			return;
		}

		double[] objectiveVals = new double[test_runs];
		File file = new File(cost_directory);
		file.mkdir();
		FileWriter csvWriter = new FileWriter(results_file, true);

		for (int itRun = 0; itRun < test_runs; itRun++) {
			FullSpecDef fullspecdef = new FullSpecDef(model_summary_loc, cost_directory);

			// EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
			// ea.setGenerations(generations);
			// ea.setPopulationSize(population_size);
			// ea.setParentsPerGeneration(parents_per_generation);
			// ea.setOffspringsPerGeneration(offsprings_per_generation);
			// ea.setCrossoverRate(crossover_rate);

			// Module specModule = new Opt4JModule() {

			// @Override
			// protected void config() {
			// SpecificationWrapperInstance sw = new
			// SpecificationWrapperInstance(fullspecdef.getSpecification());
			// bind(SpecificationWrapper.class).toInstance(sw);
			// String objectives_s = "cost_of_mapping";
			// ExternalEvaluator evaluator = new ExternalEvaluator(objectives_s);

			// Multibinder<ImplementationEvaluator> multibinder =
			// Multibinder.newSetBinder(binder(),
			// ImplementationEvaluator.class);
			// multibinder.addBinding().toInstance(evaluator);

			// }
			// };

			// OptimizationModule opt = new OptimizationModule();

			// Collection<Module> modules = new ArrayList<Module>();
			// modules.add(ea);
			// modules.add(opt);
			// modules.add(specModule);

			// Opt4JTask task = new Opt4JTask(false);
			// task.init(modules);
			// System.out.println("start of opt");

			// try {
			// task.execute();
			// Archive archive = task.getInstance(Archive.class);

			// for (Individual individual : archive) {
			// Specification impl = ((ImplementationWrapper)
			// individual.getPhenotype()).getImplementation();
			// // individual.getObjectives();
			// // System.out.println("objectives " +
			// individual.getObjectives().getValues());
			// // objectiveVals[itRun] = individual.getObjectives().getValues();
			// SpecificationWriter writer = new SpecificationWriter();
			// String nameSolution = new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new
			// Date());
			// writer.write(impl, cost_directory + "/" + nameSolution + "_solution.xml");
			// objectiveVals[itRun] =
			// individual.getObjectives().getValues().iterator().next().getDouble();
			// System.out.println(objectiveVals[itRun]);
			// csvWriter.append("\n");
			// csvWriter.append(String.join(",", Integer.toString(itRun), nameSolution,
			// Integer.toString(generations), Integer.toString(population_size),
			// Integer.toString(parents_per_generation),
			// Integer.toString(offsprings_per_generation),
			// Double.toString(crossover_rate), Double.toString(objectiveVals[itRun])));

			// for (Mapping<Task, Resource> m : impl.getMappings()) {
			// System.out.println(m.getSource().getId() + " type " +
			// m.getSource().getAttribute("type")
			// + " HW " + m.getTarget().getId() + " number of shaves : "
			// + m.getTarget().getAttribute("num_of_shaves"));
			// }

			// }

			// } catch (Exception ex) {
			// ex.printStackTrace();
			// } finally {
			// task.close();
			// }

			// csvWriter.flush();
			// csvWriter.close();
		}

	}
}
