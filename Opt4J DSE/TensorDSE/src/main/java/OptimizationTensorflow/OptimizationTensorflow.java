package OptimizationTensorflow;

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

import org.opt4j.core.Individual;
import org.opt4j.core.optimizer.Archive;
import org.opt4j.core.start.Opt4JModule;
import org.opt4j.core.start.Opt4JTask;
import org.opt4j.optimizers.ea.EvolutionaryAlgorithmModule;
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
 * @author z0040rwx Ines Ben Hmida 
 *
 */

public class OptimizationTensorflow {
	
	public static void main(String[] args) throws IOException {
		System.out.println("on y est ");
		//if (args[0] == null || args[0].trim().isEmpty()) {
		if (1==0) {
	        System.out.println("You need to specify a path!");
	        return;
	    } else {
	    	
	    	String modelsummaryString = args[0];
	    	String folderSolutions = args[1];
	    	String csvresult_file = args[2];
	    	int runs = Integer.parseInt(args[3]);
	    	int genN = Integer.parseInt(args[4]);
	    	int alpha = Integer.parseInt(args[5]); 
	    	int Mu = Integer.parseInt(args[6]);
	    	int Lambda = Integer.parseInt(args[7]);
	    	double crossRate = Double.parseDouble(args[8]);
	    	
	    	
	    	//System.out.println(args[0]+ args[1]);
	    	/*
	    	String modelsummaryString = "src/main/resources/models_summaries/mgen_summary.csv";
	    	String folderSolutions = "src/main/resources/mobilnet_test";
	    	int runs = 2;
	    	int genN = 100;
	    	int alpha =50; 
	    	int Mu = 25;
	    	int Lambda = 25;
	    	double crossRate = 0.95;
	    	*/
	    	double[] objectiveVals = new double[runs]; 
	    	File file = new File(folderSolutions);
			file.mkdir();
	    	FileWriter csvWriter = new FileWriter(csvresult_file, true);
			//FileWriter csvWriter = new FileWriter(folderSolutions + "\\results_" + Double.toString(runs) + ".csv");
	    	//csvWriter.append("Iteration,");
	    	//csvWriter.append("solution_name,");
	    	//csvWriter.append("generations,");
	    	//csvWriter.append("alpha,");
	    	//csvWriter.append("Mu,");
	    	//csvWriter.append("Lambda,");
	    	//csvWriter.append("Crossover_rate,");
	    	//csvWriter.append("Cost_of_mapping");
	    	
	    	//FileWriter csvWriter2 = new FileWriter("src/main/resources/p_100.csv");
	    	//csvWriter2.append("Cost_of_mapping");
	    	
	    	//csvWriter2.append("\n");

	    	for(int itRun= 0; itRun <runs; itRun ++) {
	    		FullSpecDef fullspecdef = new FullSpecDef(modelsummaryString);
	    		
				EvolutionaryAlgorithmModule ea = new EvolutionaryAlgorithmModule();
				ea.setGenerations(genN);
				ea.setAlpha(alpha);
				ea.setMu(Mu);
				ea.setLambda(Lambda);
				ea.setCrossoverRate(crossRate);
					
				
				Module specModule = new Opt4JModule() {

					@Override
					protected void config() {
						SpecificationWrapperInstance sw = new SpecificationWrapperInstance(fullspecdef.getSpecification());
						bind(SpecificationWrapper.class).toInstance(sw);
						String objectives_s = "cost_of_mapping";
						ExternalEvaluator evaluator = new ExternalEvaluator(objectives_s);
						
						Multibinder<ImplementationEvaluator> multibinder = Multibinder.newSetBinder(binder(),
								ImplementationEvaluator.class);
						multibinder.addBinding().toInstance(evaluator);
						
					}
				};

				OptimizationModule opt = new OptimizationModule();

				Collection<Module> modules = new ArrayList<Module>();
				modules.add(ea);
				modules.add(opt);
				modules.add(specModule);

				Opt4JTask task = new Opt4JTask(false);
				task.init(modules);
				System.out.println("start of opt");

				try {
					task.execute();
					Archive archive = task.getInstance(Archive.class);
					
					for (Individual individual : archive) {
						Specification impl = ((ImplementationWrapper) individual.getPhenotype()).getImplementation();
						//individual.getObjectives();
						//System.out.println("objectives " + individual.getObjectives().getValues());
						//objectiveVals[itRun] = individual.getObjectives().getValues();
						SpecificationWriter writer = new SpecificationWriter();
						String nameSolution = new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date());
						writer.write(impl, folderSolutions + "/" + nameSolution + "_solution.xml");
						objectiveVals[itRun] = individual.getObjectives().getValues().iterator().next().getDouble();
						System.out.println(objectiveVals[itRun]);
						csvWriter.append("\n");
						csvWriter.append(String.join(",", Integer.toString(itRun), nameSolution, Integer.toString(genN), Integer.toString(alpha), Integer.toString(Mu) , Integer.toString(Lambda), Double.toString(crossRate), Double.toString(objectiveVals[itRun])));
					
						
						for(Mapping<Task,Resource> m: impl.getMappings()) {
							System.out.println(m.getSource().getId() + " type "+ m.getSource().getAttribute("type") + " HW " + m.getTarget().getId() +  " number of shaves : " + m.getTarget().getAttribute("num_of_shaves"));
						}
						
					}

				} catch (Exception e) {
					e.printStackTrace();
				} finally {
					task.close();
					}
	    		
	    	}
	    csvWriter.flush();
	    csvWriter.close();
			
	    }	
	
	}

	

}
