package net.sf.opendse.TensorDSE;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import net.sf.opendse.io.SpecificationWriter;
import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Communication;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.model.parameter.Parameters;
import net.sf.opendse.visualization.SpecificationViewer;


/**
 * The {@code FullSpecDef} is the class defining the Specification that corresponds to the graph
 * of the deep learning model to which we want to optimize the performance.
 * 
 * @author z0040rwx Ines Ben Hmida 
 *
 *@param modelsummary
 *		 user entry of the file path a csv file that summarized the graph of the considered deep learning model
 *@param costsfile
 *		A csv file that contains the costs deducted from running the benchmarks 
 *
 */
public class FullSpecDef {
	
	public static Specification specification;
	// ******************************************************************************
	//op_costs is set with the csv file containing the latest tests we have conducted 
	//and which is included in the resources folder of this project  
	//*******************************************************************************
	private OpCosts op_costs;// = new OpCosts("src/main/resources/costfiles/costs2910.csv");
	public String fullSpecName;
	
	public FullSpecDef(String modelsummary, String costsfilepath) throws IOException {
		this.op_costs = new OpCosts(costsfilepath);
//		this.specification = ApplicationFromTfModel(modelsummary);
	}
	
	public Specification ApplicationFromTfModel(String modelsummary) throws IOException {
//		Application<Task, Dependency> application = new Application<Task, Dependency>();
//		Task t;
//		HashSet<Task> all_tasks = new HashSet<Task>();
//		Dependency d;
//		FileInputStream f = new FileInputStream(modelsummary);
//		BufferedReader br = new BufferedReader(new InputStreamReader(f));
//		String strLine;
//		String output_shape;
//		String inputs;
//		String task_name;
//		String task_type;
//		String headerLine = br.readLine();
//		while ((strLine = br.readLine()) != null) {
//			String[] data = strLine.split(",");
//			task_name = data[1].trim();
//			task_type = data[2].trim();
//			inputs = data[3].trim();
//			output_shape = data[4].trim();		
//			
//			t = new Task(task_name);
//			t.setAttribute("output_shape", output_shape);
//			t.setAttribute("type", task_type);
//			t.setAttribute("inputs", inputs);
//			all_tasks.add(t);
//			application.addVertex(t);
//			
//		}
//		
//		Iterator<Task> it_task = all_tasks.iterator();
//		String[] input_liste;
//		Task current_task;
//		String input_shape;
//		String[] input_shape_liste;
//		
//		while (it_task.hasNext()) {
//			int input_size = 0;
//			int input_size_new = 0;
//			current_task = it_task.next();
//			inputs = current_task.getAttribute("inputs");
//			System.out.println(current_task);
//			if (inputs.isEmpty()){
//				continue;
//			}
//			else {
//				input_liste = inputs.trim().split("\\| ");
//				for(int i=0; i<input_liste.length; i++) {
//					if (quick_search(all_tasks, input_liste[i].trim())) { 
//						Communication c = new Communication(current_task.getId() + i);
//						application.addVertex(c);
//						//Random r = new Random();
//						Random r = new Random();
//						application.addEdge(new Dependency(application.getVertex(input_liste[i].trim()).getId() + "i" + Integer.toString(r.nextInt(100*all_tasks.size()))+ Integer.toString(r.nextInt(100*all_tasks.size())) + current_task.getId()), application.getVertex(input_liste[i].trim()), c);
//						//d = new Dependency(current_task.getId() + i);
//						Random r1 = new Random();
//						application.addEdge(new Dependency(current_task.getId() + Integer.toString(r1.nextInt(100*all_tasks.size())) + i+ c.getId() ),  c, current_task);
//						input_shape = application.getVertex(input_liste[i]).getAttribute("output_shape");
//						input_shape_liste = input_shape.trim().substring(1,input_shape.length()-1).split("\\|");
//						if (!(input_shape_liste.length==0)){
//							input_size_new = 1;
//							for(int j=0; j<input_shape_liste.length; j++) {
//								try
//								{
//								  input_size_new = input_size_new*Integer.parseInt(input_shape_liste[j]);
//								}
//								catch(NumberFormatException e)
//								{
//								  //not a double
//									continue;
//								}
//								
//							}
//					
//						}
//					}
//					if (input_size_new> input_size) input_size=input_size_new;
//					
//						
//				}
//				current_task.setAttribute("input_shape", input_size);
//			
//			}
//			
//		}
//		
//		//*******************************************************************************
//		//Architecture : In our setup, the architecture is defined by 3 resources :
//		//the NCS2 Hardware accelerator the labtop and the coral edge tpu. W also add 
//		//the busses because the two hardware accelerators are connected through USB com
//		//For the NCS it is possible to vary the number of shaves and for desktop the number of threads
//		// #TODO Nevertheless to be done remove the number of threads for coral when calculating the costs
//		//*******************************************************************************
//		Architecture<Resource, Link> architecture = new Architecture<Resource, Link>();
//		Resource r1 = new Resource("NCS2");
//		Resource r2 = new Resource("Desktop(CPU + GPU)");
//		Resource r3 = new Resource("coral edgetpu");
//		Resource bus12 = new Resource("bus12");
//		Resource bus23 = new Resource("bus23");
//		r1.setAttribute("num_of_shaves", Parameters.select(12, 12, 1, 2, 3, 4, 5, 6 ,7, 8, 9 , 10, 11));
//		r2.setAttribute("num_of_shaves", Parameters.select(12, 12, 1, 2, 3, 4, 5, 6 ,7, 8, 9 , 10, 11));
//		r3.setAttribute("num_of_shaves", Parameters.select(12, 12, 1, 2, 3, 4, 5, 6 ,7, 8, 9 , 10, 11));
//		//r1.setAttribute("num_of_shaves", 12);
//		//r2.setAttribute("num_of_shaves", 12);
//		//r3.setAttribute("num_of_shaves", 12);
//		r1.setAttribute("input_type", "float");
//		r2.setAttribute("input_type", "float");
//		r3.setAttribute("input_type", "uint8");
//		//bus12.setAttribute("cost",  op_costs.comCost[0]);
//		//bus23.setAttribute("cost", op_costs.comCost[1]);
//		architecture.addVertex(r1);
//		architecture.addVertex(bus12);
//		architecture.addVertex(bus23);
//		architecture.addVertex(r2);
//		architecture.addVertex(r3);
//		Link l1 = new Link("l1");
//		Link l2 = new Link("l2");
//		Link l3 = new Link("l3");
//		Link l4 = new Link("l4");
//		l1.setAttribute("cost", op_costs.comCost[0]/2);
//		l2.setAttribute("cost", op_costs.comCost[0]/2);
//		l3.setAttribute("cost", op_costs.comCost[1]/2);
//		l4.setAttribute("cost", op_costs.comCost[1]/2);
//		architecture.addEdge(l1, r1, bus12);
//		architecture.addEdge(l2, bus12, r2);
//		architecture.addEdge(l3,r2,bus23);
//		architecture.addEdge(l4, bus23, r3);
//		
//		Mappings<Task, Resource> mappings = new Mappings<Task, Resource>();
//		Iterator<Task> task_it = all_tasks.iterator();
//		HashSet<Mapping<Task, Resource>> all_mappings = new HashSet<Mapping<Task, Resource>>();
//		Mapping<Task, Resource> m;
//		String task_name_lower;
//		while (task_it.hasNext()) {
//			current_task = task_it.next();
//			if (current_task instanceof Communication) {
//				continue;
//			}
//			task_name_lower = current_task.getAttribute("type").toString().toLowerCase();
//
//			
//			//NCS2
//			if (op_costs.OpCost.get(r1.getId()).containsKey(task_name_lower)) {
//				m = new Mapping<Task, Resource>(current_task.getId() + r1.getId() ,current_task, r1);
//				all_mappings.add(m);
//				mappings.add(m);
//			}
//			
//			//desktop
//			if  (op_costs.OpCost.get(r2.getId()).containsKey(task_name_lower)) {
//				m = new Mapping<Task, Resource>(current_task.getId() + r2.getId() ,current_task, r2);
//				if  (current_task.isDefined("input_shape")) {
//					//m.setAttribute("cost_of_mapping", MappingCost(m));
//				} else {
//					System.out.println("operation without input shape setting cost to 0 of " + task_name_lower+ "on cpu+gpu");
//					//m.setAttribute("cost_of_mapping", 0.0);
//				}
//
//				all_mappings.add(m);
//				mappings.add(m);
//			}
//			else {
//				System.out.println("operation not found setting cost to 0 of " + task_name_lower + "on cpu+gpu");
//				m = new Mapping<Task, Resource>(current_task.getId() + r2.getId() ,current_task, r2);
//				//m.setAttribute("cost_of_mapping", 0.0);
//				all_mappings.add(m);
//				mappings.add(m);
//
//
//			}
//			
//			if  (op_costs.OpCost.get(r3.getId()).containsKey(task_name_lower))  {
//				m = new Mapping<Task, Resource>(current_task.getId() + r3.getId() ,current_task, r3);
//				all_mappings.add(m);
//				mappings.add(m);
//			}
//			
//			//Coral edge tpu 
//			
//			
//			/*
//			//NCS2
//			if (op_costs.OpCost.get(r1.getId()).containsKey(task_name_lower)){
//				m = new Mapping<Task, Resource>(current_task.getId() + r1.getId() ,current_task, r1);
//				if (current_task.isDefined("input_shape")) {
//					//m.setAttribute("cost_of_mapping", MappingCost(m));
//					all_mappings.add(m);
//					mappings.add(m);
//				}
//			}
//
//			//cpu+gpu
//			if  (op_costs.OpCost.get(r2.getId()).containsKey(task_name_lower)) {
//				m = new Mapping<Task, Resource>(current_task.getId() + r2.getId() ,current_task, r2);
//				if  (current_task.isDefined("input_shape")) {
//					//m.setAttribute("cost_of_mapping", MappingCost(m));
//				} else {
//					System.out.println("operation without input shape setting cost to 0 of " + task_name_lower+ "on cpu+gpu");
//					//m.setAttribute("cost_of_mapping", 0.0);
//				}
//
//				all_mappings.add(m);
//				mappings.add(m);
//			}
//			else {
//				System.out.println("operation not found setting cost to 0 of " + task_name_lower + "on cpu+gpu");
//				m = new Mapping<Task, Resource>(current_task.getId() + r2.getId() ,current_task, r2);
//				//m.setAttribute("cost_of_mapping", 0.0);
//				all_mappings.add(m);
//				mappings.add(m);
//
//
//			}
//			
//			//Coral
//			if  (op_costs.OpCost.get(r3.getId()).containsKey(task_name_lower))  {
//				m = new Mapping<Task, Resource>(current_task.getId() + r3.getId() ,current_task, r3);
//				if  (current_task.isDefined("input_shape")) {
//					//m.setAttribute("cost_of_mapping", MappingCost(m));
//					all_mappings.add(m);
//					mappings.add(m);
//				}
//
//			}
//			*/
//			
//		}
//		
//		Specification specification = new Specification(application, architecture, mappings);
//		
//		SpecificationWriter writer = new SpecificationWriter();
//		writer.write(specification, "src/main/resources/generatedspecs/spec_"+ new SimpleDateFormat("yyyy-MM--dd_hh-mm-ss").format(new Date()) + ".xml");
//
//		/*
//		 * It is also possible to view the specification in a GUI.
//		 */
//		//SpecificationViewer.view(specification);
//		return specification;	
		return null;
	}
	
	/**
	 * 
	 * @param all_tasks
	 * 			The table containing all tasks of the application
	 * @param task_id
	 * 			The task id of the considered task 
	 * @return
	 * 			True if the task is in the table and False ( means the task is not in the application)
	 */
	private boolean quick_search(HashSet<Task> all_tasks, String task_id) {
		boolean dec = false;
		Iterator<Task> it_task = all_tasks.iterator();
		Task current_task=null;
		while (it_task.hasNext()) {
			current_task = it_task.next();
			if (current_task.getId().equals(task_id)) {
				dec = true;
				break;
			}
		}
		//System.out.println(dec + task_id);
		return dec;
		
	}
	
	public Specification getSpecification() {
		return (specification);
	}
	
	
	
}
