package net.sf.opendse.TensorDSE;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import java.util.regex.Pattern;
import java.util.regex.Matcher;
import org.javatuples.Pair;
import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.optimization.ImplementationEvaluator;
import net.sf.opendse.visualization.SpecificationViewer;

public class EvaluatorMinimizeCost implements ImplementationEvaluator {

	protected final Map<String, Objective> map = new HashMap<String, Objective>();
	protected int priority;
	private OperationCosts operation_costs = null;
	public List<Task> starting_tasks;
	public HashMap<Integer, HashMap<Integer, Task>> tasks;
	// public Integer longest_model;

	public EvaluatorMinimizeCost(String objectives,
			SpecificationDefinition SpecificationDefinition) {
		super();
		this.operation_costs = SpecificationDefinition.GetOperationCosts();
		this.starting_tasks = SpecificationDefinition.getStarting_tasks();
		// this.longest_model = SpecificationDefinition.longest_model;
		this.tasks = SpecificationDefinition.getApplication_graphs();

		for (String s : objectives.split(",")) {
			Objective obj = new Objective(s, Objective.Sign.MIN);
			map.put(s, obj);
		}
	}

	/**
	 * @brief Evaluates a solution's specification and sets the cost to the objective
	 * @param solution_specification
	 * @param objectives
	 * @return Specification
	 */
	@Override
	public Specification evaluate(Specification solution_specification, Objectives objectives) {

		// Pieces that comprise the solution's specification
		// Architecture<Resource, Link> architecture = solution_specification.getArchitecture();
		// Application<Task, Dependency> application = solution_specification.getApplication();
		Mappings<Task, Resource> mappings = solution_specification.getMappings();
		Routings<Task, Resource, Link> routings = solution_specification.getRoutings();

		double cost_of_mapping = 0.0;

		// Specification for viewing and debugging
		// Specification specification = new Specification(application, architecture, mappings);
		SpecificationViewer.view(solution_specification);
		ScheduleSolver schedule_solver = new ScheduleSolver(solution_specification, this.tasks,
				this.starting_tasks, this.operation_costs);
		schedule_solver.solveDSESchedule();

		for (Mapping<Task, Resource> m : mappings) {
			Task current_task = m.getSource();
			if (current_task.isDefined("input_shape")) {
				cost_of_mapping = cost_of_mapping + MappingCost(m);
			}
		}

		for (Architecture<Resource, Link> r : routings.getRoutings()) {

			Iterator<Link> routing_it = r.getEdges().iterator();
			while (routing_it.hasNext()) {
				Link link_n = routing_it.next();
				Double link_cost = link_n.getAttribute("cost");
				cost_of_mapping = cost_of_mapping + link_cost;
			}

		}

		objectives.add(map.get("cost_of_mapping"), cost_of_mapping);
		/*
		 * try { FileWriter csvOutput = new FileWriter("src/main/resources/perspec_100.csv", true);
		 * csvOutput.append(Double.toString(cost_of_mapping)); csvOutput.append("\n");
		 * csvOutput.flush(); csvOutput.close(); } catch (IOException e) { // TODO Auto-generated
		 * catch block e.printStackTrace(); }
		 */
		return null;

	}


	/**
	 * @return int
	 */
	@Override
	public int getPriority() {
		return priority;
	}

	/**
	 * This {@code MappingCost} calculates the mapping cost for the considered mapping
	 * 
	 * @param mapping Mapping<Task, Resource>
	 * @return a double with the cost of mapping.
	 */
	private double MappingCost(Mapping<Task, Resource> mapping) {
		Double cost = 0.0;
		String layer = mapping.getSource().getAttribute("type").toString().toLowerCase();
		String target_device = mapping.getTarget().getId();
		String data_type = mapping.getSource().getAttribute("dtype");

		// Extract target device
		Pattern p = Pattern.compile("([a-z]+)\\d+");
		Matcher m = p.matcher(target_device);
		if (m.find())
			target_device = m.group(1);

		// Execution cost
		if (operation_costs.GetOpTypeTable(target_device).containsKey(layer)) {
			if (operation_costs.GetOpDataTypeTable(target_device, layer).containsKey(data_type)) {
				cost = operation_costs.GetOpCost(target_device, layer, data_type);
			} else {
				cost = 0.0;
			}
		}

		// Communication cost
		Pair<Double, Double> comm_cost =
				operation_costs.GetCommCost(target_device, layer, data_type);

		cost += comm_cost.getValue0() + comm_cost.getValue1();

		return cost;
	}
}
