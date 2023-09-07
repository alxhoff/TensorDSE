package net.sf.opendse.TensorDSE;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.optimization.ImplementationEvaluator;
import net.sf.opendse.visualization.SpecificationViewer;

public class EvaluatorMinimizeCost implements ImplementationEvaluator {

	protected final Map<String, Objective> map = new HashMap<String, Objective>();
	protected int priority;
	private OperationCosts operation_costs = null;
	private List<Task> starting_tasks;
	Mappings<Task, Resource> possible_mappings;
	private Boolean verbose;
	private Boolean visualise;

	private Integer evaluation_count = 0;

	public EvaluatorMinimizeCost(String objectives, SpecificationDefinition SpecificationDefinition,
			Boolean verbose, Boolean visualise) {
		super();
		this.operation_costs = SpecificationDefinition.getOperation_costs();
		this.starting_tasks = SpecificationDefinition.getStarting_tasks();
		this.possible_mappings = SpecificationDefinition.getSpecification().getMappings();
		this.verbose = verbose;
		this.visualise = visualise;

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
		// Mappings<Task, Resource> mappings = solution_specification.getMappings();
		// Routings<Task, Resource, Link> routings = solution_specification.getRoutings();

		// Specification for viewing and debugging
		if (this.visualise == true && this.verbose == true)
			SpecificationViewer.view(solution_specification);

		ScheduleSolver schedule_solver = new ScheduleSolver(solution_specification,
				this.starting_tasks, this.operation_costs, this.verbose);

		double cost_of_mapping = schedule_solver.solveGASchedule(getPossible_mappings());

		objectives.add(map.get("cost_of_mapping"), cost_of_mapping);
		/*
		 * try { FileWriter csvOutput = new FileWriter("src/main/resources/perspec_100.csv", true);
		 * csvOutput.append(Double.toString(cost_of_mapping)); csvOutput.append("\n");
		 * csvOutput.flush(); csvOutput.close(); } catch (IOException e) { // TODO Auto-generated
		 * catch block e.printStackTrace(); }
		 */

		System.out.println(String.format("Evaluation #: %d", evaluation_count));
		evaluation_count += 1;

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
	// private double MappingCost(Mapping<Task, Resource> mapping) {
	// Double cost = 0.0;
	// String layer = mapping.getSource().getAttribute("type").toString().toLowerCase();
	// String target_device = mapping.getTarget().getId();
	// String data_type = mapping.getSource().getAttribute("dtype");

	// // Extract target device
	// Pattern p = Pattern.compile("([a-z]+)\\d+");
	// Matcher m = p.matcher(target_device);
	// if (m.find())
	// target_device = m.group(1);

	// // Execution cost
	// if (operation_costs.GetOpTypeTable(target_device).containsKey(layer)) {
	// if (operation_costs.GetOpDataTypeTable(target_device, layer).containsKey(data_type)) {
	// cost = operation_costs.GetOpCost(target_device, layer, data_type);
	// } else {
	// cost = 0.0;
	// }
	// }

	// // Communication cost
	// Pair<Double, Double> comm_cost =
	// operation_costs.GetCommCost(target_device, layer, data_type);

	// cost += comm_cost.getValue0() + comm_cost.getValue1();

	// return cost;
	// }

	public Map<String, Objective> getMap() {
		return map;
	}

	public void setPriority(int priority) {
		this.priority = priority;
	}

	public OperationCosts getOperation_costs() {
		return operation_costs;
	}

	public void setOperation_costs(OperationCosts operation_costs) {
		this.operation_costs = operation_costs;
	}

	public List<Task> getStarting_tasks() {
		return starting_tasks;
	}

	public void setStarting_tasks(List<Task> starting_tasks) {
		this.starting_tasks = starting_tasks;
	}

	public Mappings<Task, Resource> getPossible_mappings() {
		return possible_mappings;
	}

	public void setPossible_mappings(Mappings<Task, Resource> possible_mappings) {
		this.possible_mappings = possible_mappings;
	}

	public Boolean getVerbose() {
		return verbose;
	}

	public void setVerbose(Boolean verbose) {
		this.verbose = verbose;
	}

	public Boolean getVisualise() {
		return visualise;
	}

	public void setVisualise(Boolean visualise) {
		this.visualise = visualise;
	}
}
