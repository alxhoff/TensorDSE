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
	private HashMap<String, OperationCosts> operation_costs = null;
	SpecificationDefinition specification_definition = null;
	private List<Task> starting_tasks;
	private Double K;
	Mappings<Task, Resource> possible_mappings;
	private Boolean verbose;
	private Boolean visualise;
	private Boolean real_time;
	private Boolean hard_real_time;
	private String objective;

	private Integer evaluation_count = 0;

	public EvaluatorMinimizeCost(String objectives, SpecificationDefinition SpecificationDefinition, Double K,
			Boolean real_time, Boolean hard_real_time, String objective, Boolean verbose, Boolean visualise) {
		super();

		this.specification_definition = SpecificationDefinition;
		this.K = K;
		this.operation_costs = SpecificationDefinition.getOperation_costs();
		this.starting_tasks = SpecificationDefinition.getStarting_tasks();
		this.possible_mappings = SpecificationDefinition.getSpecification().getMappings();
		this.verbose = verbose;
		this.visualise = visualise;
		this.real_time = real_time;
		this.hard_real_time = hard_real_time;
		this.objective = objective;

		for (String s : objectives.split(",")) {
			Objective obj = new Objective(s, Objective.Sign.MIN);
			map.put(s, obj);
		}
	}

	/**
	 * @brief Evaluates a solution's specification and sets the cost to the
	 *        objective
	 * @param solution_specification
	 * @param objectives
	 * @return Specification
	 */
	@Override
	public Specification evaluate(Specification solution_specification, Objectives objectives) {

		// Specification for viewing and debugging
		if (this.visualise == true && this.verbose == true)
			SpecificationViewer.view(solution_specification);

		ScheduleSolver schedule_solver = new ScheduleSolver(this.specification_definition,
				solution_specification.getMappings(), this.K, this.verbose, false);

		Double cost_of_mapping = 0.0;
		try {
			cost_of_mapping = schedule_solver.solveGASchedule(getPossible_mappings(), this.real_time,
					this.hard_real_time, this.objective);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		objectives.add(map.get("cost_of_mapping"), cost_of_mapping);

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

	public Map<String, Objective> getMap() {
		return map;
	}

	public void setPriority(int priority) {
		this.priority = priority;
	}

	public HashMap<String, OperationCosts> getOperation_costs() {
		return operation_costs;
	}

	public void setOperation_costs(HashMap<String, OperationCosts> operation_costs) {
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

	public SpecificationDefinition getSpecification_definition() {
		return specification_definition;
	}

	public void setSpecification_definition(SpecificationDefinition specification_definition) {
		this.specification_definition = specification_definition;
	}

	public Double getK() {
		return K;
	}

	public void setK(Double k) {
		K = k;
	}

	public Integer getEvaluation_count() {
		return evaluation_count;
	}

	public void setEvaluation_count(Integer evaluation_count) {
		this.evaluation_count = evaluation_count;
	}

	public Boolean getReal_time() {
		return real_time;
	}

	public void setReal_time(Boolean real_time) {
		this.real_time = real_time;
	}

	public Boolean getHard_real_time() {
		return hard_real_time;
	}

	public void setHard_real_time(Boolean hard_real_time) {
		this.hard_real_time = hard_real_time;
	}
}
