package net.sf.opendse.TensorDSE;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;

import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Dependency;
import net.sf.opendse.model.Element;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mapping;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Routings;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.optimization.ImplementationEvaluator;

public class TensorDSEEvaluator implements ImplementationEvaluator {

	protected final Map<String, Objective> map = new HashMap<String, Objective>();
	protected int priority;
	private OperationCosts operation_costs = null;

	public TensorDSEEvaluator(String objectives, SpecificationDefinition SpecificationDefinition) {
		super();
		this.operation_costs = SpecificationDefinition.GetOpCosts();

		for (String s : objectives.split(",")) {
			Objective obj = new Objective(s, Objective.Sign.MIN);
			map.put(s, obj);
		}

	}

	@Override
	public Specification evaluate(Specification impl, Objectives objectives) {

		Architecture<Resource, Link> architecture = impl.getArchitecture();
		Mappings<Task, Resource> mappings = impl.getMappings();
		Routings<Task, Resource, Link> routings = impl.getRoutings();
		Set<Element> elements = new HashSet<Element>();
		elements.addAll(architecture.getVertices());
		elements.addAll(architecture.getEdges());
		elements.addAll(mappings.getAll());
		Application<Task, Dependency> app = impl.getApplication();
		double cost_of_mapping = 0.0;

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
				cost_of_mapping =
						cost_of_mapping + ((Double) link_n.getAttribute("cost")).doubleValue();
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

	@Override
	public int getPriority() {
		return priority;
	}

	/**
	 * This {@code MappingCost} calculates the mapping cost for the considered
	 * mapping
	 * 
	 * @param mapping
	 *                Mapping<Task, Resource>
	 * @return
	 *         a double with the cost of mapping.
	 */
	private double MappingCost(Mapping<Task, Resource> mapping) {
		Double cost = 0.0;
		String layer = mapping.getSource().getAttribute("type").toString().toLowerCase();
		String target_device = mapping.getTarget().getId();
		Integer input_shape = ((Integer) (mapping.getSource().getAttribute("input_shape"))).intValue();
		String data_type = mapping.getTarget().getAttribute("input_type");

		// Execution cost
		if (operation_costs.GetOpTypeTable(target_device).containsKey(layer)) {
			if (operation_costs.GetOpDataTypeTable(target_device, layer).containsKey(input_shape)) {
				cost = operation_costs.GetOpCost(target_device, layer, data_type);
			} else {
				cost = 0.0;
			}
		}

		//Communication cost
		Double comm_cost = operation_costs.GetCommCost(target_device, layer, data_type);
		cost += comm_cost;

		return cost;
	}
}