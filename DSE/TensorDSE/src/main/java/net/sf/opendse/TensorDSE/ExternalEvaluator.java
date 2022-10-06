package net.sf.opendse.TensorDSE;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;

import net.sf.opendse.model.Application;
import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Communication;
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

public class ExternalEvaluator implements ImplementationEvaluator {

	protected final Map<String, Objective> map = new HashMap<String, Objective>();
	protected int priority;
	private OpCosts op_costs = null;

	public ExternalEvaluator(String objectives, FullSpecDef fullspecdef) {
		super();
		this.op_costs = fullspecdef.GetOpCosts();
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
				cost_of_mapping = cost_of_mapping + ((Double) link_n.getAttribute("cost")).doubleValue();
			}

		}

		objectives.add(map.get("cost_of_mapping"), cost_of_mapping);
		/*
		 * try {
		 * FileWriter csvOutput = new FileWriter("src/main/resources/perspec_100.csv",
		 * true);
		 * csvOutput.append(Double.toString(cost_of_mapping));
		 * csvOutput.append("\n");
		 * csvOutput.flush();
		 * csvOutput.close();
		 * } catch (IOException e) {
		 * // TODO Auto-generated catch block
		 * e.printStackTrace();
		 * }
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
		double cost = 0.0;
		String task_name = mapping.getSource().getAttribute("type").toString().toLowerCase();
		String device = mapping.getTarget().getId();
		Integer input_shape = ((Integer) (mapping.getSource().getAttribute("input_shape"))).intValue();
		String input_type = mapping.getTarget().getAttribute("input_type");
		String number_of_shaves = Integer.toString(mapping.getTarget().getAttribute("num_of_shaves"));

		if (op_costs.GetOpMap(device).containsKey(task_name)) {
			if (op_costs.GetTypeMap(device, task_name).containsKey(input_shape)) {
				if (op_costs.GetShaveMap(device, task_name, input_type).containsKey(number_of_shaves)) {

					cost = op_costs.GetOpCoeffucuentA(device, task_name, input_type, number_of_shaves) * input_shape +
							op_costs.GetOpCoeffucuentB(device, task_name, input_type, number_of_shaves);
				}
			} else {
				cost = 0.0;
			}
		}
		return cost;
	}
}