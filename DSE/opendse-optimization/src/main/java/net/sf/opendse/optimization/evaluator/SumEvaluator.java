/*******************************************************************************
 * Copyright (c) 2015 OpenDSE
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/
package net.sf.opendse.optimization.evaluator;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import net.sf.opendse.model.Architecture;
import net.sf.opendse.model.Element;
import net.sf.opendse.model.Link;
import net.sf.opendse.model.Mappings;
import net.sf.opendse.model.Resource;
import net.sf.opendse.model.Specification;
import net.sf.opendse.model.Task;
import net.sf.opendse.optimization.ImplementationEvaluator;

import org.opt4j.core.Objective;
import org.opt4j.core.Objectives;

public class SumEvaluator implements ImplementationEvaluator {

	protected final Map<String, Objective> map = new HashMap<String, Objective>();

	protected int priority;

	public SumEvaluator(String sum, int priority, boolean min) {
		super();
		for (String s : sum.split(",")) {
			Objective obj = new Objective(s, min?Objective.Sign.MIN:Objective.Sign.MAX);
			map.put(s, obj);
		}
		this.priority = priority;
	}

	@Override
	public Specification evaluate(Specification implementation, Objectives objectives) {

		Architecture<Resource, Link> architecture = implementation.getArchitecture();
		Mappings<Task, Resource> mappings = implementation.getMappings();

		Set<Element> elements = new HashSet<Element>();
		elements.addAll(architecture.getVertices());
		elements.addAll(architecture.getEdges());
		elements.addAll(mappings.getAll());

		for (Entry<String, Objective> entry : map.entrySet()) {
			double value = 0;

			String attribute = entry.getKey();
			Objective objective = entry.getValue();

			for (Element e : elements) {
				for (String attributeName : e.getAttributeNames()) {
					if ((attributeName.contains(".") && attributeName.substring(0, attributeName.indexOf(".")).equals(
							attribute))
							|| attributeName.equals(attribute)) {
						value += ((Number) e.getAttribute(attributeName)).doubleValue();
					}
				}
			}

			objectives.add(objective, value);
		}

		return null;
	}

	@Override
	public int getPriority() {
		return priority;
	}

}
