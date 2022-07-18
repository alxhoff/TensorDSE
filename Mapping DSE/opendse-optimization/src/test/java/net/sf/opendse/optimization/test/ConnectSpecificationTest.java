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
package net.sf.opendse.optimization.test;

import net.sf.opendse.io.SpecificationReader;
import net.sf.opendse.io.SpecificationWriter;
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
import net.sf.opendse.model.parameter.Parameters;
import net.sf.opendse.optimization.constraints.ElementList;
import net.sf.opendse.optimization.constraints.SpecificationConstraints;
import net.sf.opendse.optimization.encoding.SingleImplementation;
import net.sf.opendse.visualization.SpecificationViewer;

public class ConnectSpecificationTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		withCommunication();
	}

	public static void noCommunication() {

		// 1. Application
		Application<Task, Dependency> application = new Application<Task, Dependency>();
		Task t1 = new Task("t1");
		Task t2 = new Task("t2");
		application.addVertex(t1);
		application.addVertex(t2);
		application.addEdge(new Dependency("d1"), t1, t2);
		// 2. Architecture
		Architecture<Resource, Link> architecture = new Architecture<Resource, Link>();
		Resource r1 = new Resource("r1");
		r1.setAttribute("costs", 100);
		Resource r2 = new Resource("r2");
		r2.setAttribute("costs", 50);
		Link l1 = new Link("l1");
		architecture.addVertex(r1);
		architecture.addVertex(r2);
		architecture.addEdge(l1, r1, r2);
		// 3. Mappings
		Mappings<Task, Resource> mappings = new Mappings<Task, Resource>();
		Mapping<Task, Resource> m1 = new Mapping<Task, Resource>("m1", t1, r1);
		Mapping<Task, Resource> m2 = new Mapping<Task, Resource>("m2", t2, r2);
		mappings.add(m1);
		mappings.add(m2);
		// 4. Routings
		Routings<Task, Resource, Link> routings = new Routings<Task, Resource, Link>();

		Specification specification = new Specification(application, architecture, mappings, routings);

		SpecificationViewer.view(specification);

	}

	public static void withCommunication() {

		// 1. Application
		Application<Task, Dependency> application = new Application<Task, Dependency>();
		Task t1 = new Task("t1");
		t1.setAttribute("memory", 40);
		Communication c1 = new Communication("c1");
		Task t2 = new Task("t2");
		t2.setAttribute("memory", 40);
		application.addVertex(t1);
		application.addVertex(t2);
		application.addEdge(new Dependency("d1"), t1, c1);
		application.addEdge(new Dependency("d2"), c1, t2);

		// 2. Architecture
		Architecture<Resource, Link> architecture = new Architecture<Resource, Link>();
		Resource r1 = new Resource("r1");
		r1.setAttribute("costs", 100);
		r1.setAttribute("memory" + SpecificationConstraints.CAPACITY_MAX, 128);
		r1.setAttribute("GPIO" + SpecificationConstraints.CONNECT_MAX, Parameters.select(1, 1, 2, 3, 4, 5, 6));
		Resource r2 = new Resource("r2");
		r2.setAttribute("costs", 50);
		r2.setAttribute("variant", Parameters.select("alpha", "alpha", "beta", "gamma"));
		r2.setAttribute("memory" + SpecificationConstraints.CAPACITY_MAX, 64);
		r2.setAttribute("GPIO" + SpecificationConstraints.CONNECT_MAX, Parameters.selectRef("variant", 1, 1, 2, 3));
		r2.setAttribute("GPIO" + SpecificationConstraints.CONNECT_MIN, Parameters.selectRef("variant", 1, 1, 2, 3));
		Resource r3 = new Resource("r3");
		r3.setAttribute("costs", 50);
		r3.setAttribute("memory" + SpecificationConstraints.CAPACITY_MAX, Parameters.select(64, 128, 196));
		r3.setAttribute("GPIO" + SpecificationConstraints.CONNECT_MAX, 2);

		Link l12 = new Link("l1-2");
		Link l11 = new Link("l1-1");
		Link l13 = new Link("l1-3");
		Link l2 = new Link("l2");
		l12.setType("GPIO:1");
		l11.setType("GPIO:2");
		l13.setType("GPIO:3");
		
		addAtMostOne(l12,l11,l13);
		
		l2.setType("GPIO:2");
		architecture.addVertex(r1);
		architecture.addVertex(r2);
		
		architecture.addEdge(l12, r1, r2);
		architecture.addEdge(l11, r1, r2);
		architecture.addEdge(l13, r1, r2);
		architecture.addEdge(l2, r1, r3);

		// 3. Mappings
		Mappings<Task, Resource> mappings = new Mappings<Task, Resource>();
		//Mapping<Task, Resource> m1 = new Mapping<Task, Resource>("m1", t1, r1);
		Mapping<Task, Resource> m2 = new Mapping<Task, Resource>("m2", t1, r2);
		Mapping<Task, Resource> m3 = new Mapping<Task, Resource>("m3", t2, r3);

		// mappings.add(m1);
		mappings.add(m2);
		mappings.add(m3);

		Specification specification = new Specification(application, architecture, mappings);

		SpecificationWriter writer = new SpecificationWriter();
		SpecificationReader reader = new SpecificationReader();
		
		writer.write(specification, "spec.xml");
		specification = reader.read("spec.xml");
		
		SingleImplementation single = new SingleImplementation();
		specification = single.get(specification, true);

		SpecificationViewer.view(specification);

	}
	
	protected static void addAtMostOne(Element... elements){
		ElementList list = ElementList.elements(elements);
		
		for(Element element: elements){
			element.setAttribute(SpecificationConstraints.ELEMENTS_EXCLUDE, list.without(element));
		}
	}

}
