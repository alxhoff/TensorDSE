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
package net.sf.opendse.optimization;

import java.awt.Component;
import java.awt.Point;

import javax.swing.JPanel;
import javax.swing.JPopupMenu;

import net.sf.opendse.model.Specification;
import net.sf.opendse.visualization.SpecificationPanel;

import org.opt4j.core.Individual;
import org.opt4j.viewer.IndividualMouseListener;
import org.opt4j.viewer.Viewport;
import org.opt4j.viewer.Widget;
import org.opt4j.viewer.WidgetParameters;

import com.google.inject.Inject;

public class ImplementationWidgetService implements IndividualMouseListener {

	protected final Viewport viewport;

	@Inject
	public ImplementationWidgetService(Viewport viewport) {
		super();
		this.viewport = viewport;
	}

	@WidgetParameters(title = "Implementation")
	class ImplementationWidget implements Widget {

		protected final Specification implementation;
		
		public ImplementationWidget(Specification implementation){
			this.implementation = implementation;
		}
		
		@Override
		public JPanel getPanel() {
			SpecificationPanel panel = new SpecificationPanel(implementation);
			return panel;
		}

		@Override
		public void init(Viewport viewport) {
		}
	}

	@Override
	public void onDoubleClick(Individual individual, Component component, Point p) {
		ImplementationWrapper wrapper = (ImplementationWrapper)individual.getPhenotype();
		Widget widget = new ImplementationWidget(wrapper.getImplementation());
		viewport.addWidget(widget);
	}

	@Override
	public void onPopup(Individual individual, Component component, Point p, JPopupMenu menu) {

	}

}
