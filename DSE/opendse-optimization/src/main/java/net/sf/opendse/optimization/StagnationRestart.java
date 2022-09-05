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

import org.opt4j.core.Individual;
import org.opt4j.core.IndividualSet;
import org.opt4j.core.IndividualSetListener;
import org.opt4j.core.common.archive.CrowdingArchive;
import org.opt4j.core.optimizer.Archive;
import org.opt4j.core.optimizer.OptimizerIterationListener;
import org.opt4j.core.optimizer.Population;
import org.opt4j.core.start.Constant;

import com.google.inject.Inject;

/**
 * A class that clears the population if no new individuals (individuals that
 * are either dominated or dominate other population members) came into the
 * population for a defined number of iterations. This state, where no new
 * individuals are generated for a defined period of time is considered as a
 * stagnation. The population is then cleared and new individuals are created
 * from scratch to prevent being stuck in a local optimum.
 * 
 * @author lukasiewycz
 *
 */
public class StagnationRestart implements IndividualSetListener, OptimizerIterationListener {

	protected final Archive archive = new CrowdingArchive(100);
	protected final Population population;

	protected int iteration = 0;
	protected int lastUpdate = 0;
	protected final int diff;

	@Inject
	public StagnationRestart(Population population,
			@Constant(value = "maximalNumberStagnatingGenerations", namespace = StagnationRestart.class) int diff) {
		this.population = population;
		this.diff = diff;
	}

	@Override
	public void iterationComplete(int iteration) {
		this.iteration = iteration;

		for (Individual in0 : population) {
			for (Individual in1 : archive) {
				if (in0.getObjectives().dominates(in1.getObjectives())) {
					// new individuals are found
					lastUpdate = iteration;
				}
			}
		}

		archive.update(population);

		if (iteration - lastUpdate > diff) {
			// the case where no individuals were found for the last diff
			// generations
			lastUpdate = iteration;
			archive.clear();
			population.clear();
		}

	}

	@Override
	public void individualAdded(IndividualSet collection, Individual individual) {
		// No reaction needed
	}

	@Override
	public void individualRemoved(IndividualSet collection, Individual individual) {
		// No reaction needed
	}

}
