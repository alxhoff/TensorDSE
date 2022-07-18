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

import net.sf.opendse.optimization.ImplementationEvaluator;

import org.opt4j.core.config.annotations.Multi;
import org.opt4j.core.config.annotations.Order;

import com.google.inject.multibindings.Multibinder;

@Multi
public class SumEvaluatorModule extends EvaluatorModule {

	@Order(0)
	protected String sum = "area,power";
	@Order(2)
	protected int priority = 0;
	@Order(1)
	protected Type type = Type.MIN;
	
	public enum Type {
		MIN,MAX;
	}
	
	public String getSum() {
		return sum;
	}

	public void setSum(String sum) {
		this.sum = sum;
	}

	public Type getType() {
		return type;
	}

	public void setType(Type type) {
		this.type = type;
	}

	public int getPriority() {
		return priority;
	}

	public void setPriority(int priority) {
		this.priority = priority;
	}

	@Override
	protected void config() {
		SumEvaluator evaluator = new SumEvaluator(sum, priority, type == Type.MIN);
		
		Multibinder<ImplementationEvaluator> multibinder = Multibinder.newSetBinder(binder(),
				ImplementationEvaluator.class);
		multibinder.addBinding().toInstance(evaluator);
	}

}
