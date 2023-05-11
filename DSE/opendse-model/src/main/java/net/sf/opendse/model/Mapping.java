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
package net.sf.opendse.model;

/**
 * The {@code Mapping} represents a mapping from a {@link Task} to a
 * {@link Resource}.
 * 
 * @author Martin Lukasiewycz
 * 
 * @param <T>
 *            the type of task
 * @param <R>
 *            the type of resource
 */
public class Mapping<T extends Task, R extends Resource> extends Element {

	protected T source;
	protected R target;

	/**
	 * Constructs a new mapping.
	 * 
	 * @param id
	 *            the id
	 * @param source
	 *            the source task
	 * @param target
	 *            the target resource
	 */
	public Mapping(String id, T source, R target) {
		super(id);
		assert (source != null);
		assert (target != null);
		setSource(source);
		setTarget(target);
	}

	/**
	 * Constructs a new mapping
	 * 
	 * @param parent
	 *            the parent node
	 * @param source
	 *            the source
	 * @param target
	 *            the destination
	 */
	public Mapping(Element parent, T source, R target) {
		super(parent);
		assert (source != null);
		assert (target != null);
		setSource(source);
		setTarget(target);
	}

	/**
	 * Returns the source of the mapping.
	 * 
	 * @return the source
	 */
	public T getSource() {
		return source;
	}

	/**
	 * Sets the source of the mapping.
	 * 
	 * @param task
	 *            the source
	 */
	public void setSource(T task) {
		source = task;
	}

	/**
	 * Returns the target of a mapping.
	 * 
	 * @return the target
	 */
	public R getTarget() {
		return target;
	}

	/**
	 * Sets the target of a mapping.
	 * 
	 * @param resource
	 *            the target
	 */
	public void setTarget(R resource) {
		target = resource;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see net.sf.adse.model.Element#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		return super.equals(obj);
	}

}
