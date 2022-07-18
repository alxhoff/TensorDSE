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
package net.sf.opendse.model.parameter;

import net.sf.opendse.model.Element;

/**
 * The {@code ParameterReference} is a reference to a {@link Parameter} of an
 * {@link Element}. This reference is defined by the element {@code id} and the
 * name of the {@code Attribute}.
 * 
 * @author Martin Lukasiewycz
 * 
 */
public class ParameterReference implements Comparable<ParameterReference> {

	protected final String id;
	protected final String attribute;
	protected final Parameter parameter;

	/**
	 * Constructs a {@code ParameterReference}.
	 * 
	 * @param element
	 *            the element
	 * @param attribute
	 *            the attribute name
	 */
	public ParameterReference(Element element, String attribute) {
		this(element.getId(), attribute, element.getAttributeParameter(attribute));
	}

	/**
	 * Constructs a {@code ParameterReference}.
	 * 
	 * @param id
	 *            the element id
	 * @param attribute
	 *            the attribute name
	 * @param parameter
	 *            the parameter
	 */
	public ParameterReference(String id, String attribute, Parameter parameter) {
		super();
		this.id = id;
		this.attribute = attribute;
		this.parameter = parameter;
	}

	/**
	 * Returns the {@code id} of the referenced {@code Element}.
	 * 
	 * @return the id of the element
	 */
	public String getId() {
		return id;
	}

	/**
	 * Returns the name of the {@code Attribute}.
	 * 
	 * @return the name of the attribute
	 */
	public String getAttribute() {
		return attribute;
	}

	/**
	 * Returns the {@code Parameter}.
	 * 
	 * @return the parameter
	 */
	public Parameter getParameter() {
		return parameter;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((attribute == null) ? 0 : attribute.hashCode());
		result = prime * result + ((id == null) ? 0 : id.hashCode());
		return result;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		ParameterReference other = (ParameterReference) obj;
		if (attribute == null) {
			if (other.attribute != null)
				return false;
		} else if (!attribute.equals(other.attribute))
			return false;
		if (id == null) {
			if (other.id != null)
				return false;
		} else if (!id.equals(other.id))
			return false;
		return true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(ParameterReference o) {
		int v = id.compareTo(o.id);
		if (v != 0)
			return v;
		v = attribute.compareTo(o.attribute);
		return v;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		return "[id=" + id + ", attribute=" + attribute + "]";
	}

}
