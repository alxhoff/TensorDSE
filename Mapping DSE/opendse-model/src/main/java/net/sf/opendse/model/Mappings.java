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

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.commons.collections15.Bag;
import org.apache.commons.collections15.bag.HashBag;
import org.apache.commons.collections15.functors.InstantiateFactory;
import org.apache.commons.collections15.map.LazyMap;

/**
 * The {@code Mappings} represent a set of {@link Mapping} elements.
 * 
 * @author Martin Lukasiewycz
 * 
 * @param <T>
 *            the type of tasks
 * @param <R>
 *            the type of resources
 */
@SuppressWarnings({ "unchecked", "rawtypes" })
public class Mappings<T extends Task, R extends Resource> implements Iterable<Mapping<T, R>> {

	protected Set<Mapping<T, R>> mappings = new HashSet<Mapping<T, R>>();

	protected Map<T, Set<Mapping<T, R>>> taskMappings = LazyMap.decorate(new HashMap<T, Set<Mapping<T, R>>>(),
			new InstantiateFactory(HashSet.class));
	protected Map<R, Set<Mapping<T, R>>> resourceMappings = LazyMap.decorate(new HashMap<R, Set<Mapping<T, R>>>(),
			new InstantiateFactory(HashSet.class));
	protected Map<T, Bag<R>> targets = LazyMap
			.decorate(new HashMap<T, Bag<R>>(), new InstantiateFactory(HashBag.class));
	protected Map<R, Bag<T>> sources = LazyMap
			.decorate(new HashMap<R, Bag<T>>(), new InstantiateFactory(HashBag.class));

	/**
	 * Adds a mapping.
	 * 
	 * @param mapping
	 *            the mapping to be added
	 */
	public void add(Mapping<T, R> mapping) {
		mappings.add(mapping);
		T source = mapping.getSource();
		R target = mapping.getTarget();

		Set<Mapping<T, R>> s0 = taskMappings.get(source);
		Set<Mapping<T, R>> s1 = resourceMappings.get(target);

		s0.add(mapping);
		s1.add(mapping);

		Bag<R> ts = targets.get(source);
		Bag<T> ss = sources.get(target);

		ts.add(target);
		ss.add(source);
	}

	/**
	 * Removes a mapping
	 * 
	 * @param mapping
	 *            the mapping to be removed
	 * @return {@code true} if the mapping was removed
	 */
	public boolean remove(Mapping<T, R> mapping) {
		boolean exist = mappings.remove(mapping);
		if (exist) {
			T source = mapping.getSource();
			R target = mapping.getTarget();

			Set<Mapping<T, R>> s0 = taskMappings.get(source);
			Set<Mapping<T, R>> s1 = resourceMappings.get(target);

			s0.remove(mapping);
			s1.remove(mapping);

			Bag<R> ts = targets.get(source);
			Bag<T> ss = sources.get(target);

			ts.remove(target);
			ss.remove(source);
		}
		return exist;
	}
	
	public boolean removeAll(Collection<Mapping<T,R>> mappings){
		boolean b = false;
		for(Mapping<T,R> mapping: mappings){
			b |= remove(mapping);
		}
		return b;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Iterable#iterator()
	 */
	@Override
	public Iterator<Mapping<T, R>> iterator() {
		return mappings.iterator();
	}

	/**
	 * Returns the number of mappings.
	 * 
	 * @return the number of mappings
	 */
	public int size() {
		return mappings.size();
	}

	/**
	 * Returns all mappings with the specified source.
	 * 
	 * @param source
	 *            the source
	 * @return all mappings with the specified source
	 */
	public Set<Mapping<T, R>> get(T source) {
		return Collections.unmodifiableSet(taskMappings.get(source));
	}

	/**
	 * Returns all targets for a source.
	 * 
	 * @param source
	 *            the source
	 * @return all targets
	 */
	public Set<R> getTargets(T source) {
		return targets.get(source).uniqueSet();
	}

	/**
	 * Returns all mappings with the specified target.
	 * 
	 * @param target
	 *            the target
	 * @return all mappings with the specified target
	 */
	public Set<Mapping<T, R>> get(R target) {
		return Collections.unmodifiableSet(resourceMappings.get(target));
	}

	/**
	 * Returns all sources for a target.
	 * 
	 * @param target
	 *            the target
	 * @return all sources
	 */
	public Set<T> getSources(R target) {
		return sources.get(target).uniqueSet();
	}

	/**
	 * Returns all mappings with the specified source and target.
	 * 
	 * @param source
	 *            the source
	 * @param target
	 *            the target
	 * @return the mappings with the specified source and target
	 */
	public Set<Mapping<T, R>> get(T source, R target) {
		Set<Mapping<T, R>> set = new HashSet<Mapping<T, R>>(get(source));
		set.retainAll(get(target));
		return Collections.unmodifiableSet(set);
	}

	/**
	 * Returns all mappings as a set.
	 * 
	 * @return all mappings as a set
	 */
	public Set<Mapping<T, R>> getAll() {
		return Collections.unmodifiableSet(mappings);
	}

}
