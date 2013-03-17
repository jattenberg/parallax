/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

/**
 * an iterator for branching pipes. idea- store yet-to-be iterated over contexts
 * in a queue when no branches need a context, pop the queue. when a branch
 * needs a context that's not in the queue, add to the queue from the context
 * source.
 * 
 * @author jattenberg
 * 
 */
public class BranchingIteratorMaker<I> {

	private final Iterator<Context<I>> contextSource;
	private final LinkedList<Context<I>> contextQueue;
	private Set<BranchIterator> branchingIterators;

	public BranchingIteratorMaker(Iterator<Context<I>> contextSource) {
		this.contextSource = contextSource;
		contextQueue = Lists.newLinkedList();
		Set<BranchIterator> tmp = Sets.newHashSet();
		branchingIterators = Collections.synchronizedSet(tmp);
	}

	private boolean advance() {
		if (contextSource.hasNext()) {
			contextQueue.add(contextSource.next());
			return true;
		} else
			return false;
	}

	public Iterator<Context<I>> buildBranchIterator() {
		BranchIterator it = new BranchIterator();
		branchingIterators.add(it);
		return it;
	}

	public boolean remove(Iterator<Context<I>> it) {
		if (it instanceof BranchingIteratorMaker.BranchIterator) {
			BranchIterator bit = (BranchIterator) it;
			boolean removed = branchingIterators.remove(bit);
			cleanUp(bit.position);
			return removed;
		}
		return false;
	}

	private void cleanUp() {
		cleanUp(Integer.MAX_VALUE);
	}

	private void cleanUp(int min) {
		for (BranchIterator it : branchingIterators)
			min = it.position < min ? it.position : min;
		for (int i = 0; i < min; i++)
			contextQueue.remove();
		for (BranchIterator it : branchingIterators)
			it.position -= min;
	}

	private class BranchIterator implements Iterator<Context<I>> {

		int position;

		BranchIterator() {
			position = 0;
		}

		@Override
		public boolean hasNext() {
			if (BranchingIteratorMaker.this.contextQueue.size() > position)
				return true;
			else
				return BranchingIteratorMaker.this.advance();
		}

		@Override
		public Context<I> next() {
			synchronized (BranchingIteratorMaker.this) {
				if (!(BranchingIteratorMaker.this.contextQueue.size() > position))
					BranchingIteratorMaker.this.advance();
				Context<I> out = new Context<I>(
						BranchingIteratorMaker.this.contextQueue
								.get(position++));
				BranchingIteratorMaker.this.cleanUp();

				return out;
			}
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
					"branching iterator does not support delete");
		}

	}

}
