/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.dsi.parallax.ml.util.DefuturingFunction;
import com.google.common.collect.Collections2;
import com.google.common.collect.Maps;

public abstract class AbstractBranchingPipe<I, C, O> implements
		BranchingPipe<I, C, O> {

	private static final long serialVersionUID = 392650399714337862L;
	private Map<Pipe<I, C>, IteratorCallable> branches;
	private Combiner<C, O> combiner;
	private BranchingIteratorMaker<I> bIteratorMaker;
	private BranchingPipeIterator branchIterator;
	private final DefuturingFunction<Context<C>> defuturer = DefuturingFunction.createDefuturingFunction();
	private final ExecutorService executor;
	private final static int THREADS = 3;
	
	protected AbstractBranchingPipe() {
		branches = Maps.newHashMap();
		executor = Executors.newFixedThreadPool(THREADS);
	}

	@Override
	public Iterator<Context<O>> processIterator(Iterator<Context<I>> context) {
		bIteratorMaker = new BranchingIteratorMaker<I>(context);
		branchIterator = new BranchingPipeIterator(context);
		return branchIterator;
	}

	/**
	 * adds a branch to the pipe. all data is run through all the branches
	 * present. note that the branches are internally backed by a set. this
	 * requires that each
	 */
	@Override
	public void addBranch(Pipe<I, C> pipe) {
		branches.put(
				pipe,
				null == bIteratorMaker ? null : new IteratorCallable(pipe
						.processIterator(bIteratorMaker.buildBranchIterator())));
	}

	@Override
	public boolean removeBranch(Pipe<I, C> pipe) {
		return (null == branches.remove(pipe) ? false : true);
	}

	@Override
	public int size() {
		return branches.size();
	}

	@Override
	public void addCombiner(Combiner<C, O> combiner) {
		this.combiner = combiner;
	}

	private class BranchingPipeIterator implements Iterator<Context<O>> {

		private final Iterator<Context<I>> inputContext;

		BranchingPipeIterator(Iterator<Context<I>> context) {
			inputContext = context;

			for (Pipe<I, C> branch : branches.keySet()) {
				branches.put(
						branch,
						new IteratorCallable(branch
								.processIterator(bIteratorMaker
										.buildBranchIterator())));
			}

		}

		@Override
		public boolean hasNext() {
			return inputContext.hasNext();
		}

		@Override
		public Context<O> next() {
			Collection<Context<C>> intermediate;
			try {
				intermediate = Collections2.transform(executor.invokeAll(branches.values()), defuturer);
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}


			return combiner.combineExamples(intermediate);
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
					"Remove is not supported from branching pipes");
		}
	}

	private class IteratorCallable implements Callable<Context<C>> {

		private final Iterator<Context<C>> iterator;

		public IteratorCallable(Iterator<Context<C>> iterator) {
			this.iterator = iterator;
		}

		@Override
		public Context<C> call() throws Exception {
			return iterator.next();
		}

	}
}
