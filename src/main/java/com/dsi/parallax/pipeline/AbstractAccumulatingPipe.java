/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;

public abstract class AbstractAccumulatingPipe<I, O> implements
		AccumulatingPipe<I, O> {

	private static final long serialVersionUID = -8338935023938996230L;
	protected boolean isTrained;
	protected final int readForInitialization;
	transient protected PipelineFunction function;
	protected transient UncontextifyingFunction<I> uncontextifier;

	/**
	 * constructor for abstract
	 * 
	 * @param readForInitialization
	 */
	public AbstractAccumulatingPipe(int readForInitialization) {
		super();
		this.readForInitialization = readForInitialization;
		this.uncontextifier = new UncontextifyingFunction<I>();
		function = new PipelineFunction();
		isTrained = false;
	}

	@Override
	public Iterator<Context<O>> processIterator(Iterator<Context<I>> source) {
		if (isTrained()) {
			return Iterators.transform(source, function);
		} else {
			return batchProcess(source);
		}
	}

	@Override
	public boolean isTrained() {
		return isTrained;
	}

	protected abstract Context<O> operate(Context<I> context);

	protected abstract void batchProcess(List<Context<I>> infoList);

	@Override
	public Iterator<Context<O>> batchProcess(Iterator<Context<I>> source) {
		Iterator<Context<I>> information = readForInitialization > 0 ? Iterators
				.limit(source, readForInitialization) : source;
		List<Context<I>> infoList = Lists.newArrayList(information);
		batchProcess(infoList);
		return Iterators.transform(
				Iterators.concat(infoList.iterator(), source), function);

	}

	private class PipelineFunction implements Function<Context<I>, Context<O>> {
		@Override
		public Context<O> apply(Context<I> context) {
			return operate(context);
		}
	}

}
