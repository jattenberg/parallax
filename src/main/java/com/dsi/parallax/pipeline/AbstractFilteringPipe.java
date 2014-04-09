/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.util.Iterator;

import com.google.common.base.Predicate;
import com.google.common.collect.Iterators;

public abstract class AbstractFilteringPipe<O> implements Pipe<O,O> {

	private static final long serialVersionUID = 7261224792924050612L;
	transient protected PipePredicate predicate;
	
	protected AbstractFilteringPipe() {
		this.predicate = new PipePredicate();
	}
	
	@Override
	public Iterator<Context<O>> processIterator(Iterator<Context<O>> source) {
		return Iterators.filter(source, predicate);
	}
	
	abstract protected boolean operate(Context<O> context);
	
	private class PipePredicate implements Predicate<Context<O>>{

		@Override
		public boolean apply(Context<O> context) {
			return operate(context);
		}

	}
}
