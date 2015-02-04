/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;

import java.util.Iterator;

/**
 * class for pipes producing higher cardinality output than input. 
 *
 * @author jattenberg
 */
public abstract class AbstractExpandingPipe<I, O> implements Pipe<I,O> {

	private static final long serialVersionUID = 7429592274335451246L;
	protected transient ExpandingPipelineFunction function;
    protected transient ContextAddingFunction<O> contextAddingFunction;

    /**
     * Class constructor
     */
	protected AbstractExpandingPipe() {
		function = new ExpandingPipelineFunction();
        contextAddingFunction = new ContextAddingFunction<O>();
	}
	
	// returns an iterator that applies function to each element of the 
    // source iterator, by calling its apply() method on each element.
    // the apply() method in turns calls the operate() method of this class
	@Override
	public Iterator<Context<O>> processIterator(Iterator<Context<I>> source) {
		return Iterators.concat(Iterators.transform(source, function));
	}
	
	abstract protected Iterator<Context<O>> operate(Context<I> context);

	private class ExpandingPipelineFunction implements Function<Context<I>, Iterator<Context<O>>> {
		
		// the apply method of this inner class calls the operate() method of the outer 
		// class on the input context. hence every class that extends the outer class 
		// should define an operate() method
		@Override
		public Iterator<Context<O>> apply(Context<I> context) {
			return operate(context);
		}	
	}
}
