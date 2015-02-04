/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;

import java.util.Iterator;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * The abstract class that implements the common functionality seen in pipes by 
 * implementing the Pipe interface. All the pipe classes should extend this class
 *
 * @author jattenberg
 */
public abstract class AbstractPipe<I,O> implements Pipe<I,O> {

    private static final long serialVersionUID = -1495191273036923098L;
    transient protected PipelineFunction function;
    
    protected AbstractPipe() {
    	function = new PipelineFunction();
    }
    
    // returns an iterator that applies function to each element of the 
    // source iterator, by calling its apply() method on each element.
    // the apply() method in turns calls the operate() method of this class
	@Override
	public Iterator<Context<O>> processIterator(Iterator<Context<I>> source) {
		return Iterators.transform(source, function);
	}
    
	protected void checkValidSizes(int[] in) {
		for(int i : in)
			checkArgument(i>0);
	}
	
	abstract protected Context<O> operate(Context<I> context);

	private class PipelineFunction implements Function<Context<I>, Context<O>> {
		@Override
		// the apply method of this inner class calls the operate() method of the outer 
		// class on the input context. hence every class that extends the outer class 
		// should define an operate() method
		public Context<O> apply(Context<I> context) {
			return operate(context);
		}	
	}
}
