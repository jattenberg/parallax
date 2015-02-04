/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

/**
 * counts the number of examples passed through
 *
 *  @author jattenberg
 */
public class CountingPipe<I> extends AbstractAccumulatingPipe<I, Integer>{

    private static final long serialVersionUID = -2250215264366058830L;
    private int size = 0;
    
    public CountingPipe() {
    	super(-1);
    }
    
    @SuppressWarnings("unchecked")
	@Override
    public Iterator<Context<Integer>> batchProcess(
            Iterator<Context<I>> source) {
        int ct = 0;
        while(source.hasNext()) {
            ct++;
            @SuppressWarnings("unused")
            Context<I> tmp = source.next();
        }
        isTrained = false;
        Context<Integer> out = Context.createContext(ct);
        return Iterators.forArray(out);   
    }

    /**
     * The method returns the class's Type "CountingPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<CountingPipe<I>>(){}.getType();
	}


	@Override
	protected Context<Integer> operate(Context<I> context) {
		return Context.createContext(size);
	}

	@Override
	protected void batchProcess(List<Context<I>> infoList) {
		size = infoList.size();
	}
}
