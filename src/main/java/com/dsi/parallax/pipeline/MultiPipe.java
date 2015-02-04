/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;


/**
 * class for stringing together several pipes
 * basically the same thing as a pipeline but without the source
 * @author jattenberg
 *
 */
public class MultiPipe<I,O> implements Pipe<I,O> {

    private static final long serialVersionUID = -3633232603255985225L;
    private List<Pipe<?,?>> pipes;
    
    private MultiPipe(){
        this.pipes = Lists.newLinkedList();
    }
    
    private MultiPipe(List<Pipe<?, ?>> pipes) {
        this.pipes = Lists.newLinkedList(pipes);
    }

    /**
     * The method accepts pipe and builds to MultiPipe
     * @param pipe pipe
     * @param <I>
     * @param <O>
     * @return MultiPipe
     */
    public static <I,O> MultiPipe<I,O> buildMultiPipe(Pipe<I,O> pipe){
        MultiPipe<I,O> mp = new MultiPipe<I,O>();
        mp.pipes.add(pipe);
        return mp;
    }

    /**
     * The method adds pipe to MultiPipe
     * @param pipe pipe
     * @param <T>
     * @return
     */
    public <T> MultiPipe<I,T> addPipe(Pipe<O,T> pipe) {
        MultiPipe<I,T> mp = new MultiPipe<I,T>(pipes);
        mp.pipes.add(pipe);
        return mp;
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public Iterator<Context<O>> processIterator(Iterator<Context<I>> context) {
    	@SuppressWarnings("rawtypes")
        Iterator it = context;
        for (@SuppressWarnings("rawtypes")
        Pipe pipe : pipes) {
            it = pipe.processIterator(it);
        }
        return it;
    }

    /**
     * The method returns the class's Type "MultiPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<MultiPipe<I,O>>(){}.getType();
	}
}
