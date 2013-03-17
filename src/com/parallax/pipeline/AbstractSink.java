/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

/** 
 * The AbstractSink class which should be the super class of any sink class. 
 * It implements the Sink interface, and declares and initializes the 
 * uncontextifier function. 
 * 
 * @author spchopra
 *
 * @param <A>
 * @param <B>
 */
public abstract class AbstractSink<A,B> implements Sink<A,B> {
    
	protected transient UncontextifyingFunction<A> uncontextifier;
	
	protected AbstractSink(){
		this.uncontextifier = new UncontextifyingFunction<A>();
	}
    @Override
    public void setSource(Pipeline<?,A> pipeline) {
        setSource(pipeline.process());
    }
    
    @Override
    public void remove() {
        throw new UnsupportedOperationException("remove isnt supported.");
    }
}
