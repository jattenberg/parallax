/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.util.Iterator;

/**
 * accepts the output of a pipe and returns objects
 *
 * @param <I>
 * @param <O>
 * @author jattenberg
 */
public interface Sink<I,O> extends Iterator<O>{
    
    public void setSource(Pipeline<?,I> pipeline);
    public void setSource(Iterator<Context<I>> source);
    public O next();
    public boolean hasNext();
}
