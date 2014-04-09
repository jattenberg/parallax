/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.Iterator;

/**
 * The interface that any pipe should implement. 
 * @author jattenberg
 *
 * @param <I> the type of input iterator
 * @param <O> the type of output iterator
 */
public interface Pipe<I,O>  extends Serializable {
    
    /**
     * the method used to to flow instances through
     * similar to flat map in functional programming.
     * typically a collapsing iterator processing through {@link processIterator }
     * @param source iterator of input contexts
     * @return interator of output contexts 
     */
    public Iterator<Context<O>> processIterator(Iterator<Context<I>> source);
    
    
    /**
     * Type, used for serialization
     */    
    public Type getType();
}
