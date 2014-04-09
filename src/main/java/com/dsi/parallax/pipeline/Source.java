/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.Iterator;

/**
 * The interface for the starting end of a pipe. 
 * Every class that implements this interface should define two functions
 *  
 * @author jattenberg
 *
 * 
 */
public interface Source<O> extends Serializable {
   
	/**
	 * Function that returns the generic iterator over the context
	 * @return iterator over the Context of some type
	 */
    public Iterator<Context<O>> provideData();
    
    /**
     * 
     * @return returns the type of Source
     */
    public Type getType();
}
