/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.util.Iterator;

public interface AccumulatingPipe<I,O> extends Pipe<I,O> {
    /**
     * assumes that all of the input is needed at once.
     * ex: building document frequency counts in a dataset
     * @param source
     * @return
     */
    public Iterator<Context<O>> batchProcess(Iterator<Context<I>> source);
    public boolean isTrained();
}
