/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.util.Iterator;

/**
 * FilteringPipeIterator is interface that defines filter and addFilter method
 *
 * @param <T>
 * @author Josh Attenberg
 */
public interface FilteringPipeIterator<T> extends Iterator<Context<T>> {
    public boolean filter(Context<T> example);
    public void addFilter(PipeFilter<T> filter);
}
