/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.Collection;
import java.util.Iterator;

/**
 * accepts the output of several pipes emitting type I,
 * combines into single stream of type O
 * @author jattenberg
 *
 * @param <I>
 * @param <O>
 */
public interface Combiner<I,O> extends Serializable{

    public Iterator<Context<O>> combine(Collection<Iterator<Context<I>>> branches);
    Context<O> combineExamples(Collection<Context<I>> branchOutput);
    public Type getType();
}
