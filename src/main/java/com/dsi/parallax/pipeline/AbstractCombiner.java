/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import com.google.common.collect.Lists;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public abstract class AbstractCombiner<I, O> implements Combiner<I, O> {

    private static final long serialVersionUID = -3332138785714997475L;

    @Override
    public Iterator<Context<O>> combine(Collection<Iterator<Context<I>>> branches) {
        return new CombiningIterator(branches);
    }

    protected class CombiningIterator implements Iterator<Context<O>> {

        private List<Iterator<Context<I>>> branches;

        CombiningIterator(Collection<Iterator<Context<I>>> branches) {
            this.branches = Lists.newArrayList(branches);
        }

        @Override
        public boolean hasNext() {  
            return branches.get(0).hasNext();
        }

        @Override
        public Context<O> next() {
            List<Context<I>> intermediateOutput = Lists.newLinkedList();
            
            for (Iterator<Context<I>> branch : branches) {
                intermediateOutput.add(branch.next());
            }
            return combineExamples(intermediateOutput);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("remove isnt supported");
        }

    }

}
