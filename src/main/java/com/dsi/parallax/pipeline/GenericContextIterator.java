/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import java.util.Iterator;

/**
 * GenericContextIterator rewrites Java.util.Iterator. In the next() method 
 * instead of returning the next item, it returns the context of the next item
 *
 * @param <T>
 * @author Josh Attenberg
 */
public class GenericContextIterator<T> implements Iterator<Context<T>> {

    Iterator<T> iterator;

    /**
     * Class constructor specifying Iterator to create
     * @param iterator Iterator
     */
    public GenericContextIterator(Iterator<T> iterator) {
        this.iterator = iterator;
    }

    /**
     * The method rewrites Iterator's hasNext method
     * @return boolean
     */
    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    /**
     * The method rewrites Iterator's next method by returning a context of the 
     * next item
     * @return Context
     */
    @Override
    public Context<T> next() {
        T t = iterator.next();
        return new Context<T>(t);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("remove isnt supported.");  
    }
}
