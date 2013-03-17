/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * DummyIterator dummy iterator
 * @param <T>
 * @author Josh Attenberg
 */
public class DummyIterator<T> implements Iterator<T> {

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public T next() {
        throw new NoSuchElementException("no elements in dummy iterator");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("remove isnt supported");
    }

}
