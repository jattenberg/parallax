/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.StringSequenceToNumericDictionaryPipe;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;

public class TestStringSequenceToNumericDictionaryPipe {
    List<String> tokens = Lists.newArrayList("foo", "bar", "baz");

    @Test
    public void test() {
        StringSequenceToNumericDictionaryPipe pipe = new StringSequenceToNumericDictionaryPipe(
                1000000);
        Context<List<String>> context = new Context<List<String>>(tokens);
        @SuppressWarnings("unchecked")
		Iterator<Context<LinearVector>> it = pipe.processIterator(Iterators.forArray(context));
        assertTrue(it.hasNext());
        Context<LinearVector> next = it.next();
        assertTrue(next.getData() instanceof LinearVector);
        LinearVector vect = next.getData();
        assertTrue(vect.size() == 1000000);
        int ct = 0;
        for (@SuppressWarnings("unused") int x : vect)
            ct++;
        assertTrue(ct == tokens.size());
    }

}
