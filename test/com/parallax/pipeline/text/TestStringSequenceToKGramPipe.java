/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.parallax.pipeline.Context;

public class TestStringSequenceToKGramPipe {
    List<String> tokens = Lists.newArrayList("foo");
    List<String> tokens2 = Lists.newArrayList("foo", "bar", "baz");
    @Test
    public void test() {
        StringSequenceToKShinglePipe pipe = new StringSequenceToKShinglePipe(
                new int[]{2});
        Context<List<String>> context = new Context<List<String>>(tokens);
        @SuppressWarnings("unchecked")
		Iterator<Context<List<String>>> it = pipe.processIterator(Iterators.forArray(context));
        assertTrue(it.hasNext());
        Context<List<String>> next = it.next();
        
        List<String> vect = next.getData();
        int ct = 0;
        for (@SuppressWarnings("unused") String x : vect) {
            ++ct;
        }
        assertEquals(3,ct);
    }

}
